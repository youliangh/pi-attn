// PyTorch binding of the engine's GEMM entry points (pi_launch.cu, compiled in whole) for
// vLLM: linear() is the engine's pi_linear with the engine's tuner in front of it (pi.rs
// tune(), ported here), so vLLM runs exactly the tile choice and kernels the engine runs.
// Every tile gives identical bits; tuning only changes speed.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <pybind11/stl.h>

#include <set>

#include "pi_launch.cu"

namespace {

std::mutex attempted_mutex;
std::set<std::tuple<int, int, int>> attempted;  // (N, K, bucket) tuned or being tuned
int64_t heuristic_calls = 0;                      // calls that ran an untuned bucket's fixed tile
std::set<std::pair<int, int>> seen;               // (N, K) of every weight linear() ran

void check(const char* error) { TORCH_CHECK(error == nullptr, error); }

// pi.rs tune(): the first time (N, K, bucket) is seen outside graph capture, time every
// candidate on scratch input (and, for buckets <= 512, over weight copies in a 2x-L2 flush
// buffer) and remember the fastest. Inside capture the bucket keeps the fixed heuristic.
void tune(const at::Tensor& w, int N, int K, int M, cudaStream_t st) {
  const int bucket = pi_linear_bucket(M);
  if (pi_linear_tuned_kernel(N, K, bucket) != 0) return;
  if (pi_stream_capturing(st)) {
    ++heuristic_calls;
    return;
  }
  {
    std::lock_guard<std::mutex> lock(attempted_mutex);
    if (!attempted.insert({N, K, bucket}).second) return;
  }
  auto x = at::empty({bucket, K}, w.options());
  auto y = at::empty({bucket, N}, w.options());
  const int64_t flush_bytes = bucket <= 512 ? 2 * pi_l2_bytes() : 0;
  at::Tensor flush;
  if (flush_bytes) flush = at::empty({flush_bytes}, w.options().dtype(at::kByte));
  check(pi_linear_tune(w.data_ptr(), N, K, bucket, x.data_ptr(), y.data_ptr(),
                       flush_bytes ? flush.data_ptr() : nullptr, flush_bytes, st));
}

}  // namespace

// y [M, N] bf16 = x [M, K] w [N, K]^T (x rows contiguous, any row stride; w contiguous).
at::Tensor linear(at::Tensor x, at::Tensor w) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && w.scalar_type() == at::kBFloat16);
  TORCH_CHECK(x.dim() == 2 && w.dim() == 2 && x.size(1) == w.size(1) && w.is_contiguous() && x.stride(1) == 1);
  const c10::cuda::CUDAGuard guard(x.device());
  const int M = x.size(0), K = x.size(1), N = w.size(0);
  auto y = at::empty({M, N}, x.options());
  if (M == 0) return y;
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  {
    std::lock_guard<std::mutex> lock(attempted_mutex);
    seen.insert({N, K});
  }
  tune(w, N, K, M, st);
  check(pi_linear(x.data_ptr(), x.stride(0), w.data_ptr(), y.data_ptr(), M, N, K, 0, st));
  return y;
}

// vLLM owns output and scratch storage; the dense page table can have a padded
// row stride. All launch planning depends on host shape bounds, not GPU values.
void check_attention(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                     const at::Tensor& table, const at::Tensor& lens, const at::Tensor& out) {
  TORCH_CHECK(q.is_cuda() && q.scalar_type() == at::kBFloat16 && q.dim() == 3 && q.is_contiguous());
  TORCH_CHECK(k.is_cuda() && v.is_cuda() && k.scalar_type() == at::kBFloat16 &&
              v.scalar_type() == at::kBFloat16 && k.dim() == 4 && k.is_contiguous() && v.is_contiguous());
  TORCH_CHECK(k.sizes() == v.sizes() && q.size(2) == 128 && k.size(3) == 128);
  TORCH_CHECK(k.size(1) > 0 && k.size(2) > 0 && q.size(1) > 0 &&
              q.size(1) % k.size(2) == 0 && q.size(1) / k.size(2) <= 16);
  TORCH_CHECK(out.sizes() == q.sizes() && out.is_contiguous() && out.scalar_type() == at::kBFloat16);
  TORCH_CHECK(table.is_cuda() && table.scalar_type() == at::kInt && table.dim() == 2 && table.stride(1) == 1);
  TORCH_CHECK(lens.is_cuda() && lens.scalar_type() == at::kInt && lens.dim() == 1 && lens.is_contiguous());
  for (const auto& t : {k, v, table, lens, out}) TORCH_CHECK(t.device() == q.device());
}

void decode_out(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor table, at::Tensor lens,
                at::Tensor out, at::Tensor ws_o, at::Tensor ws_ml, at::Tensor counters, double scale) {
  check_attention(q, k, v, table, lens, out);
  const c10::cuda::CUDAGuard guard(q.device());
  TORCH_CHECK(table.size(0) == q.size(0) && lens.size(0) == q.size(0));
  TORCH_CHECK(ws_o.is_contiguous() && ws_o.scalar_type() == at::kFloat && ws_o.dim() == 3 &&
              ws_o.size(1) == q.size(1) && ws_o.size(2) == 128);
  TORCH_CHECK(ws_ml.is_contiguous() && ws_ml.scalar_type() == at::kFloat &&
              ws_ml.numel() >= ws_o.size(0) * q.size(1) * 8);
  TORCH_CHECK(counters.is_contiguous() && counters.scalar_type() == at::kInt &&
              counters.numel() >= ws_o.size(0) * q.size(1) * 4);
  for (const auto& t : {ws_o, ws_ml, counters}) TORCH_CHECK(t.device() == q.device());
  check(pi_attn_decode(q.data_ptr(), k.data_ptr(), v.data_ptr(), table.data_ptr<int>(), table.stride(0),
      lens.data_ptr<int>(), ws_o.data_ptr<float>(), ws_ml.data_ptr<float>(), counters.data_ptr<int>(),
      ws_o.size(0), out.data_ptr(), q.size(0), q.size(1), k.size(2), k.size(1), scale,
      at::cuda::getCurrentCUDAStream()));
}

void prefill_out(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor qo, at::Tensor table,
                 at::Tensor lens, at::Tensor out, at::Tensor ws_o, at::Tensor ws_ml,
                 int max_q, int max_kv, double scale, int forced_splits) {
  check_attention(q, k, v, table, lens, out);
  const c10::cuda::CUDAGuard guard(q.device());
  const int batch = lens.size(0), hq = q.size(1), hkv = k.size(2), page = k.size(1);
  TORCH_CHECK(qo.is_cuda() && qo.device() == q.device() && qo.scalar_type() == at::kInt &&
              qo.dim() == 1 && qo.is_contiguous() && qo.numel() == batch + 1 && table.size(0) == batch);
  if (batch == 0 || q.size(0) == 0) return;
  TORCH_CHECK(max_q > 0 && max_kv >= max_q && max_kv <= table.size(1) * page);
  auto info = device_info();
  int splits = 1, groups = 0;
  pi_attn_prefill_plan(batch, hq, hkv, max_q, max_kv, &splits, &groups);
  if (forced_splits > 0) splits = std::min(forced_splits, groups);
  if (splits > 1) {
    TORCH_CHECK(ws_o.is_contiguous() && ws_ml.is_contiguous() && ws_o.scalar_type() == at::kFloat &&
                ws_ml.scalar_type() == at::kFloat && ws_o.device() == q.device() && ws_ml.device() == q.device());
    TORCH_CHECK(ws_o.numel() >= q.size(0) * hq * groups * 128 && ws_ml.numel() >= q.size(0) * hq * groups * 2);
  }
  const int warps = prefill_warps(info.prefill_qreg, batch, hq, hkv, max_q, info.sms);
  const int tiles = (max_q * (hq / hkv) + warps * 16 - 1) / (warps * 16);
  auto st = at::cuda::getCurrentCUDAStream();
  auto launch = [&](auto kernel, int smem) {
    set_smem(kernel, smem);
    kernel<<<dim3(tiles, hkv, batch * splits), warps * 32, smem, st>>>(
      (const __nv_bfloat16*)q.data_ptr(), (__nv_bfloat16*)out.data_ptr(),
      (const __nv_bfloat16*)k.data_ptr(), (const __nv_bfloat16*)v.data_ptr(),
      qo.data_ptr<int>(), nullptr, table.data_ptr<int>(), table.stride(0), lens.data_ptr<int>(), nullptr,
      hq, hkv, page, page_shift_of(page), (float)(scale * 1.4426950408889634), splits,
      splits > 1 ? ws_o.data_ptr<float>() : nullptr, splits > 1 ? (float2*)ws_ml.data_ptr<float>() : nullptr, groups);
  };
  if (warps == 8) launch(prefill_kernel<true, 8>, prefill_smem_bytes<true, 8>());
  else if (info.prefill_qreg) launch(prefill_kernel<true>, prefill_smem_bytes<true>());
  else launch(prefill_kernel<false>, prefill_smem_bytes<false>());
  check(check_launch("vLLM PI prefill"));
  if (splits > 1) {
    prefill_merge_kernel<<<dim3(hq, q.size(0)), kHeadDim, 0, st>>>(
      ws_o.data_ptr<float>(), (const float2*)ws_ml.data_ptr<float>(), qo.data_ptr<int>(),
      lens.data_ptr<int>(), nullptr, batch, (__nv_bfloat16*)out.data_ptr(), hq, groups);
    check(check_launch("vLLM PI prefill merge"));
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("decode_out", &decode_out);
  m.def("prefill_out", &prefill_out);
  m.def("linear", &linear);
  m.def("bucket", &pi_linear_bucket, "M bucket the tuner keys on");
  m.def("tuned_kernel", &pi_linear_tuned_kernel, "kernel id tuned for (N, K, bucket), 0 = not tuned");
  m.def("shapes", [] {
    std::lock_guard<std::mutex> lock(attempted_mutex);
    return std::vector<std::pair<int, int>>(seen.begin(), seen.end());
  }, "(N, K) of every weight linear() ran");
  m.def("heuristic_calls", [] { return heuristic_calls; }, "calls inside graph capture on an untuned bucket");
  m.def("tuning_status", [] {
    std::vector<std::tuple<int, int, int>> keys;
    {
      std::lock_guard<std::mutex> lock(attempted_mutex);
      keys.assign(attempted.begin(), attempted.end());
    }
    std::vector<std::tuple<int, int, int, int>> result;
    for (auto [n, k, bucket] : keys)
      result.emplace_back(n, k, bucket, pi_linear_tuned_kernel(n, k, bucket));
    return result;
  }, "(N, K, M bucket, kernel id) for each tuning attempt");
}
