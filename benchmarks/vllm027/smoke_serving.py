import argparse
from serving_utils import server, save, difference, assert_pi_routing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="pi")
    ap.add_argument("--model", default="Qwen3-0.6B")
    ap.add_argument("--config", default="eager")
    a = ap.parse_args()
    with server(a.model, a.mode, a.config, tag="smoke-" + a.config) as (c, folder):
        ids = [9707, 11, 358, 2776, 264, 1657, 13]
        rows = []
        for sampling in ({"temperature": 0}, {"temperature": 0.8, "top_k": 50, "top_p": 0.95, "seed": 42}):
            c.flush()
            first = c.generate(ids, 16, sampling)
            second = c.generate(ids, 16, sampling)
            d = difference(first, second)
            assert d["token_mismatches"] == d["logprob_mismatches"] == 0, d
            rows.append({"sampling": sampling, "output": first, "difference": d})
        if a.mode == "pi":
            assert_pi_routing(c.rpc("pi_status"), a.model)
        save(folder / "checks.json", {"complete": True, "rows": rows})
        print("Serving smoke passed", flush=True)


if __name__ == "__main__":
    main()
