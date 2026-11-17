# Source from any directory; setup_env.sh records the selected profile.
export ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [[ -z "${PI_ATTN_PROFILE:-}" && -f "$ROOT/env/profile" ]]; then
  PI_ATTN_PROFILE=$(cat "$ROOT/env/profile")
fi
case "${PI_ATTN_PROFILE:-legacy}" in
  rtx5070) source "$ROOT/scripts/env_rtx5070.sh" ;;
  legacy)
    if [[ -f "$ROOT/env/legacy.sh" ]]; then
      source "$ROOT/env/legacy.sh"
    else
      source "$ROOT/scripts/env_legacy.sh"
    fi
    ;;
  *) echo "Unknown PI_ATTN_PROFILE: $PI_ATTN_PROFILE" >&2; return 2 ;;
esac
