#!/usr/bin/env bash
set -euo pipefail

# Print recent rank-0 scalar records and TensorBoard event locations.
asset_root="${ASSET_ROOT:-/data1/models/lingbot-vla2}"
run_dir="${1:-$asset_root/outputs/competition_clean_4gpu}"
loss_file="$run_dir/checkpoints/loss.jsonl"
env_prefix="${LINGBOT_ENV_PREFIX:-$asset_root/envs/lingbotvla}"

[[ -f "$loss_file" ]] || { echo "Loss file not available yet: $loss_file" >&2; exit 1; }
"$env_prefix/bin/python" - "$loss_file" <<'PY'
import json
import sys

records = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
for row in records[-20:]:
    print(
        f"step={row.get('step')} loss={row.get('loss'):.6f} "
        f"vla={row.get('vla_loss'):.6f} lr={row.get('lr'):.3e} "
        f"grad_norm={row.get('grad_norm')}"
    )
PY
find "$run_dir/runs" -type f -name 'events.out.tfevents.*' -printf 'TensorBoard event: %p\n' 2>/dev/null || true
