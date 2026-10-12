#!/usr/bin/env bash
set -euo pipefail

# Wait for the active trainer to finish, then archive every completed recent
# checkpoint. The final full-state checkpoint is verified in ModelScope but is
# retained locally for immediate evaluation.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
checkpoint_root=""
trainer_pid=""

usage() {
  echo "Usage: $0 --checkpoint-root PATH --trainer-pid PID" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint-root) checkpoint_root="$2"; shift 2 ;;
    --trainer-pid) trainer_pid="$2"; shift 2 ;;
    *) usage; exit 2 ;;
  esac
done

[[ -n "$checkpoint_root" && -n "$trainer_pid" ]] || { usage; exit 2; }
[[ "$trainer_pid" =~ ^[0-9]+$ ]] || { echo "Invalid trainer PID" >&2; exit 2; }

while kill -0 "$trainer_pid" 2>/dev/null; do
  sleep 60
done

# The trainer waits for its asynchronous HF exporter, but leave a short buffer
# for filesystem metadata visibility before finding the final checkpoint.
sleep 30
exec bash "$repo_root/scripts/archive_checkpoints_modelscope.sh" \
  --checkpoint-root "$checkpoint_root" \
  --include-latest-full --keep-latest-full
