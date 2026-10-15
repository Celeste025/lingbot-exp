#!/usr/bin/env bash
set -euo pipefail

# Low-frequency, non-invasive health watchdog for the competition run. It
# records every check and emits a desktop notification only for actionable
# failures; it never kills, restarts, or otherwise alters training.
asset_root="${ASSET_ROOT:-/data1/models/lingbot-vla2}"
run_dir="${RUN_DIR:-$asset_root/outputs/competition_clean_4gpu}"
interval_seconds="${INTERVAL_SECONDS:-5400}"
health_dir="$asset_root/health_checks"
health_log="$health_dir/health.log"
alert_log="$health_dir/alerts.log"
pipeline_log_dir="$asset_root/pipeline_logs"
loss_file="$run_dir/checkpoints/loss.jsonl"

mkdir -p "$health_dir"
[[ "$interval_seconds" =~ ^[0-9]+$ ]] || { echo "Invalid INTERVAL_SECONDS" >&2; exit 2; }

record_alert() {
  local message="$1"
  printf '[%s] ALERT %s\n' "$(date -Is)" "$message" | tee -a "$health_log" "$alert_log" >&2
  if command -v notify-send >/dev/null 2>&1; then
    notify-send 'LingBot training alert' "$message" || true
  fi
}

check_once() {
  local now loss_age avail_bytes last_step issues=()
  now="$(date +%s)"

  # A completed final checkpoint plus a stopped trainer is success, not an
  # alert. Stop the watchdog only after the final loss record is durable.
  if [[ -f "$loss_file" ]]; then
    last_step="$(tail -1 "$loss_file" | sed -n 's/.*"step": *\([0-9][0-9]*\).*/\1/p')"
    if [[ "$last_step" == "16000" ]] && ! pgrep -f 'tasks/vla/train_lingbotvla.py.*competition_clean_4gpu' >/dev/null; then
      printf '[%s] COMPLETE step=16000; stopping health watchdog\n' "$(date -Is)" | tee -a "$health_log"
      return 10
    fi
  fi

  pgrep -f 'tasks/vla/train_lingbotvla.py.*competition_clean_4gpu' >/dev/null || \
    issues+=("training process is not running")
  pgrep -f 'archive_checkpoints_modelscope.sh.*competition_clean_4gpu/checkpoints' >/dev/null || \
    issues+=("ModelScope archive monitor is not running")

  if [[ -f "$loss_file" ]]; then
    loss_age=$((now - $(stat -c %Y "$loss_file")))
    (( loss_age <= 1800 )) || issues+=("loss log has not advanced for ${loss_age}s")
  else
    issues+=("loss log is missing")
  fi

  avail_bytes="$(df -B1 --output=avail "$asset_root" | tail -1 | tr -d ' ')"
  (( avail_bytes >= 60 * 1024 * 1024 * 1024 )) || \
    issues+=("available disk below 60GiB: ${avail_bytes} bytes")

  if find "$pipeline_log_dir" "$run_dir" -type f -mmin -100 \( -name '*.log' -o -name '*.jsonl' \) \
       -exec grep -El 'Traceback \(most recent call last\)|RuntimeError:|CUDA out of memory|No space left on device|ProcessExitedException' {} + \
       2>/dev/null | head -1 | grep -q .; then
    issues+=("recent log contains a fatal-error signature")
  fi

  if (( ${#issues[@]} == 0 )); then
    printf '[%s] OK process=live loss_age=%ss disk_avail=%s checkpoint_dirs=%s\n' \
      "$(date -Is)" "${loss_age:-na}" "$avail_bytes" \
      "$(find "$run_dir/checkpoints" -mindepth 1 -maxdepth 1 -type d -name 'global_step_*' 2>/dev/null | wc -l)" \
      >> "$health_log"
  else
    record_alert "$(IFS='; '; echo "${issues[*]}")"
  fi
}

while true; do
  set +e
  check_once
  status=$?
  set -e
  [[ "$status" -eq 0 ]] || exit "$status"
  sleep "$interval_seconds"
done
