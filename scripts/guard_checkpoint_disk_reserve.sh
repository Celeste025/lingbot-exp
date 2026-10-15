#!/usr/bin/env bash
set -euo pipefail

# Hold a small, reclaimable file while the trainer is between checkpoint saves.
# This is not a filesystem quota: it only prevents unrelated writers from using
# the held capacity until a new checkpoint directory appears. The reserve is
# then released immediately so checkpoint creation has the capacity available.
asset_root="${ASSET_ROOT:-/data1/models/lingbot-vla2}"
checkpoint_root=""
reserve_gib=32
poll_seconds=10
trainer_pid=""

usage() {
  echo "Usage: $0 --checkpoint-root PATH [--reserve-gib N] [--poll-seconds N] [--trainer-pid PID]" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint-root) checkpoint_root="$2"; shift 2 ;;
    --reserve-gib) reserve_gib="$2"; shift 2 ;;
    --poll-seconds) poll_seconds="$2"; shift 2 ;;
    --trainer-pid) trainer_pid="$2"; shift 2 ;;
    *) usage; exit 2 ;;
  esac
done

[[ -n "$checkpoint_root" ]] || { usage; exit 2; }
[[ "$reserve_gib" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid reserve size" >&2; exit 2; }
[[ "$poll_seconds" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid poll interval" >&2; exit 2; }

reserve_file="$asset_root/.lingbot_checkpoint_reserve_${reserve_gib}GiB"
log_dir="$(dirname "$checkpoint_root")/archive_logs"
mkdir -p "$log_dir"
log_file="$log_dir/disk_reserve_guard.log"

log() { printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$log_file"; }
release() {
  [[ -e "$reserve_file" ]] || return 0
  rm -f -- "$reserve_file"
  log "released reserve: $reserve_file"
}
cleanup() { release; }
trap cleanup EXIT INT TERM

latest_step() {
  find "$checkpoint_root" -mindepth 1 -maxdepth 1 -type d -name 'global_step_*' -printf '%f\n' 2>/dev/null \
    | sed -n 's/^global_step_\([0-9][0-9]*\)$/\1/p' | sort -n | tail -1
}

checkpoint_set_settled() {
  local full_count nonfull_count path
  full_count=0
  nonfull_count=0
  while IFS= read -r path; do
    [[ -n "$path" ]] || continue
    if [[ -d "$path/optimizer" ]]; then
      ((full_count+=1))
    else
      ((nonfull_count+=1))
    fi
  done < <(find "$checkpoint_root" -mindepth 1 -maxdepth 1 -type d -name 'global_step_*' -print 2>/dev/null)
  (( full_count == 1 && nonfull_count == 0 ))
}

hold() {
  [[ -e "$reserve_file" ]] && return 0
  # Training writes only small logs between checkpoints. Keep a 20GiB floor
  # while holding the reserve; the full reserve is released on checkpoint
  # directory creation, before the large distributed save begins.
  local avail_bytes reserve_bytes min_free_bytes
  avail_bytes="$(df --output=avail -B1 "$asset_root" | tail -1 | tr -d ' ')"
  reserve_bytes=$((reserve_gib * 1024 * 1024 * 1024))
  min_free_bytes=$((20 * 1024 * 1024 * 1024))
  if (( avail_bytes < reserve_bytes + min_free_bytes )); then
    log "not reserving: available=${avail_bytes} below reserve+20GiB"
    return 0
  fi
  fallocate -l "${reserve_gib}G" "$reserve_file"
  log "held ${reserve_gib}GiB reserve at $reserve_file"
}

last_step="$(latest_step)"
log "started; last_checkpoint=${last_step:-none}, reserve=${reserve_gib}GiB"

while true; do
  if [[ -n "$trainer_pid" ]] && ! kill -0 "$trainer_pid" 2>/dev/null; then
    log "trainer pid $trainer_pid exited; releasing reserve and stopping"
    exit 0
  fi

  current_step="$(latest_step)"
  if [[ -n "$current_step" && "$current_step" != "$last_step" ]]; then
    release
    log "new checkpoint step=$current_step detected; capacity made available"
    last_step="$current_step"
  elif checkpoint_set_settled; then
    hold
  fi
  sleep "$poll_seconds"
done
