#!/usr/bin/env bash
set -euo pipefail

# Archive completed checkpoints only after ModelScope has accepted an SHA-256
# manifest. Model-only saves are archived after a newer checkpoint exists;
# full-state saves retain the newest local optimizer state for resumption.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
asset_root="${ASSET_ROOT:-/data1/models/lingbot-vla2}"
env_prefix="${LINGBOT_ENV_PREFIX:-$asset_root/envs/lingbotvla}"
checkpoint_root=""
repo_id="${MODELSCOPE_CHECKPOINT_REPO:-Celeste025/lingbot-vla2-robotwin-clean50-checkpoints}"
watch=false
poll_seconds=120
include_latest_full=false
keep_latest_full=false

usage() {
  echo "Usage: $0 --checkpoint-root PATH [--repo-id OWNER/REPO] [--watch] [--poll-seconds N] [--include-latest-full] [--keep-latest-full]" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint-root) checkpoint_root="$2"; shift 2 ;;
    --repo-id) repo_id="$2"; shift 2 ;;
    --watch) watch=true; shift ;;
    --poll-seconds) poll_seconds="$2"; shift 2 ;;
    --include-latest-full) include_latest_full=true; shift ;;
    --keep-latest-full) keep_latest_full=true; include_latest_full=true; shift ;;
    *) usage; exit 2 ;;
  esac
done

[[ -n "$checkpoint_root" ]] || { usage; exit 2; }
[[ -x "$env_prefix/bin/ms-hub" ]] || { echo "ModelScope client missing: $env_prefix/bin/ms-hub" >&2; exit 1; }
[[ -d "$checkpoint_root" ]] || exit 0
[[ "$poll_seconds" =~ ^[0-9]+$ ]] || { echo "Invalid poll interval: $poll_seconds" >&2; exit 2; }

ms_hub="$env_prefix/bin/ms-hub"
log_dir="$(dirname "$checkpoint_root")/archive_logs"
mkdir -p "$log_dir"
log_file="$log_dir/modelscope_archive.log"

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$log_file"
}

step_of() {
  basename "$1" | sed -n 's/^global_step_\([0-9][0-9]*\)$/\1/p'
}

list_ready_checkpoints() {
  find "$checkpoint_root" -mindepth 1 -maxdepth 1 -type d -name 'global_step_*' -print \
    | while IFS= read -r path; do
        step="$(step_of "$path")"
        [[ -n "$step" && -d "$path/hf_ckpt" ]] || continue
        printf '%012d\t%s\n' "$step" "$path"
      done | sort -n
}

write_manifest() {
  local source_dir="$1"
  (
    cd "$source_dir"
    find . -type f ! -name CHECKSUMS.sha256 -print0 | sort -z | xargs -0 -r sha256sum > CHECKSUMS.sha256
  )
}

archive_and_prune() {
  local kind="$1"
  local checkpoint_dir="$2"
  local keep_local="${3:-false}"
  local step source_dir remote_dir verify_dir remote_manifest local_manifest
  step="$(step_of "$checkpoint_dir")"
  if [[ "$kind" == full ]]; then
    source_dir="$checkpoint_dir"
    remote_dir="full_state/global_step_${step}"
  else
    source_dir="$checkpoint_dir/hf_ckpt"
    remote_dir="inference/global_step_${step}"
  fi
  [[ -d "$source_dir" ]] || return 0

  log "uploading $kind checkpoint step=$step from $source_dir"
  write_manifest "$source_dir"
  local_manifest="$source_dir/CHECKSUMS.sha256"
  "$ms_hub" upload "$repo_id" "$source_dir" "$remote_dir" --repo-type model \
    --commit-message "Archive ${kind} checkpoint at step ${step}" --max-workers 1 >>"$log_file" 2>&1

  verify_dir="$(mktemp -d "$asset_root/tmp/modelscope-verify.XXXXXX")"
  trap 'rm -rf -- "$verify_dir"' RETURN
  "$ms_hub" download "$repo_id" "$remote_dir/CHECKSUMS.sha256" --repo-type model \
    --local-dir "$verify_dir" --force --max-workers 1 >>"$log_file" 2>&1
  remote_manifest="$(find "$verify_dir" -type f -name CHECKSUMS.sha256 -print -quit)"
  [[ -n "$remote_manifest" ]] || { log "remote manifest missing for step=$step; keeping local checkpoint"; return 1; }
  cmp --silent "$local_manifest" "$remote_manifest" || {
    log "manifest mismatch for step=$step; keeping local checkpoint"
    return 1
  }
  rm -rf -- "$verify_dir"
  trap - RETURN
  if "$keep_local"; then
    log "verified remote $kind checkpoint step=$step; retaining local $checkpoint_dir"
    return 0
  fi
  log "verified remote $kind checkpoint step=$step; deleting local $checkpoint_dir"
  rm -rf -- "$checkpoint_dir"
}

run_once() {
  mapfile -t ready < <(list_ready_checkpoints)
  (( ${#ready[@]} >= 1 )) || return 0

  local latest_full_path="" line path
  for line in "${ready[@]}"; do
    path="${line#*$'\t'}"
    [[ -d "$path/optimizer" ]] && latest_full_path="$path"
  done

  for line in "${ready[@]}"; do
    path="${line#*$'\t'}"
    if [[ -d "$path/optimizer" ]]; then
      # Keep exactly one newest local full-state checkpoint for recovery.
      if [[ "$path" == "$latest_full_path" && "$include_latest_full" != true ]]; then
        continue
      fi
      if [[ "$path" == "$latest_full_path" && "$keep_latest_full" == true ]]; then
        archive_and_prune full "$path" true || log "archive failed; will retry full checkpoint $path"
      else
        archive_and_prune full "$path" || log "archive failed; will retry full checkpoint $path"
      fi
    else
      # Model-only checkpoints are evaluation artifacts. Once their HF export
      # exists, uploading and verifying it immediately avoids accumulating a
      # 2k-step local backlog before the next full-state checkpoint.
      archive_and_prune inference "$path" || log "archive failed; will retry inference checkpoint $path"
    fi
  done
}

while true; do
  run_once
  "$watch" || break
  sleep "$poll_seconds"
done
