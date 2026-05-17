# Source this file (don't exec). Dispatches a list of audit passes across one
# or more GPUs.
#
# Caller must set:
#   PASS_TAGS    bash array of tags (filename stems for .jsonl outputs)
#   PASS_MODELS  bash array of model paths (parallel to PASS_TAGS)
#   PASS_MODES   bash array of audit modes  (parallel to PASS_TAGS)
#   SUBSET       path to the audit subset JSONL
#   RUN_DIR      output dir for per-pass JSONL + per-pass .log
#   EXTRA_ARGS   bash array of additional flags forwarded to run_audit_pass
#                (e.g. --limit / --debug)
#
# GPU selection:
#   SMOKE_GPUS   comma-separated list of GPU ids, e.g. "0,1,2,3,4"  → parallel
#   SMOKE_GPU    single GPU id (default 0)                          → sequential
#
# Parallel mode redirects each pass's stdout+stderr to ${RUN_DIR}/${tag}.log
# (so interleaved output doesn't garble the terminal). Sequential mode keeps
# stderr live so `--debug` is human-readable.

_dispatch_passes() {
  local n_pass="${#PASS_TAGS[@]}"
  if [ "${n_pass}" -eq 0 ]; then
    echo "ERROR: no passes to dispatch (PASS_TAGS empty)" >&2
    return 1
  fi

  local gpus_str="${SMOKE_GPUS:-${SMOKE_GPU:-0}}"
  local -a gpus
  IFS=',' read -ra gpus <<<"${gpus_str}"
  local n_gpu="${#gpus[@]}"

  if [ "${n_gpu}" -le 1 ]; then
    echo ">>> sequential mode on GPU ${gpus[0]}"
    local i
    for (( i = 0; i < n_pass; i++ )); do
      local tag="${PASS_TAGS[$i]}" model="${PASS_MODELS[$i]}" mode="${PASS_MODES[$i]}"
      local out="${RUN_DIR}/${tag}.jsonl"
      if [ -f "${out}" ]; then
        echo ">>> [${tag}] already exists — skipping"
        continue
      fi
      echo ">>> [${tag}] GPU=${gpus[0]} model=${model} mode=${mode}"
      CUDA_VISIBLE_DEVICES="${gpus[0]}" python -m mllmopd.diagnostics.run_audit_pass \
        --subset "${SUBSET}" \
        --model "${model}" \
        --mode "${mode}" \
        --out "${out}" \
        "${EXTRA_ARGS[@]}"
    done
    return 0
  fi

  echo ">>> parallel mode across ${n_gpu} GPU(s): ${gpus_str}"
  local -a pids tags logs
  local i
  for (( i = 0; i < n_pass; i++ )); do
    local tag="${PASS_TAGS[$i]}" model="${PASS_MODELS[$i]}" mode="${PASS_MODES[$i]}"
    local out="${RUN_DIR}/${tag}.jsonl"
    if [ -f "${out}" ]; then
      echo ">>> [${tag}] already exists — skipping"
      continue
    fi
    local gpu="${gpus[$(( i % n_gpu ))]}"
    local log="${RUN_DIR}/${tag}.log"
    echo ">>> [${tag}] GPU=${gpu} model=${model} mode=${mode}  log=${log}"
    (
      CUDA_VISIBLE_DEVICES="${gpu}" python -m mllmopd.diagnostics.run_audit_pass \
        --subset "${SUBSET}" \
        --model "${model}" \
        --mode "${mode}" \
        --out "${out}" \
        "${EXTRA_ARGS[@]}"
    ) >"${log}" 2>&1 &
    pids+=($!)
    tags+=("${tag}")
    logs+=("${log}")
  done

  local fail=0
  for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
      echo ">>> [${tags[$i]}] done"
    else
      echo "ERROR: pass ${tags[$i]} (pid ${pids[$i]}) failed; tail of ${logs[$i]}:" >&2
      tail -n 20 "${logs[$i]}" >&2 || true
      fail=1
    fi
  done

  if [ "${fail}" -ne 0 ]; then
    return 1
  fi
}

_dispatch_passes
