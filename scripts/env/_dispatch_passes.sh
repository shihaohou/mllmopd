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

  # Backend module: HF transformers (default) or sglang. Caller sets
  # AUDIT_BACKEND_MODULE (typically via run_smoke.sh interpreting
  # AUDIT_BACKEND=sglang).
  local audit_module="${AUDIT_BACKEND_MODULE:-mllmopd.diagnostics.run_audit_pass}"

  local gpus_str="${SMOKE_GPUS:-${SMOKE_GPU:-0}}"
  local -a gpus
  IFS=',' read -ra gpus <<<"${gpus_str}"
  local n_gpu="${#gpus[@]}"

  if [ "${n_gpu}" -le 1 ]; then
    echo ">>> sequential mode on GPU ${gpus[0]} (backend: ${audit_module})"
    local i
    for (( i = 0; i < n_pass; i++ )); do
      local tag="${PASS_TAGS[$i]}" model="${PASS_MODELS[$i]}" mode="${PASS_MODES[$i]}"
      local out="${RUN_DIR}/${tag}.jsonl"
      if [ -f "${out}" ]; then
        echo ">>> [${tag}] already exists — skipping"
        continue
      fi
      local -a pass_extra=()
      if [ -n "${PASS_SYSTEM_PROMPTS+set}" ] && [ -n "${PASS_SYSTEM_PROMPTS[$i]:-}" ]; then
        pass_extra+=(--system-prompt-text "${PASS_SYSTEM_PROMPTS[$i]}")
      fi
      echo ">>> [${tag}] GPU=${gpus[0]} model=${model} mode=${mode}"
      CUDA_VISIBLE_DEVICES="${gpus[0]}" python -m "${audit_module}" \
        --subset "${SUBSET}" \
        --model "${model}" \
        --mode "${mode}" \
        --out "${out}" \
        "${EXTRA_ARGS[@]}" "${pass_extra[@]}"
    done
    return 0
  fi

  echo ">>> parallel-per-gpu mode across ${n_gpu} GPU(s): ${gpus_str} (backend: ${audit_module})"
  echo "    ${n_pass} passes distributed round-robin; passes assigned to the same GPU run"
  echo "    sequentially in a per-GPU subshell to avoid co-located sglang OOM."

  # Group pass indices by their target GPU (round-robin: pass i -> gpus[i % n_gpu]).
  # Within each group we run sequentially so only one sglang engine occupies a
  # given GPU at a time. Without this, two engines each grabbing
  # mem_fraction_static=0.7 OOM the GPU on engine init.
  local -a per_gpu_passes
  local i
  for (( i = 0; i < n_gpu; i++ )); do per_gpu_passes[$i]=""; done
  for (( i = 0; i < n_pass; i++ )); do
    local g=$(( i % n_gpu ))
    per_gpu_passes[$g]+="${i} "
  done

  local -a pids gpu_labels
  local g
  for (( g = 0; g < n_gpu; g++ )); do
    local gpu="${gpus[$g]}"
    local idxs="${per_gpu_passes[$g]}"
    [ -z "${idxs// }" ] && continue
    (
      for idx in $idxs; do
        local tag="${PASS_TAGS[$idx]}" model="${PASS_MODELS[$idx]}" mode="${PASS_MODES[$idx]}"
        local out="${RUN_DIR}/${tag}.jsonl"
        if [ -f "${out}" ]; then
          echo ">>> [${tag}] already exists — skipping"
          continue
        fi
        local log="${RUN_DIR}/${tag}.log"
        # Per-pass overrides: when PASS_SYSTEM_PROMPTS is defined (parallel to
        # PASS_TAGS), pull this pass's system prompt and forward it to the
        # runner. Used to inject MMR1's training-time prompt for MMR1 models
        # only — Base passes get an empty string and the arg is omitted.
        local -a pass_extra=()
        if [ -n "${PASS_SYSTEM_PROMPTS+set}" ] && [ -n "${PASS_SYSTEM_PROMPTS[$idx]:-}" ]; then
          pass_extra+=(--system-prompt-text "${PASS_SYSTEM_PROMPTS[$idx]}")
        fi
        echo ">>> [${tag}] GPU=${gpu} model=${model} mode=${mode}  log=${log}"
        CUDA_VISIBLE_DEVICES="${gpu}" python -m "${audit_module}" \
          --subset "${SUBSET}" \
          --model "${model}" \
          --mode "${mode}" \
          --out "${out}" \
          "${EXTRA_ARGS[@]}" "${pass_extra[@]}" >"${log}" 2>&1
      done
    ) &
    pids+=($!)
    gpu_labels+=("${gpu}")
  done

  local fail=0
  for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
      echo ">>> GPU ${gpu_labels[$i]} pass group done"
    else
      echo "ERROR: GPU ${gpu_labels[$i]} pass group failed (pid ${pids[$i]})" >&2
      echo "       check the corresponding per-tag .log files under ${RUN_DIR}/" >&2
      fail=1
    fi
  done

  if [ "${fail}" -ne 0 ]; then
    return 1
  fi
}

_dispatch_passes
