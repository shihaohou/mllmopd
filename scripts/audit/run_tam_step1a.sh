#!/usr/bin/env bash
# Step 1a TAM-VD calibration launcher (teacher_greedy mode).
#
# Two passes inside one Python call:
#   Pass 1: teacher full forward (greedy gen + hidden + attn + scores) +
#           teacher blank-image scoring on the same response_ids.
#           Writes ${OUT_DIR}/teacher_cache.jsonl (checkpoint-invariant).
#   Pass 2: for each student ckpt, teacher-forced scoring of teacher's
#           response → student_lp / student_entropy / adv / quad.
#           Writes ${OUT_DIR}/tam_step1a.jsonl (one row per sample × ckpt).
#
# Wall-time (per v0.1.2 schema §forward-cost ledger):
#   teacher pass: ~17 min for 200 samples × 1 ckpt (eager attn ~5x FA2)
#   student pass: ~15 min × 4 ckpts
#   → total ~30-40 min on H800 if teacher cache is fresh; ~15 min if reused.
#
# Required env (sourced from .env or pre-exported):
#   MLLMOPD_RUNS         — output base
#   MMR1_7B_RL_CKPT      — teacher (default: MMR1/MMR1-7B-RL)
#   MMR1_3B_SFT_CKPT     — T1-0 student
#   CKPT_T1_2 / CKPT_T1_3 / CKPT_T2_1 — student ckpts (optional)
#
# Optional env (defaults shown):
#   RUN_ID               — output subdir (default: tam_step1a_$(date +%Y%m%d-%H%M%S))
#   SUBSET               — calibration sample JSONL
#                          (default: data/audit/tam_calibration_subset_v0.jsonl)
#   MAX_NEW_TOKENS       — gen cap (default: 512)
#   CUDA_VISIBLE_DEVICES — which GPU (default: 0)
#   LIMIT                — limit n samples for smoke (default: 0 = all)
#   MLLMOPD_ATTN_IMPL    — eager|sdpa|flash_attention_2 (default: eager —
#                          attention baseline requires it)
#   SKIP_STUDENT         — 1 = write teacher cache only, no student pass
#   TEACHER_CACHE        — reuse existing teacher cache JSONL
#   NUM_GPUS             — data-parallel shard count (default: 1; set 8 on
#                          H800 to fan out — each GPU runs its own
#                          subprocess on rows[shard_id::NUM_GPUS]). Speedup
#                          ~linear in NUM_GPUS; ~6-7 min wall for 200×4 on 8
#                          H800s vs ~40 min single-GPU
#
# Usage:
#   bash scripts/audit/run_tam_step1a.sh
#   SKIP_STUDENT=1 bash scripts/audit/run_tam_step1a.sh   # teacher pass only
#   LIMIT=10 bash scripts/audit/run_tam_step1a.sh         # 10-sample smoke

set -euo pipefail

if [ -f .env ]; then
  # shellcheck disable=SC1091
  source .env
fi
: "${MLLMOPD_RUNS:?MLLMOPD_RUNS must be set}"
: "${MMR1_3B_SFT_CKPT:?MMR1_3B_SFT_CKPT must be set (T1-0 baseline)}"

unset LD_LIBRARY_PATH || true
# shellcheck source=../env/_activate.sh disable=SC1091
source scripts/env/_activate.sh

# Proxy gotcha (per project_h800_proxy memory)
unset -v http_proxy https_proxy no_proxy || true

# v0.1.2: code commit + eager attention for attention baseline
export MLLMOPD_CODE_COMMIT="$(git rev-parse --short=10 HEAD 2>/dev/null || echo unknown)"
export MLLMOPD_ATTN_IMPL="${MLLMOPD_ATTN_IMPL:-eager}"

# Mitigate per-shard allocator fragmentation on long-CoT samples
# (ChartQA / HallusionBench produced 77/86 of the OOMs on `926e8f4`).
# expandable_segments:True lets PyTorch grow / shrink virtual segments
# instead of locking into fixed blocks that fragment with variable-size
# generations. PyTorch ≥ 2.1 required; the error message itself
# explicitly recommends this.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# spaCy preflight
python -c "import spacy; spacy.load('en_core_web_sm')" 2>/dev/null || {
  echo "!! spaCy en_core_web_sm not loadable. POS-based token_category will"
  echo "!! degrade to 'other' for content_noun / pronoun / visual_attribute."
  echo "!! Fix: pip install spacy && python -m spacy download en_core_web_sm"
}

TS="$(date +%Y%m%d-%H%M%S)"
RUN_ID="${RUN_ID:-tam_step1a_${TS}}"
RUN_DIR="${MLLMOPD_RUNS}/audit/${RUN_ID}"
SUBSET="${SUBSET:-data/audit/tam_calibration_subset_v0.jsonl}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
LIMIT="${LIMIT:-0}"
TEACHER_CKPT="${MMR1_7B_RL_CKPT:-MMR1/MMR1-7B-RL}"
NUM_GPUS="${NUM_GPUS:-1}"
# Cross-box sharding (2026-05-26): when running precompute on multiple
# boxes that share a ceph mount, set BOX_RANK / NUM_BOXES per box. Each
# box owns shards [BOX_RANK*NUM_GPUS .. (BOX_RANK+1)*NUM_GPUS-1] of the
# cluster-wide TOTAL_SHARDS = NUM_BOXES * NUM_GPUS. All boxes write to
# the SAME RUN_DIR; final merge is manual (see banner at end).
BOX_RANK="${BOX_RANK:-0}"
NUM_BOXES="${NUM_BOXES:-1}"
# SHARDS_PER_GPU > 1 (2026-05-26): run multiple processes per physical GPU
# when teacher_pass underutilizes compute. Useful for high-mem GPUs (H800
# 141GB) where one MMR1-7B-RL teacher only occupies ~20GB. Each extra
# process pays for its own CUDA context + model copy (~14GB weights), so
# the practical max is ~6 procs/gpu before OOM. Speedup is typically
# 1.3-1.5x at SHARDS_PER_GPU=2 (compute, not memory, is the bottleneck).
SHARDS_PER_GPU="${SHARDS_PER_GPU:-1}"
LOCAL_SHARDS=$(( NUM_GPUS * SHARDS_PER_GPU ))
TOTAL_SHARDS=$(( LOCAL_SHARDS * NUM_BOXES ))
if [ "${NUM_GPUS}" = "1" ]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
fi
if [ "${BOX_RANK}" -ge "${NUM_BOXES}" ]; then
  echo "ERROR: BOX_RANK=${BOX_RANK} must be < NUM_BOXES=${NUM_BOXES}" >&2
  exit 1
fi

# Build --students NAME=PATH args from env. Skip any that are empty.
STUDENT_ARGS=(--students "T1_0=${MMR1_3B_SFT_CKPT}")
if [ -n "${CKPT_T1_2:-}" ]; then STUDENT_ARGS+=(--students "T1_2=${CKPT_T1_2}"); fi
if [ -n "${CKPT_T1_3:-}" ]; then STUDENT_ARGS+=(--students "T1_3=${CKPT_T1_3}"); fi
if [ -n "${CKPT_T2_1:-}" ]; then STUDENT_ARGS+=(--students "T2_1=${CKPT_T2_1}"); fi

EXTRA=()
if [ "${LIMIT}" != "0" ]; then EXTRA+=(--limit "${LIMIT}"); fi
if [ "${SKIP_STUDENT:-0}" = "1" ]; then EXTRA+=(--skip-student); fi
if [ -n "${TEACHER_CACHE:-}" ]; then EXTRA+=(--teacher-cache "${TEACHER_CACHE}"); fi

mkdir -p "${RUN_DIR}"

cat <<EOF
========================================
TAM Step 1a launching
========================================
  RUN_ID       = ${RUN_ID}
  RUN_DIR      = ${RUN_DIR}
  TEACHER      = ${TEACHER_CKPT}
  STUDENTS     = ${STUDENT_ARGS[@]}
  SUBSET       = ${SUBSET}
  MAX_NEW_TOK  = ${MAX_NEW_TOKENS}
  LIMIT        = ${LIMIT}
  ATTN_IMPL    = ${MLLMOPD_ATTN_IMPL}
  NUM_GPUS        = ${NUM_GPUS}
  SHARDS_PER_GPU  = ${SHARDS_PER_GPU}
  LOCAL_SHARDS    = ${LOCAL_SHARDS}  (= NUM_GPUS × SHARDS_PER_GPU)
  NUM_BOXES       = ${NUM_BOXES}
  BOX_RANK        = ${BOX_RANK}
  TOTAL_SHARDS    = ${TOTAL_SHARDS}  (= LOCAL_SHARDS × NUM_BOXES)
  this box owns shard IDs [$(( BOX_RANK * LOCAL_SHARDS ))..$(( (BOX_RANK + 1) * LOCAL_SHARDS - 1 ))]
  CUDA            = ${CUDA_VISIBLE_DEVICES:-<unset>}
========================================
EOF

mkdir -p "${RUN_DIR}"

if [ "${NUM_GPUS}" = "1" ]; then
  PYTHONPATH=src python -m scripts.audit.tam_step1a \
    --subset "${SUBSET}" \
    --teacher "${TEACHER_CKPT}" \
    "${STUDENT_ARGS[@]}" \
    --out-dir "${RUN_DIR}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --image-root . \
    "${EXTRA[@]}"
else
  # ---- Multi-GPU data-parallel fan-out (optionally cross-box) ----
  # Each shard process is pinned to one GPU via CUDA_VISIBLE_DEVICES, gets
  # the stride-(global_id)'th rows of the subset (where global_id =
  # BOX_RANK*NUM_GPUS + i, and stride = TOTAL_SHARDS). All boxes write to
  # the same RUN_DIR/shard_<global_id> via shared ceph.
  if [ "${NUM_BOXES}" -gt 1 ]; then
    echo ">>> Cross-box mode: BOX_RANK=${BOX_RANK} of ${NUM_BOXES}; "\
         "this box runs ${LOCAL_SHARDS} shards out of cluster-wide ${TOTAL_SHARDS}"
    echo ">>> Auto-merge SKIPPED (other boxes may still be running)."
    echo "    After all boxes finish, run ON ONE BOX:"
    echo "      cat ${RUN_DIR}/shard_*/teacher_cache.jsonl > ${RUN_DIR}/teacher_cache.jsonl"
    echo "      cat ${RUN_DIR}/shard_*/tam_step1a.jsonl   > ${RUN_DIR}/tam_step1a.jsonl"
  fi
  if [ "${SHARDS_PER_GPU}" -gt 1 ]; then
    echo ">>> Multi-process per GPU: ${SHARDS_PER_GPU} shards share each of ${NUM_GPUS} GPUs"
    echo "    (each shard reloads MMR1-7B-RL ~14GB; expect 1.3-1.5x throughput, not Nx)"
  fi
  echo ">>> Fanning out to ${LOCAL_SHARDS} local shards on this box (cluster-wide stride sharding)"
  PIDS=()
  for local_i in $(seq 0 $((LOCAL_SHARDS - 1))); do
    GPU_IDX=$(( local_i % NUM_GPUS ))
    GLOBAL_ID=$(( BOX_RANK * LOCAL_SHARDS + local_i ))
    SHARD_DIR="${RUN_DIR}/shard_${GLOBAL_ID}"
    mkdir -p "${SHARD_DIR}"
    (
      export CUDA_VISIBLE_DEVICES="${GPU_IDX}"
      PYTHONPATH=src python -m scripts.audit.tam_step1a \
        --subset "${SUBSET}" \
        --teacher "${TEACHER_CKPT}" \
        "${STUDENT_ARGS[@]}" \
        --out-dir "${SHARD_DIR}" \
        --max-new-tokens "${MAX_NEW_TOKENS}" \
        --image-root . \
        --shard-id "${GLOBAL_ID}" \
        --num-shards "${TOTAL_SHARDS}" \
        "${EXTRA[@]}" \
        > "${SHARD_DIR}/stdout.log" 2> "${SHARD_DIR}/stderr.log"
      echo "  shard ${GLOBAL_ID} done (gpu ${GPU_IDX})"
    ) &
    PIDS+=($!)
  done

  # Wait for all shards; print partial status
  echo ">>> Waiting for ${#PIDS[@]} local shards on this box "\
       "(tail per-shard logs at ${RUN_DIR}/shard_<id>/std{out,err}.log)"
  for pid in "${PIDS[@]}"; do
    wait "${pid}" || echo "!! shard pid ${pid} exited non-zero"
  done

  # Merge only when single-box (cross-box: user merges manually after
  # confirming all boxes finished).
  if [ "${NUM_BOXES}" = "1" ]; then
    echo ">>> Merging shard outputs (single-box mode)"
    cat "${RUN_DIR}"/shard_*/teacher_cache.jsonl > "${RUN_DIR}/teacher_cache.jsonl" 2>/dev/null || true
    cat "${RUN_DIR}"/shard_*/tam_step1a.jsonl   > "${RUN_DIR}/tam_step1a.jsonl"   2>/dev/null || true
  else
    echo ">>> Multi-box mode: NOT merging on this box. After all boxes:"
    echo "    cat ${RUN_DIR}/shard_*/teacher_cache.jsonl > ${RUN_DIR}/teacher_cache.jsonl"
    echo "    cat ${RUN_DIR}/shard_*/tam_step1a.jsonl   > ${RUN_DIR}/tam_step1a.jsonl"
  fi

  # Aggregate summary
  {
    echo "# Step 1a teacher_greedy (multi-GPU)  (commit=${MLLMOPD_CODE_COMMIT})"
    echo "teacher = ${TEACHER_CKPT}"
    for s in "${STUDENT_ARGS[@]}"; do
      case "${s}" in --students) ;; *=*) echo "student ${s}" ;; esac
    done
    echo "subset    = ${SUBSET}"
    echo "num_shards (NUM_GPUS) = ${NUM_GPUS}"
    echo "n_samples_total = $(wc -l < "${SUBSET}")"
    echo "n_rows_total    = $(wc -l < "${RUN_DIR}/tam_step1a.jsonl" 2>/dev/null || echo 0)"
    echo "merged JSONL = ${RUN_DIR}/tam_step1a.jsonl"
    echo ""
    echo "per-shard:"
    for i in $(seq 0 $((NUM_GPUS - 1))); do
      sd="${RUN_DIR}/shard_${i}"
      n_t=$(wc -l < "${sd}/teacher_cache.jsonl" 2>/dev/null || echo 0)
      n_r=$(wc -l < "${sd}/tam_step1a.jsonl"   2>/dev/null || echo 0)
      echo "  shard ${i}: teacher_cache=${n_t}  rows=${n_r}"
    done
  } > "${RUN_DIR}/summary.txt"
fi

echo
echo "========================================"
echo "Step 1a DONE"
echo "========================================"
echo "  Teacher cache : ${RUN_DIR}/teacher_cache.jsonl"
echo "  Final JSONL   : ${RUN_DIR}/tam_step1a.jsonl"
echo "  Summary       : ${RUN_DIR}/summary.txt"
echo
if [ -f "${RUN_DIR}/summary.txt" ]; then
  cat "${RUN_DIR}/summary.txt"
fi
