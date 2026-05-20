# GPT diagnosis brief — T1-2 persistent OOM (2026-05-20)

## TL;DR

T1-2 (FullTeacher OPD) has OOM'd **three times in a row** at the same
call site, with PyTorch allocated memory landing within a 1-GiB band
across all three attempts even after applying two memory-saving fixes
in between. We've explored:

1. `--use-distributed-optimizer` (verified active in Megatron banner)
2. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (env-var override;
   the third paste may have been mangled by a `\`-newline syntax error)
3. `SGLANG_MEM_FRACTION 0.55 → 0.45` (same caveat)

Yet peak PyTorch-allocated memory barely moved. **~70 GiB of allocation
is unaccounted for in our paper-budget; we need help identifying what's
consuming it.**

## Setup

- Repo: `https://github.com/shihaohou/mllmopd` HEAD `caba46b`
- Box: 8 × H800 140 GiB (cluster: `arc-wlf1-ge103-1`)
- GPU 0: teacher MMR1-7B-RL sglang server, mem_fraction 0.85 (~141 GiB occupied)
- GPUs 1-4: trainer + colocated student rollout sglang, DP=4 / TP=1
- GPUs 5-7: idle

### Model (student)
- Qwen2.5-VL-3B (MMR1-3B-SFT)
- hidden=2048, layers=36, FFN=11008
- num_attention_heads=16, num_query_groups=2 (GQA)
- vocab=151936
- ~3.09B params

### Launcher config
- `--micro-batch-size 1`, `--global-batch-size 64`, DP=4 → 16 micro-batches/opt-step
- `--rollout-max-prompt-len 4096`, `--rollout-max-response-len 2048` (max seq ≈ 6144)
- `--recompute-granularity full --recompute-method uniform --recompute-num-layers 1`
- `--use-distributed-optimizer` (verified)
- `--sequence-parallel` (no-op at TP=1)
- `--accumulate-allreduce-grads-in-fp32`
- `--attention-softmax-in-fp32`
- `--attention-backend flash`
- `--colocate --no-offload-train`
- `--no-gradient-accumulation-fusion` (apex unavailable)
- `--sglang-mem-fraction-static 0.55` (overridden to 0.45 in attempt 3)

### Data
- 5000-prompt train.jsonl, MMR1-RL filtered subset (`data/opd_train/v0_2k/train.jsonl`)
- truncated_ratio = **34%** of rollout samples hit MAX_RESPONSE_LEN=2048
- response_len: mean 1283, median 1419, p75/p90/p95/p99 all = 2048 (capped)
- Same launcher passed a 10-step smoke on 80-prompt subset
  (see `docs/handoff-2026-05-20.md` for smoke details).

## OOM attempts

| Attempt | Last good step | OOM at | Allocation needed | Allocated by PyTorch | Reserved-unallocated | Free |
|---:|:---:|:---:|:---:|:---:|:---:|:---:|
| v1 | step 6 (rollout 6 → actor_train) | `logits.clone()` | 638 MiB | 131.36 GiB | 797.90 MiB | 414.38 MiB |
| v2 | step 12 (after dist-opt) | `fused_vocab_parallel_cross_entropy` | 1.16 GiB | 130.14 GiB | 756.06 MiB | 952.38 MiB |
| v3 | step ?? (after fragmentation fix attempted) | `fused_vocab_parallel_cross_entropy` | 1.16 GiB | 129.46 GiB | 1.01 GiB | 1.13 GiB |

v2 and v3 call site is **identical**:

```
torch._inductor compiled call:
  buf9 = empty_strided_cuda((s10, 1, 151936), (151936, 151936*s10, 1), torch.float32)

→ third_party/Megatron-LM/megatron/core/fusions/fused_cross_entropy.py:104
  calculate_predicted_logits(...)
→ third_party/Megatron-LM/megatron/core/fusions/fused_cross_entropy.py:148
  _VocabParallelCrossEntropy.apply(vocab_parallel_logits, target, tp_group)
→ third_party/Uni-OPD/miles/miles/utils/ppo_utils.py:161
  return -fused_vocab_parallel_cross_entropy(logits, tokens, process_group)
→ third_party/Uni-OPD/miles/miles/utils/ppo_utils.py:682
  log_prob = compute_log_probs(logits.clone(), tokens, tp_group)
→ third_party/Uni-OPD/miles/miles/backends/training_utils/loss.py:589
  log_probs_and_entropy = get_log_probs_and_entropy(...)
→ third_party/Uni-OPD/miles/miles/backends/training_utils/loss.py:1006
  loss, log = func(args, parallel_state, batch, logits, sum_of_sample_mean)
```

The 1.16 GiB allocation is the **fp32 logits buffer** for the full vocab
slice during cross-entropy: `s10 × 151936 × 4 bytes` ≈ 1.16 GiB with s10 ≈ 1900.

Note that the OOM banner reports "Process X has 0 bytes memory in use,
including non-PyTorch memory" — this is the same trainer process; the
"X" pid varies per attempt. We interpret this as PyTorch's accounting,
not literal 0 usage by sglang.

## Memory budget — what we *think* is allocated (≈55 GiB)

Per trainer GPU (one of 1-4):

| Item | Size | Source |
|---|---:|---|
| bf16 weights | 6 GiB | 3.09B × 2 bytes |
| bf16 gradients | 6 GiB | 3.09B × 2 bytes |
| fp32 master copy (sharded by dist-opt) | 3 GiB | 12 GiB / DP=4 |
| Adam m + v (fp32, sharded by dist-opt) | 6 GiB | 24 GiB / DP=4 |
| fp32 grad allreduce buffer (`--accumulate-allreduce-grads-in-fp32`) | 12 GiB | 3.09B × 4 bytes |
| Activations w/ full recompute, num_layers=1 | ~10 GiB | rough estimate at max_seq 6144 |
| sglang student weights residual (post-release) | 6 GiB | MMR1-3B bf16, stays after `release_memory_occupation` |
| fp32 cross-entropy logits buffer | 1-4 GiB | 1900 × 151936 × 4 |
| CUDA context + misc buffers | ~5 GiB | guess |
| **Total** | **~55 GiB** | |

But PyTorch reports **130 GiB allocated** at OOM. **~75 GiB unaccounted.**

## What we don't understand

1. **The 75-GiB gap.** What is it? Plausible candidates:
   - Activations are *not* actually well-bounded by `--recompute-num-layers 1` —
     possibly the uniform recompute scheme is keeping all-layer activations across
     multiple micro-batches stitched into one optimizer step?
   - `torch._inductor` compiled cache grows over steps and isn't freed?
   - sglang's `release_memory_occupation` does *not* release the static
     `mem_fraction_static × 140 = 77 GiB` reservation, only KV cache?
     (If so, lowering mem_fraction would help — but smoke also used 0.55.)
   - Megatron creates extra fp32 working copies during the loss path?
   - The `--accumulate-allreduce-grads-in-fp32` buffer is allocated *per micro-batch*
     and not freed until the optimizer step?
2. **Why is the allocation ~constant across v1/v2/v3** (131 / 130 / 129 GiB)?
   Suggests we're hitting a structural ceiling that small env-var changes don't
   move — i.e., the 75-GiB gap is the same workload, not sensitive to allocator
   or sglang reservation.
3. **The fact that the same launcher passed a 10-step smoke (80 prompts, all
   short)** is consistent with our hypothesis that long-response samples blow
   the budget. With `truncated_ratio=34%`, T1's 5k pool guarantees a long-sample
   hit by step ~6.

## Things we have considered but not tried

| Option | Cost | Risk |
|---|---|---|
| `--fp16-lm-cross-entropy` | save ~1-2 GiB on logits buffer | small numeric impact on cross-entropy; may matter for RL log-prob math |
| Drop `--accumulate-allreduce-grads-in-fp32` | save 12 GiB on grad buffer | gradient numerics drift; may worsen RL training stability |
| `ROLLOUT_MAX_RESPONSE_LEN 2048 → 1024` | save ~half of activation memory peak | truncates ~34% of samples to half length; changes experiment |
| `GLOBAL_BATCH_SIZE 64 → 32` | save ~half of grad accum peak | half the throughput; statistical power changes |
| Increase `--recompute-num-layers` past 1 | unclear (uniform with 1 is already full recompute) | uniform with num_layers > 1 means *less* recompute (recompute fewer layers) — wrong direction |
| Disable inductor compile for cross-entropy | unknown | unknown — depends on Megatron's `@jit_fuser` decorator support for opt-out |
| Switch to non-fused cross-entropy | save the fp32 buffer entirely | might lose performance; codepath is in Megatron core |

## Questions for GPT

1. **What is most likely eating the 75-GiB gap?** Specifically: is there a way to
   inspect PyTorch's actual allocation profile so we can attribute the 130 GiB to
   tensor categories?
2. **Does `--recompute-method uniform --recompute-num-layers 1` truly recompute
   every layer in this Megatron version?** Or does it instead checkpoint every
   layer (keep full activations)? Our reading was "recompute every layer", but
   the memory numbers suggest the opposite.
3. **Does sglang's `release_memory_occupation` give back the static
   `mem_fraction_static` block?** Or only the KV cache pool? If the static block
   is permanent, the 77 GiB at 0.55 (or 63 GiB at 0.45) is *baseline*, and
   trainer has ~63-77 GiB to work with — which would partly explain the gap.
4. **Is dropping `--accumulate-allreduce-grads-in-fp32` safe** for RL training
   (PPO/OPD with low LR 1e-6, KL=0, entropy=0, clip 0.2/0.28) on a 3B model?
   The 12-GiB savings would push us decisively away from the ceiling.
5. **Is `--fp16-lm-cross-entropy` safe** for OPD training where the reward is
   `teacher_log_probs - student_log_probs`? The fp32 numerics matter for the
   cross-entropy reduction; we're worried about bias / instability.
6. **What's the minimum-disturbance fix you'd recommend?** Our preferences:
   keep MAX_RESPONSE_LEN=2048 and GBS=64 (these are experimental variables we
   don't want to perturb between T1-2 and prior baselines).

## Files to look at

All on `main` at HEAD `caba46b`:

- [scripts/train/opd_mmr1_3b_baseline.sh](https://github.com/shihaohou/mllmopd/blob/main/scripts/train/opd_mmr1_3b_baseline.sh) — full launcher with current memory args
- [third_party/Uni-OPD/miles/miles/utils/ppo_utils.py](https://github.com/shihaohou/mllmopd/blob/main/third_party/Uni-OPD/miles/miles/utils/ppo_utils.py) (line 682: `compute_log_probs`)
- [third_party/Uni-OPD/miles/miles/backends/training_utils/loss.py](https://github.com/shihaohou/mllmopd/blob/main/third_party/Uni-OPD/miles/miles/backends/training_utils/loss.py) (lines 589, 1006)
- [third_party/Megatron-LM/megatron/core/fusions/fused_cross_entropy.py](https://github.com/shihaohou/mllmopd/blob/main/third_party/Megatron-LM/megatron/core/fusions/fused_cross_entropy.py) — actual OOM call site
- [docs/handoff-2026-05-20.md](https://github.com/shihaohou/mllmopd/blob/main/docs/handoff-2026-05-20.md) — smoke pass narrative, why we believe pipeline is correct end-to-end
- [docs/gpt-review-2026-05-20-t1-full-implementation.md](https://github.com/shihaohou/mllmopd/blob/main/docs/gpt-review-2026-05-20-t1-full-implementation.md) — prior GPT review brief (method-layer was sound)

## Archive of failed runs (preserved on ceph)

- `runs/t1_v0_T1_2_full_oom_step6_20260520_1612/` (v1, 6 diagnostics steps)
- `runs/t1_v0_T1_2_full_oom_step12/` (v2, 12 diagnostics steps)
- v3's run dir would be `runs/t1_v0_T1_2_full/` (if not yet archived).
