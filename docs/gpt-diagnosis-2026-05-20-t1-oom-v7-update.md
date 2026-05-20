# GPT diagnosis brief — T1-2 OOM, v6 + v7 update (2026-05-20)

## Context

Follow-up to [docs/gpt-diagnosis-2026-05-20-t1-oom.md](https://github.com/shihaohou/mllmopd/blob/main/docs/gpt-diagnosis-2026-05-20-t1-oom.md).

GPT identified the root cause as sglang's static `mem_fraction_static`
block staying resident across `release_memory_occupation` calls.
Recommended fix stack (in order):

1. Rank 1: `SGLANG_MEM_FRACTION 0.55 → 0.25`
2. Rank 2: `--log-probs-chunk-size 256`
3. Rank 3: skip entropy compute when `entropy_coef==0`

I applied all three (commits `4b2358b`, `ba0abcc`). T1-2 then OOM'd
**twice more** with informative new patterns. I need a second-pass
diagnosis on the new data.

## Updated OOM table

| Attempt | Last good step | OOM site | Needed | PyTorch alloc | Free | Reserved-unalloc |
|---:|:---:|:---:|---:|---:|---:|---:|
| v1 | 6 | `logits.clone()` | 638 MiB | 131.36 GiB | 414 MiB | 798 MiB |
| v2 | 12 | `fused_vocab_parallel_cross_entropy` | 1.16 GiB | 130.14 GiB | 952 MiB | 756 MiB |
| v3 | ?? | same as v2 | 1.16 GiB | 129.46 GiB | 1.13 GiB | 1.01 GiB |
| v4 | init crash | TMS sanity_check raise on expandable_segments | — | — | — | — |
| v5 | 12 | `compute_entropy_from_logits` (`logits_max - X`) | 150 MiB | 130.35 GiB | 124 MiB | 1.17 GiB |
| **v6** | **13** | `backward_step` (`custom_backward`) | **8.48 GiB** | **115.22 GiB** | 7.54 GiB | **8.86 GiB** |
| **v7** | **0** | `fused_vocab_parallel_cross_entropy` (s10×151936 fp32) | 150 MiB | **95.64 GiB** | **2.38 MiB** | 93 MiB |

## What changed between v6 and v7

In commit `82606ed` I:

1. **Restored** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
   (originally added in `aa26bfd`, retracted in `c6a0bfd` because of
   the TMS sanity-check crash).
2. **Added** `--no-offload-rollout` to MISC_ARGS so sglang's
   `_TorchMemorySaverAdapter.create(enable=...)` picks the
   `_TorchMemorySaverAdapterNoop` variant (a literal `@contextmanager`
   yield, no `torch_memory_saver` import), eliminating the conflict
   with expandable_segments.

This worked at the layer it targeted (no more TMS crash), but the run
OOM'd at step 0 with a very different profile.

## The contradiction

```
v6:  PyTorch alloc = 115 GiB, free =  7.54 GiB → non-PyTorch ≈ 17 GiB
v7:  PyTorch alloc =  95 GiB, free =  2.38 MiB → non-PyTorch ≈ 44 GiB
```

PyTorch alloc dropped 20 GiB (expandable_segments is doing its job of
not over-reserving), but the **non-PyTorch chunk grew from 17 → 44
GiB**. Net: trainer has less usable memory than before.

The 44 GiB matches the budget for sglang at `mem_fraction=0.25`:

```
0.25 × 140 GiB KV cache pool        = 35 GiB
MMR1-3B student weights (bf16)      =  6 GiB
CUDA graph buffers / pinned mem     = ~3 GiB
                                     -------
total sglang static                 = 44 GiB
```

So with `--no-offload-rollout`, sglang now **really does** reserve the
full 0.25-fraction block and never yields it (the Noop adapter's
`pause/resume` are literal pass). expandable_segments lets PyTorch be
"polite" and not over-reserve, but sglang fills the void.

`max-tokens-per-gpu=16384` and the default `max_running_requests` keep
sglang on tight rollout batching, so we're paying the static cost
without using the throughput.

## Where I think this points

1. The `--no-offload-rollout` + expandable_segments combination is net
   worse for our workload than either alone, because sglang's static
   block is the binding constraint, not PyTorch fragmentation. By
   "freeing PyTorch from over-reserving" we just gave that headroom to
   sglang.

2. The sequence of OOMs shows PyTorch alloc itself is now well-under
   the budget (95 GiB at v7 vs 131 GiB at v1). The remaining gap is
   sglang static.

3. Hypotheses for the next fix (ranked by my own guess at safety):
   - **A.** Drop `SGLANG_MEM_FRACTION` further, `0.25 → 0.15` (frees
     ~14 GiB on the static block — sglang KV cache pool shrinks to
     ~21 GiB which is still > MMR1-3B's weight footprint).
   - **B.** Move `--sglang-disable-cuda-graph` out of DEBUG_MODE so
     it's on for full runs too. (CUDA graphs add cached batch-size
     captures inside sglang's mem block; disabling them frees some of
     the 44-GiB static chunk. Smoke uses this; production was missing
     it — GPT Q6 Rank 1 explicitly mentioned this.)
   - **C.** Revert `--no-offload-rollout` and run without
     expandable_segments. v6 PyTorch peak 115 + sglang ~17 = 132 GiB,
     leaves ~8 GiB headroom; the v6 OOM was 8.48 GiB which is
     borderline. Marginal but back to a known-safe-ish point.
   - **D.** Drop `--accumulate-allreduce-grads-in-fp32` (GPT Rank 4 —
     saves 12 GiB PyTorch alloc but is medium-risk on RL numerics).

4. I have a softer guess that the **real** answer is to disable cuda
   graphs in sglang and accept slower rollout. With cuda graphs off
   and a low mem_fraction, sglang's static footprint drops to mostly
   just `weights + small KV pool`, plausibly ~10-12 GiB. The earlier
   v6 numbers (when the offload was still on, so adapter Real,
   release worked imperfectly but still released *something*) suggest
   sglang's true minimum footprint after a successful release is ~17
   GiB. Matching that without `--offload-rollout` requires manually
   making sglang allocate less in the first place.

## Files for re-fetch

All on `main`:

- [scripts/train/opd_mmr1_3b_baseline.sh](https://github.com/shihaohou/mllmopd/blob/main/scripts/train/opd_mmr1_3b_baseline.sh)
  — current launcher (HEAD as of this brief)
- [third_party/sglang/python/sglang/srt/utils/torch_memory_saver_adapter.py](https://github.com/shihaohou/mllmopd/blob/main/third_party/sglang/python/sglang/srt/utils/torch_memory_saver_adapter.py)
  — confirms Noop adapter is a literal yield-only context manager
- [third_party/Uni-OPD/miles/miles/ray/actor_group.py](https://github.com/shihaohou/mllmopd/blob/main/third_party/Uni-OPD/miles/miles/ray/actor_group.py)
  lines 62-71 — `LD_PRELOAD` injection condition (gated only on
  `offload_train`, not `offload_rollout`)
- [scripts/setup/patch_uni_opd.sh](https://github.com/shihaohou/mllmopd/blob/main/scripts/setup/patch_uni_opd.sh)
  — current patch chain, includes the new P7 entropy-zero skip

## Questions for GPT

1. **Was applying `--no-offload-rollout` + expandable_segments a
   mistake?** It eliminated the v6 fragmentation OOM but now sglang
   eats the freed budget, giving us less effective headroom than
   before. Should we revert this and explore a different fix?
2. **Is dropping `SGLANG_MEM_FRACTION` to 0.15 safe** for our workload
   (MMR1-3B student, rollout_max_prompt=4096, rollout_max_response=2048,
   sample_n=8, rollout_batch_size=8 → 64 concurrent generations per
   rollout)?
3. **Should `--sglang-disable-cuda-graph` be on for production runs**
   too, not just smoke? Your Q6 Rank 1 mentioned it. What's the
   throughput cost roughly?
4. **Is there a way to drop sglang static below mem_fraction × GPU**
   — e.g. a `--kv-cache-tokens` or `--kv-cache-size` flag that
   bypasses the mem_fraction allocation?
5. **Given my data, which of A/B/C/D do you recommend?** If multiple,
   what order — apply one at a time and test, or stack two?
6. **Am I missing a non-sglang memory contributor?** The PyTorch alloc
   has dropped consistently across all our fixes (131 → 95) but the
   OOM keeps moving rather than disappearing. Is there a fundamental
   memory-budget calculation we're getting wrong?
