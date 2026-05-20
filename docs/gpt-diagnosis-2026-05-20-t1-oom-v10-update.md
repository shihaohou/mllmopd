# GPT diagnosis brief — T1-2 OOM, v9 + v10 (2026-05-20)

## Context

Third-pass diagnosis after I applied your Rank 4 (drop
`--accumulate-allreduce-grads-in-fp32`) plus chunk 256 → 128.

Both [docs/gpt-diagnosis-2026-05-20-t1-oom.md](https://github.com/shihaohou/mllmopd/blob/main/docs/gpt-diagnosis-2026-05-20-t1-oom.md)
and [docs/gpt-diagnosis-2026-05-20-t1-oom-v7-update.md](https://github.com/shihaohou/mllmopd/blob/main/docs/gpt-diagnosis-2026-05-20-t1-oom-v7-update.md)
remain context. v8 verified sglang shrink solved memory OOM but cost
throughput; v9 (MilesRouter + cuda_graph_max_bs=32 + max_running=32
+ continuous_decode=4 + the P8 FastAPI lifespan patch) recovered
throughput cleanly: 4 engines balanced at 65% util, ~6k tok/s
aggregate, 10 steps in 7 min. But it OOM'd at step 10 actor_train.

I then applied Rank 4 and chunk 128 (commit `fdc89d9`). **v10 OOM'd
at the same call site with effectively the same PyTorch alloc.**

## Updated OOM table

| Attempt | Step at OOM | OOM call site | PyTorch alloc | Free | Notes |
|---:|---:|---|---:|---:|---|
| v6 | 13 backward | `custom_backward` | 115.22 GiB | 7.54 GiB | dist-opt only |
| v7 | 0 | `fused_vocab_parallel_cross_entropy` | 95.64 GiB | 2.38 MiB | added expandable_segments + no-offload-rollout; sglang expanded 17 → 44 GiB |
| v8 | clean | (slow) | ~55 GiB | many GiB | mem_fraction 0.15, chunk 256, entropy skip; throughput cratered 10x |
| v9 | 10 actor_train | `fused_vocab_parallel_cross_entropy` | **118.33 GiB** | 8 MiB | throughput recovery stack, 4 engines balanced |
| **v10** | 12 actor_train | **same** | **118.98 GiB** | 6 MiB | dropped fp32 grad allreduce + chunk 256→128 |

Three observations:

1. **Dropping `--accumulate-allreduce-grads-in-fp32` saved
   essentially zero GiB** (118.33 → 118.98, even slightly *higher*).
   I believe Megatron's `--use-distributed-optimizer` already handles
   fp32 accumulation internally, making the flag redundant in our
   config. Or there's another 12-GiB allocator that absorbed the
   "savings".

2. **Halving log-probs-chunk-size (256 → 128) saved nothing
   measurable.** The 1.16 GiB CE buffer was not the binding peak.

3. **PyTorch trainer alloc has converged to ~118 GiB at actor_train
   peak across v6 / v9 / v10**, completely insensitive to the fixes
   we've applied to that side of the budget.

## The 83 GiB I can't account for

Per-rank budget my own arithmetic produces for actor_train:

```text
bf16 weights (3.09B × 2)               =  6.2 GiB
bf16 gradients (3.09B × 2)             =  6.2 GiB
fp32 master + m + v, sharded by DP=4   =  9.3 GiB
Activations, recompute-num-layers=1
  (per-micro-batch peak ~2.5 GiB, 1 mb at a time during fwd-bwd)
                                       = ~10 GiB
fp32 CE chunk @ 128 tokens × 151936
                                       =  0.3 GiB
small buffers + NCCL                   = ~3 GiB
                                         ------
total expected                         ≈ 35 GiB
total observed                         ≈ 118 GiB
unexplained                            ≈ 83 GiB
```

Either my activation accounting is wildly off, or there's something
structural eating 70-80 GiB that the fixes can't reach.

Hypotheses:

- The `--recompute-method uniform --recompute-num-layers 1` flags
  aren't actually doing what I think. Maybe Megatron's "uniform" with
  num_layers=1 is the trivial chunking (recompute group of size 1 =
  recompute every layer's input, store every layer's output). The
  v7-update brief had this conversation; you said yes, it does
  recompute every layer but still stores boundary inputs. But the
  boundary storage I estimated at 36 × 6144 × 2048 × 2 = 1.8 GiB
  per micro-batch — not 70 GiB.
- The fused_cross_entropy compiled by `torch._inductor` is keeping
  cached state. The OOM stacktrace runs through
  `torch._functorch._aot_autograd.runtime_wrappers` and
  `torch._inductor.output_code`. Could the AOT-Autograd
  graph capture be retaining 70+ GiB?
- The dual_teacher OPD reward path holds teacher_log_probs +
  diagnostic lp_full/lp_blank for the entire rollout (64 samples ×
  2048 response tokens × 2 bytes = 0.5 GiB; not enough).
- A Megatron buffer related to `--accumulate-allreduce-grads-in-fp32`
  that **didn't** go away when I dropped the flag (the launcher's
  `--use-distributed-optimizer` may keep its own fp32 grad scratch
  the same size).
- Sequence-parallel at TP=1 is technically a no-op, but maybe its
  buffer plumbing still allocates.

## Current launcher (HEAD `fdc89d9`)

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GLOBAL_BATCH_SIZE=64
MICRO_BATCH_SIZE=1
DP_SIZE=4
TP_SIZE=1
ROLLOUT_MAX_PROMPT_LEN=4096
ROLLOUT_MAX_RESPONSE_LEN=2048

Megatron PERF_ARGS:
  --tensor-model-parallel-size 1
  --sequence-parallel
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --use-distributed-optimizer
  --log-probs-chunk-size 128
  --use-dynamic-batch-size
  --max-tokens-per-gpu 16384

Megatron MISC_ARGS:
  --attention-softmax-in-fp32
  --attention-backend flash
  --colocate
  --megatron-to-hf-mode bridge
  --no-offload-train
  --no-offload-rollout
  --use-miles-router
  --no-gradient-accumulation-fusion
  # --accumulate-allreduce-grads-in-fp32 NO LONGER set (this is v10)

SGLang:
  --sglang-mem-fraction-static 0.15
  --sglang-max-total-tokens 200000
  --sglang-max-running-requests 32
  --sglang-cuda-graph-max-bs 32
  --sglang-num-continuous-decode-steps 4

Uni-OPD patches applied (patch_uni_opd.sh):
  P1-P6: existing
  P7: skip entropy compute when entropy_coef=0
  P8: MilesRouter FastAPI lifespan migration
```

## Cluster topology

8 × H800-140GiB on one box:

```text
GPU 0: standalone teacher sglang server (MMR1-7B-RL, mem_fraction 0.85)
GPU 1-4: colocated trainer (Megatron) + rollout sglang student engines
         (4 engines, 1 GPU each, mem_fraction_static 0.15 each)
GPU 5-7: idle
```

That's the colocate-on-shared-GPU pattern. v7→v10 sit at
"trainer 118 GiB + sglang 21 GiB = 139 / 140 GiB" on each of GPU 1-4.

## Two structural alternatives I'm considering

### Alt A: Dedicated rollout (no colocate)

Move sglang engines off the trainer GPUs:

```text
GPU 0: teacher sglang (unchanged)
GPU 1-3: dedicated rollout sglang (3 engines, 1 GPU each)
GPU 4-7: dedicated trainer (DP=4, no sglang on these GPUs)
```

Pros:
- Trainer GPUs no longer share with sglang static block (21 GiB freed)
- Drops `--colocate` and all the TMS/no-offload-rollout/expandable_segments
  carefulness that 5 previous fixes were spent on
- Each trainer GPU effectively has 140 GiB to itself

Cons:
- 3 rollout engines instead of 4 (so 64 generations / 3 ≈ 22 per engine)
- Requires launcher rewrite around `--colocate`
- Need to verify Uni-OPD's `--rollout-num-gpus` decoupled from trainer
  GPUs

### Alt B: 8-GPU ZeRO-3 (or TP-aware)

User suggests: use all 8 GPUs with ZeRO-3, sharding *parameters* too.

```text
GPU 0: still teacher
GPU 1-7: trainer (DP=7) with ZeRO-3
```

Or with TP=2:

```text
GPU 0-1: trainer TP shard 1 (or teacher on 0, trainer 1-7 with TP=2 DP=3)
```

Pros:
- ZeRO-3 shards weights too: per-rank weight + grad = (6+6)/7 ≈ 1.7 GiB
  vs current 12 GiB. Saves ~10 GiB per rank.
- TP=2 also halves activation memory per rank.

Cons:
- Big launcher / config change.
- GBS=64 requires DP × MBS | 64. DP=7 doesn't divide 64.
  Have to adjust GBS or batch.
- TP communication overhead.
- Have to re-do smoke + dist-opt validation.

## Questions for GPT

1. **What is most likely sitting in the 70-80 GiB I can't account for?**
   Specifically: (a) does Megatron with `--use-distributed-optimizer`
   still allocate a full-size fp32 grad buffer despite my dropping
   `--accumulate-allreduce-grads-in-fp32`? (b) can `torch._inductor`
   compiled AOT graph cache reach 70 GiB? (c) am I wrong about
   `--recompute-method uniform --recompute-num-layers 1` actually
   working?

2. **How do I instrument PyTorch to actually see the breakdown** —
   `torch.cuda.memory_snapshot()` + `_dump_snapshot()` + view in
   torch_view? Where would you put the snapshot call in Uni-OPD's
   actor train loop? Right before the OOM-ing CE call seems right
   (it's deterministic across runs — same file, same line).

3. **Is dedicated-rollout (Alt A) the right structural answer**, vs.
   trying more lever tuning on the colocated setup? The 21-GiB sglang
   static block on each trainer GPU is real and unavoidable in
   colocate; freeing it would directly buy us back the headroom v6
   used to have plus 13 GiB margin.

4. **If we go ZeRO-3 (Alt B)**: is the GBS-divisibility constraint a
   real blocker or can we drop to GBS=56 (7 × 8) and adjust the
   experiment? Or DP=4 + TP=2 to keep 8 GPUs and GBS=64?

5. **Or — has Uni-OPD actually been validated on multi-GPU 3B-class
   workloads** in colocate mode at the scale we're attempting (3B
   student × max-seq 6144 × GBS=64 × ZeRO-1)? If the framework
   defaults assume bigger GPUs (H100 80G) but a different total
   memory budget (no colocate, or ZeRO-2/3), we might be running it
   outside its tested regime.

6. **One more sanity check**: in the OOM stacktrace, the
   `empty_strided_cuda((s10, 1, 151936), torch.float32)` allocation
   is from `torch._inductor` compiled code at
   `/tmp/torchinductor_root/iv/civregd...py`. This is the **same
   cache file across all our runs** (filename is content-hashed).
   Could there be a torchinductor disk-cache state that's wrong, or
   a graph that compiled poorly on first run and is now stuck? Worth
   nuking `/tmp/torchinductor_root/` between attempts?

## Files for re-fetch

- [scripts/train/opd_mmr1_3b_baseline.sh](https://github.com/shihaohou/mllmopd/blob/main/scripts/train/opd_mmr1_3b_baseline.sh) (HEAD fdc89d9)
- [third_party/Uni-OPD/miles/miles/backends/megatron_utils/model.py](https://github.com/shihaohou/mllmopd/blob/main/third_party/Uni-OPD/miles/miles/backends/megatron_utils/model.py) — train_one_step orchestration
- [third_party/Megatron-LM/megatron/core/pipeline_parallel/schedules.py](https://github.com/shihaohou/mllmopd/blob/main/third_party/Megatron-LM/megatron/core/pipeline_parallel/schedules.py) — forward/backward where activations live
- [third_party/Uni-OPD/miles/miles/backends/training_utils/loss.py](https://github.com/shihaohou/mllmopd/blob/main/third_party/Uni-OPD/miles/miles/backends/training_utils/loss.py) — OPD loss path
- [third_party/Megatron-LM/megatron/core/distributed/distributed_data_parallel.py](https://github.com/shihaohou/mllmopd/blob/main/third_party/Megatron-LM/megatron/core/distributed/distributed_data_parallel.py) — where DDP grad buffer lives
- [scripts/setup/patch_uni_opd.sh](https://github.com/shihaohou/mllmopd/blob/main/scripts/setup/patch_uni_opd.sh) — current patch chain
