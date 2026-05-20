# GPT reply — T1-2 OOM v10-update (2026-05-20)

Source: pasted by user in chat after `docs/gpt-diagnosis-2026-05-20-t1-oom-v10-update.md`
brief was sent. Lightly reformatted (citations consolidated at bottom) but
content preserved verbatim.

---

## Bottom line

The v10 result means the remaining failure is **not** SGLang KV, CE chunking,
entropy, or fp32 allreduce. It is a **trainer-side peak around ~118–119 GiB**,
plus the colocated ~21 GiB SGLang block, leaving effectively no slack on
140 GiB H800s. v10 shows the same call site and nearly identical PyTorch
allocation after dropping fp32 allreduce and halving logprob chunk size, so
the next move should be **ground-truth PyTorch memory snapshots**, then
**dedicated rollout / no-colocate** for the production T1 run.

My strongest diagnosis: the "missing 83 GiB" is not one mysterious buffer.
It is mostly a combination of **dynamic packed microbatches up to 16,384
tokens**, **full fp32 `[1, T, vocab]` logits for the entire packed
microbatch**, **Megatron DDP/distributed-optimizer full param/grad
communication buffers**, and **runtime/backward temporaries**. Your budget
is still thinking "one 6,144-token sample per microbatch + only a 128-token
CE chunk," but the launcher is using `--use-dynamic-batch-size
--max-tokens-per-gpu 16384`, and the Uni-OPD loss receives float32 logits
shaped `[1, T, V]` before the chunked CE path even starts.

---

## Q1. What is most likely sitting in the 70–80 GiB?

### 1. The biggest budget miss is probably dynamic packed microbatching

Your current launcher has:

```bash
--micro-batch-size 1
--use-dynamic-batch-size
--max-tokens-per-gpu 16384
```

So "MBS=1" is not the real memory unit at the actor-train peak. The real
unit is closer to "pack samples until this rank sees up to ~16k tokens."
The model returns full-vocab float32 logits, and the loss code asserts
that the logits are float32 and shaped `[1, T, V]`.

For `T = 16384`, `V = 151936`:

```text
full output logits = 16384 × 151936 × 4 bytes
                   ≈ 9.28 GiB
```

That is **one** full-logits tensor. It is not the CE chunk. Chunking from
256 → 128 only reduces the temporary fused-CE/logprob chunk buffer; it
does **not** eliminate the full fp32 model-output logits tensor, nor the
backward graph and output-projection temporaries that depend on it. This
matches v10: chunk 256 → 128 saved nothing measurable.

A very quick falsification test:

```bash
# Keep GBS=64 and response len fixed.
# Only reduce per-rank actor-train packed-token peak.
--max-tokens-per-gpu 8192
```

If trainer peak drops materially, the missing memory is not a leak; it
is the dynamic packed training microbatch.

### 2. Megatron distributed optimizer still has full-size DDP param/grad buffers

Your suspicion about `--use-distributed-optimizer` subsuming the
fp32-allreduce flag is directionally right, but the important detail is
this: Megatron's distributed optimizer shards optimizer state, but the DDP
path still creates `_ParamAndGradBuffer` objects for params and grads, and
with distributed optimizer it uses reduce-scatter over the full grad
buffer. The code creates buffers by parameter group, and the bucket code
explicitly operates over `bucket.param_data`; when distributed optimizer
is enabled, gradient reduction is reduce-scatter, but the local full
communication buffer still exists.

So dropping:

```bash
--accumulate-allreduce-grads-in-fp32
```

does not mean "remove the Megatron grad buffer." At best it changes its
dtype if `grad_reduce_in_fp32` actually flips. Your measured 118.33 →
118.98 GiB says either the dtype did not change, another buffer replaced
the savings, or the peak is dominated elsewhere.

I would log this at init:

```python
for i, buf in enumerate(model[0].buffers):
    print(
        f"[DDP buffer {i}] "
        f"param_data={None if buf.param_data is None else (buf.param_data.numel(), buf.param_data.dtype)} "
        f"grad_data={(buf.grad_data.numel(), buf.grad_data.dtype)}"
    )
```

That will immediately answer whether `grad_data` is bf16 or fp32.

### 3. Recompute is probably working; it just cannot touch the real peak

`--recompute-method uniform --recompute-num-layers 1` is not the trivial
"keep every layer's activations" mode. Megatron's transformer block enters
checkpointed forward when `recompute_granularity == 'full'` during
training.

But full recompute only reduces stored transformer-layer internals. It
does not remove:

```text
full fp32 output logits
output projection backward temporaries
DDP param/grad buffers
distributed optimizer state
vision / multimodal tensors
NCCL / CUDA non-PyTorch allocations
SGLang static block
```

So your recompute estimate can be correct and still miss the actual peak.

### 4. Inductor AOT graph cache is unlikely to be 70–80 GiB

The stack going through `torch._inductor.output_code` only means the
failing allocation is happening inside compiled code. It does **not**
imply that the disk cache is holding 70 GiB of GPU memory. PyTorch's
compile cache stores compiled artifacts such as FX graph, AOTAutograd,
Inductor, Triton, PGO, and autotuning cache entries on disk, under
`TORCHINDUCTOR_CACHE_DIR`; it is a compile-time cache, not a persistent
tensor store.

A compiled graph can certainly allocate large runtime tensors. But
`/tmp/torchinductor_root/...py` being the same file across runs is more
consistent with "same graph shape, same generated code, same allocation"
than with "bad stale disk state eating 80 GiB."

---

## Q2. How to instrument PyTorch memory snapshots at the actor-train call site

Use two layers of instrumentation:

1. **lifetime snapshots** around model/optimizer setup and train step
   boundaries;
2. **surgical snapshots** inside the OPD loss immediately before the
   OOMing CE/logprob call.

PyTorch's memory snapshot tool records allocation/free/OOM events with
stack traces, and `_record_memory_history(...)` plus `_dump_snapshot(...)`
is the intended workflow. The official docs also warn that snapshots only
see memory managed by PyTorch's CUDA allocator; direct CUDA/NCCL
allocations may not appear, so compare against `nvidia-smi` /
`torch.cuda.mem_get_info()` too.

### Add this helper

Something like:

```python
# src/mllmopd/training/memsnap.py
import json
import os
import socket
import time
from pathlib import Path

import torch


_STARTED = False


def _enabled() -> bool:
    return os.environ.get("MLLMOPD_MEMSNAP", "0") == "1"


def _rank_tag() -> str:
    rank = os.environ.get("RANK") or os.environ.get("LOCAL_RANK") or "unknown"
    pid = os.getpid()
    host = socket.gethostname()
    dev = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    return f"host={host}_rank={rank}_pid={pid}_cuda={dev}"


def start_memory_history_once() -> None:
    global _STARTED

    if not _enabled() or _STARTED or not torch.cuda.is_available():
        return

    max_entries = int(os.environ.get("MLLMOPD_MEMSNAP_MAX_ENTRIES", "1000000"))

    # PyTorch private API signatures vary a bit by version.
    try:
        torch.cuda.memory._record_memory_history(
            enabled="all",
            stacks="all",
            max_entries=max_entries,
        )
    except TypeError:
        try:
            torch.cuda.memory._record_memory_history(
                enabled=True,
                max_entries=max_entries,
            )
        except TypeError:
            torch.cuda.memory._record_memory_history(max_entries=max_entries)

    _STARTED = True
    print(f"[memsnap] recording enabled max_entries={max_entries}", flush=True)


def dump_memory_snapshot(tag: str, **meta) -> None:
    if not _enabled() or not torch.cuda.is_available():
        return

    start_memory_history_once()

    torch.cuda.synchronize()

    out_dir = Path(os.environ.get("MLLMOPD_MEMSNAP_DIR", "/tmp/mllmopd_memsnap"))
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = int(time.time())
    safe_tag = tag.replace("/", "_").replace(" ", "_")
    prefix = out_dir / f"{ts}_{safe_tag}_{_rank_tag()}"

    free, total = torch.cuda.mem_get_info()
    summary = {
        "tag": tag,
        "time": ts,
        "rank_tag": _rank_tag(),
        "allocated_gib": torch.cuda.memory_allocated() / 2**30,
        "reserved_gib": torch.cuda.memory_reserved() / 2**30,
        "max_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "max_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "free_gib": free / 2**30,
        "total_gib": total / 2**30,
        **meta,
    }

    with open(f"{prefix}.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    with open(f"{prefix}.summary.txt", "w") as f:
        f.write(torch.cuda.memory_summary(abbreviated=False))

    try:
        torch.cuda.memory._dump_snapshot(f"{prefix}.pickle")
        print(f"[memsnap] dumped {prefix}.pickle {summary}", flush=True)
    except Exception as e:
        print(f"[memsnap] dump failed at {tag}: {type(e).__name__}: {e}", flush=True)
```

Add these env vars through the launcher's `TRAIN_ENV_VARS_JSON` so Ray
actors inherit them:

```bash
export MLLMOPD_MEMSNAP=1
export MLLMOPD_MEMSNAP_DIR="${EXPERIMENT_DIR}/diagnostics/memsnap"
export MLLMOPD_MEMSNAP_MAX_ENTRIES=2000000
```

### Where to put it

The existing Uni-OPD train step has a custom hook before train step, but
that is **too early** to see the OOMing tensor because logits have not
been created yet. In `train_one_step`, the hook fires before
`forward_backward_func`; the logits and loss happen inside the Megatron
forward/backward schedule.

I would still use the hook for baseline snapshots:

```python
# custom_megatron_before_train_step_hook
from mllmopd.training.memsnap import start_memory_history_once, dump_memory_snapshot

def before_train_step_hook(args, rollout_id, step_id, model, optimizer, opt_param_scheduler):
    start_memory_history_once()
    dump_memory_snapshot(
        "before_train_step",
        rollout_id=rollout_id,
        step_id=step_id,
    )
```

But the decisive patch belongs in:

```text
third_party/Uni-OPD/miles/miles/backends/training_utils/loss.py
policy_loss_function(...)
```

right before:

```python
log_probs_and_entropy = get_log_probs_and_entropy(...)
```

Add:

```python
from mllmopd.training.memsnap import dump_memory_snapshot

dump_memory_snapshot(
    "pre_get_log_probs_and_entropy",
    logits_shape=tuple(logits.shape),
    logits_dtype=str(logits.dtype),
    allocated_gib=torch.cuda.memory_allocated() / 2**30,
)

try:
    log_probs_and_entropy = get_log_probs_and_entropy(
        logits,
        args=args,
        parallel_state=parallel_state,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=args.entropy_coef != 0.0,
        max_seq_lens=max_seq_lens,
    )
except torch.cuda.OutOfMemoryError:
    dump_memory_snapshot(
        "oom_get_log_probs_and_entropy",
        logits_shape=tuple(logits.shape),
        logits_dtype=str(logits.dtype),
        total_lengths=[int(x) for x in total_lengths],
        response_lengths=[int(x) for x in response_lengths],
    )
    raise
```

Also log these scalar facts at that site:

```python
print(
    "[pre_ce]",
    "logits", tuple(logits.shape), logits.dtype,
    "total_lengths", total_lengths,
    "response_lengths", response_lengths,
    "max_seq_lens", max_seq_lens,
    "alloc_gib", torch.cuda.memory_allocated() / 2**30,
    "reserved_gib", torch.cuda.memory_reserved() / 2**30,
    flush=True,
)
```

The first thing I would inspect in the `.pickle` is the active allocation
list at `pre_get_log_probs_and_entropy`. If the largest blocks include
`[1, 16384, 151936]` or nearby fp32 vocab tensors, dynamic packed logits
are confirmed. If the largest blocks are `_ParamAndGradBuffer` / DDP
buffers, the Megatron buffer hypothesis is confirmed. If the biggest
stacks point into `torch/_inductor/output_code.py`, then it is runtime
compiled-operator allocation, not stale disk cache.

---

## Q3. Is dedicated rollout, Alt A, the right structural answer?

**Yes. For T1 production, Alt A is the right structural answer.**

v10 says the system is sitting at:

```text
trainer ~118 GiB + SGLang ~21 GiB = ~139 / 140 GiB
```

on GPUs 1–4.

That means any small runtime fluctuation can kill the run. Dedicated
rollout directly removes the colocated SGLang static block from trainer
GPUs:

```text
GPU 0: teacher SGLang
GPU 1–3: rollout SGLang
GPU 4–7: trainer DP=4, no colocated SGLang
```

This buys back about **21 GiB per trainer GPU** without changing:

```text
GBS=64
MAX_RESPONSE_LEN=2048
student model
teacher
OPD objective
trainer DP=4
TP=1
```

It also removes the entire fragile colocate/TMS/no-offload/expandable-segments
interaction. Your launcher comments already explain that
`--no-offload-rollout` makes SGLang memory release effectively a no-op,
and the v7/v10 briefs show why the colocated SGLang block is real.

The throughput cost should be manageable: v9 had four balanced rollout
engines at ~6k tok/s aggregate; three dedicated engines should probably
be slower, but not catastrophically slower. More importantly, the current
colocated setup is not merely slower; it OOMs.

I would run Alt A with snapshots still enabled. Alt A answers the
engineering problem; snapshots answer the science/debugging problem.

---

## Q4. If we go ZeRO-3 / TP-aware Alt B, is GBS divisibility real?

**Yes, the GBS divisibility constraint is real in the current launcher
and Megatron setup.**

The launcher explicitly checks:

```bash
GLOBAL_BATCH_SIZE % (MICRO_BATCH_SIZE * DP_SIZE) == 0
```

and it already comments that with `GBS=64`, `TP=1`, `MBS=1`, valid
trainer GPU counts are factors of 64, not 7.

So:

```text
GPU 1–7 trainer, DP=7, MBS=1:
64 % 7 != 0
```

That fails the invariant unless you change GBS to 56 or 63/70/etc.
Dropping to `GBS=56` is an experiment change, not just a memory change.
Given you explicitly want to keep `GBS=64`, I would not use DP=7 for T1.

`DP=4 + TP=2` keeps GBS divisibility, but it needs **8 trainer GPUs**.
On a single 8-GPU box with GPU0 reserved for the teacher, you do not have
8 trainer GPUs unless you move the teacher elsewhere. `TP=2 + DP=3` on
six trainer GPUs gives `64 % 3 != 0`; the launcher itself notes that a
TP=2/DP=3 full-paper run requires adjusted GBS.

My ranking:

```text
Immediate T1:
  Alt A dedicated rollout, trainer DP=4 TP=1 on GPUs 4–7.

If trainer still OOMs after Alt A:
  lower --max-tokens-per-gpu 16384 → 8192 first.
  This preserves GBS=64 and max response len; it only increases microbatches/step.

Longer-term:
  TP=2 only if you can allocate 8 trainer GPUs or move teacher off-node.
  True ZeRO-3/FSDP is a larger framework change, not the next debugging step.
```

Alt B is a reasonable full-infra path, but not the right immediate answer
to this T1 blocker.

---

## Q5. Is Uni-OPD validated for this exact colocated 3B × 6144 × GBS64 × ZeRO-1 regime?

I would treat the current setup as **outside the safe/default regime**,
not because Uni-OPD is wrong, but because your run has accumulated many
local patches and topology-specific workarounds.

Evidence from your own current launcher and patch chain:

```text
- Ray actor CUDA-visible-device override
- NCCL / LD_LIBRARY_PATH fixes
- stripping torch_memor from LD_PRELOAD
- --no-offload-train
- --no-offload-rollout
- SGLang memory release becoming no-op
- custom SGLang weight-sync serialization/device patches
- entropy skip patch
- MilesRouter FastAPI patch
- dynamic batch size with 16k tokens/GPU
- colocated rollout/trainer on the same GPUs
```

The patch script explicitly describes local-only patches to Uni-OPD,
SGLang weight sync, Ray environment propagation, entropy skipping, and
MilesRouter migration.

So I would not assume the upstream framework has been validated for:

```text
3B VLM student
max seq ~6144
dynamic packed actor-train up to 16k tokens/GPU
GBS=64
DP=4 TP=1
Megatron distributed optimizer
colocated trainer + rollout SGLang
no real SGLang memory offload/release
```

This is another reason Alt A is attractive: it removes one entire
interaction surface without changing the learning experiment.

---

## Q6. Could `/tmp/torchinductor_root/` contain a bad cached graph?

**Possible as a one-run sanity issue, but very unlikely to explain the
70–80 GiB.**

What the same cache filename tells you:

```text
same graph + same shape/signature + same compiler config
→ same generated file
→ same runtime allocation pattern
```

It does not imply that the disk cache is storing or leaking CUDA tensors.
PyTorch's compile cache stores artifacts such as FXGraphCache,
AOTAutogradCache, Inductor/Triton artifacts, PGO, and autotuning results;
the cache directory defaults to `/tmp/torchinductor_<username>`, and
`TORCHINDUCTOR_CACHE_DIR` controls it. PyTorch also provides
`TORCHINDUCTOR_FORCE_DISABLE_CACHES=1` to force recompilation for
debugging.

I would do exactly one sanity run:

```bash
rm -rf /tmp/torchinductor_root

export TORCHINDUCTOR_CACHE_DIR="${EXPERIMENT_DIR}/torchinductor_cache"
export TRITON_CACHE_DIR="${EXPERIMENT_DIR}/triton_cache"

# Optional cold-cache diagnostic:
# export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1
```

But I would not expect memory to change. If the memory is still ~118 GiB
after clearing the cache, stop spending time on this hypothesis.

The stronger diagnostic is the memory snapshot. If the active allocation
stack says `torch/_inductor/output_code.py` for a huge tensor, that is a
**runtime allocation by compiled code**. It is not evidence that the disk
cache is stale.

---

## Concrete next steps

### Step 1 — Add the memory snapshot patch

Dump all DP ranks at:

```text
after model/optimizer setup
before actor_train step
inside loss.py before get_log_probs_and_entropy
inside loss.py on OOM
after optimizer step, if reached
```

Use the snapshot pickle plus `torch.cuda.memory_summary()` and
`torch.cuda.mem_get_info()`.

### Step 2 — Run a 20-step diagnostic with lower token packing

Before a full architecture rewrite, test:

```bash
--max-tokens-per-gpu 8192
```

Keep:

```text
GBS=64
MAX_RESPONSE_LEN=2048
DP=4
TP=1
colocate unchanged
```

If peak drops from ~118 GiB to something materially lower, your "83 GiB"
is largely the dynamic packed actor-train peak. This preserves the
experimental invariants without changing throughput.

### Step 3 — Move to Alt A for the real T1 run

Use:

```text
GPU0: teacher
GPU1-3: rollout SGLang
GPU4-7: trainer DP=4 TP=1
drop --colocate
drop no-op SGLang offload complexity
```

This is the cleanest structural fix. It directly removes the 21 GiB
colocated SGLang block that is currently consuming all safety margin.

### Step 4 — Do not take Alt B yet

Do not move to DP=7 with `GBS=64`. The divisibility issue is real. Do
not do full ZeRO-3/FSDP unless Alt A plus a sane `max-tokens-per-gpu`
still fails.

The most likely successful path is:

```text
snapshot → confirm dynamic logits/DDP buffers → Alt A dedicated rollout
→ optionally reduce max_tokens_per_gpu if trainer peak still too close
```

---

## Sources GPT cited

- mllmopd@be0035c `docs/gpt-diagnosis-2026-05-20-t1-oom-v10-update.md`
- mllmopd@be0035c `scripts/train/opd_mmr1_3b_baseline.sh`
- NVIDIA/Megatron-LM `megatron/core/distributed/distributed_data_parallel.py`
- NVIDIA/Megatron-LM `megatron/core/transformer/transformer_block.py`
- PyTorch tutorials: Compile Time Caching Configuration
- PyTorch blog: Understanding GPU Memory 1 (visualizing allocations)
- WenjinHou/Uni-OPD@b349383 `miles/miles/backends/megatron_utils/model.py`
- mllmopd@be0035c `scripts/setup/patch_uni_opd.sh`
