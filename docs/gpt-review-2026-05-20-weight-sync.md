# GPT review brief — sglang colocate weight-sync invalid argument

**Date**: 2026-05-20 (T1 smoke debugging session, ~12h in)
**Repo**: https://github.com/shihaohou/mllmopd (HEAD ≈ `380786b`)
**Prior GPT reviews this session**: two — both correctly identified Ray
GPU-binding (NOSET_*_VISIBLE_DEVICES) and TMS LD_PRELOAD as P0 issues, and
later identified `pip` shim missing + torch fallback to system NCCL. All
those fixes landed and are NOT the issue anymore. Smoke now reaches
`actor_model.update_weights()` and is blocked there.

## Where we are on T1

- T1 implementation (code + dedup + analysis pipelines): **12/15 punch list items complete**, all committed and pushed.
- The remaining 3 items (#10 / #11 / #15) are GPU-time-bound: full training runs + writeup.
- Smoke (`scripts/train/smoke_t1.sh`) is the gating step before kicking off the 2 × 4-5h training runs. Smoke must pass before T1 can launch.

## What works now (confirmed via actor inspect logs)

```
✓ Driver 535 + CUDA cu128 forward-compat via /usr/local/cuda-12.9/compat/lib.real
✓ Venv NCCL 2.27.5 loaded (compile-time matched after fixing `pip` shim per common-pitfalls E5)
✓ Each Ray train actor has CUDA_VISIBLE_DEVICES=<single id>, device_count=1
✓ LD_PRELOAD does not contain torch_memory_saver
✓ 4-rank NCCL communicator init COMPLETE, AllGather opCount 0 [nranks=4] succeeds
✓ Megatron 3B model init: "number of parameters on (tensor, pipeline) model parallel rank (0, 0): 3085938688"
✓ HF checkpoint load via `--megatron-to-hf-mode bridge` succeeds
✓ Megatron → HF weight conversion via `Qwen25VLBridge`: 100% (644/644 weights converted)
✓ Proxy stripped: actor → sglang local IPC no longer routes through Tencent squid (was 502'ing on 4 of 4 calls)
```

## Where we're blocked: sglang weight-sync `cudaErrorInvalidValue`

Trace:

```
File "miles/backends/megatron_utils/update_weight/update_weight_from_tensor.py", line 141
    results = ray.get(refs)
File "miles/backends/sglang_utils/sglang_engine.py", line 271
    update_weights_from_tensor(...)
File "sglang/srt/managers/scheduler_update_weights_mixin.py", line 96
    worker.update_weights_from_tensor(recv_req)
File "sglang/srt/model_executor/model_runner.py", line 1399
    self._update_weights_from_flattened_bucket(reconstructed_tensors)
File "sglang/srt/models/qwen2_5_vl.py", line 851
    weight_loader(param, loaded_weight)
File "sglang/srt/model_loader/weight_utils.py", line 951
    param.data.copy_(loaded_weight)
→ torch.AcceleratorError: CUDA error: invalid argument
```

With `CUDA_LAUNCH_BLOCKING=1` set and a diag wrap inserted into sglang's
`default_weight_loader`, we caught the failing tensors. Diag output
(consistent across 4 sglang engines):

```
[mllmopd diag] default_weight_loader copy_ FAILED
  param  : shape=(2048,) dtype=torch.bfloat16 device=cuda:3 stride=(1,) contig=True
  loaded : shape=(2048,) dtype=torch.bfloat16 device=cuda:0 stride=(1,) contig=True
  err    : AcceleratorError: CUDA error: invalid argument
```

**shape / dtype / stride / contig all match. The only difference is `device`.**
`param` lives on the sglang engine's assigned GPU (`cuda:N` where N is the
engine's index inside its `CUDA_VISIBLE_DEVICES="1,2,3,4"` view); `loaded_weight`
arrives from Ray IPC marked as `cuda:0`.

The 2048-dim shape is consistent with Qwen2.5-VL bias / layer-norm. The first
hits are on small tensors but failures repeat on subsequent (larger) ones too.

### Architectural context

- Layout: 1 node, 8×A800-80GB, single-host Ray. Teacher sglang on GPU 0. Trainer + rollout colocate via Uni-OPD on GPUs 1-4 (we set `ACTOR_NUM_GPUS_PER_NODE=4`, `TP=1`, `--colocate`).
- Train actors: per actor `CUDA_VISIBLE_DEVICES=<single physical id>` (after our `--train-env-vars` clearing of `RAY_EXPERIMENTAL_NOSET_*_VISIBLE_DEVICES`). Each sees one logical `cuda:0`.
- SGLang engines: deliberately retain `CUDA_VISIBLE_DEVICES="1,2,3,4"` (NOSET kept for rollout per Uni-OPD's design — see `miles/ray/rollout.py`). Each engine sees 4 logical devices but only uses one based on its assigned `base_gpu_id`. Engine N's model params land on `cuda:N`.
- Weight sync path: Megatron actor converts via `Qwen25VLBridge` to HF format → packs into a flattened bucket → `ray.put` → sglang engine `ray.get` + `_update_weights_from_flattened_bucket` → per-tensor `default_weight_loader` → `param.data.copy_(loaded_weight)`.

### Why `param.data.copy_(loaded_weight)` fails here

We're chasing the hypothesis that the Ray IPC handoff drops `loaded_weight`
on the engine process's `cuda:0` (default current device), but the engine's
own model params for this bucket live on `cuda:N` (its assigned GPU). PyTorch's
`Tensor.copy_(src)` *should* handle cross-device implicitly, but in our
colocate setup it raises `cudaErrorInvalidValue` (12.8 runtime on driver 535
via compat layer). Best guess: the source pointer was allocated in the train
actor's CUDA context and isn't a valid alloc in the engine's process — i.e.,
the device tag `cuda:0` is preserved through pickle but the address space is
not portable across processes without explicit CUDA IPC handles.

### Our attempted fix (just pushed at `380786b`)

In `scripts/setup/patch_uni_opd.sh`, the sglang `default_weight_loader`
patch now coerces `loaded_weight` to `param.device` before `copy_`:

```python
if loaded_weight.device != param.device:
    loaded_weight = loaded_weight.to(param.device, non_blocking=False)
param.data.copy_(loaded_weight)
```

Rationale: `Tensor.to(device)` knows how to stage via host memory if direct
device-to-device copy isn't available. For a 4 KB bias the overhead is
negligible; for full-model sync it's a few-% wall-time hit. We haven't yet
verified if this fix works (haven't re-run smoke).

## Specific questions for review

1. **Is the device-coerce fix the right approach?** Or are we papering over a deeper bug in either Uni-OPD's bucket pack (`miles/backends/megatron_utils/update_weight/update_weight_from_tensor.py`) or sglang's bucket unpack (`sglang/srt/model_executor/model_runner.py::_update_weights_from_flattened_bucket`)?

2. **Should sglang engines actually have `CUDA_VISIBLE_DEVICES=<single id>` like train actors do, or is the `NOSET` for rollout intentional?** Uni-OPD's `miles/ray/rollout.py` deliberately sets NOSET for sglang engines per GPT's prior diagnosis (so sglang can manage its own multi-process layout via `base_gpu_id`). If we coerced sglang engines to single-GPU view too, the weight transfer might Just Work — but we'd also need to verify rollout still works.

3. **In Uni-OPD's `update_weight_from_tensor` flow, does the train actor explicitly migrate weights to the engine's GPU before serializing, or does it rely on the engine to handle device on the receive side?** If the former: the train-actor's `_pack_for_send` step is what should be device-aware, not the engine's loader. If the latter: our patch is the right layer.

4. **Anything else this sequence pattern looks like to you?** The "all OK except 1 weight fails" pattern with `cudaErrorInvalidValue` could be a known sglang colocate gotcha we don't know about. Particularly the fact that some engines log `POST /update_weights_from_tensor 200 OK` while others crash on the same call — suggests a race-like pattern, or a per-engine bug that only fires for specific engines.

## Key files for review

Production code (ours):
- `scripts/train/opd_mmr1_3b_baseline.sh` — main launcher, lots of LD/proxy/runtime_env fixes baked in
- `scripts/setup/patch_uni_opd.sh` — in-place patches to Uni-OPD + sglang submodules (margin_shift.py, ray_launcher.py, actor.py, weight_utils.py)
- `src/mllmopd/training/dual_teacher_get_reward.py` — our custom rm path (not in this critical path but for context)
- `docs/common-pitfalls.md` — especially E5 (NCCL ABI / pip shim story)

Upstream code to grok:
- `third_party/Uni-OPD/miles/miles/backends/megatron_utils/update_weight/update_weight_from_tensor.py` — the Megatron-side pack and dispatch
- `third_party/Uni-OPD/miles/miles/backends/sglang_utils/sglang_engine.py` — the SGLang HTTP client wrapper around `/update_weights_from_tensor`
- `third_party/sglang/python/sglang/srt/model_executor/model_runner.py` — `_update_weights_from_flattened_bucket` is the unpack logic
- `third_party/sglang/python/sglang/srt/model_loader/weight_utils.py:937` — `default_weight_loader`, where we patched

## What's at stake

- **All NCCL / placement / proxy / pip-shim issues are CLOSED.** This is the last (we think) blocker before smoke passes.
- T1 code is complete and validated at the unit / integration level. Only the smoke gate remains before we can launch 2 × 4-5h training runs (T1-2 FullTeacher + T1-3 BlankTeacher arms).
- If you can sanity-check the device-coerce fix OR propose a better layer to address it, we close the loop and start training.

Thanks.
