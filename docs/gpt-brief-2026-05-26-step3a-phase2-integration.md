# Step 3a Phase 2 — Uni-OPD Integration Architecture (GPT Verdict Request)

Repo: https://github.com/shihaohou/mllmopd @ `c2ee652`

Relevant files:
- `docs/step3a-design-2026-05-25.md` — Phase 1 design (you greenlit earlier)
- `src/mllmopd/training/tam_gate.py` — Phase 1 gate module (locked at `eff2ffd`)
- `scripts/audit/tam_step3_preflight.py` — preflight + dump + cell audit infrastructure
- `runs/analysis/tam_step3_preflight_v2_full.txt` — preflight v2 results (this brief's data source)
- `third_party/Uni-OPD/miles/miles/backends/training_utils/loss.py:437-458,840` — per-token loss reduction site
- `src/mllmopd/training/dual_teacher_get_reward.py` — current `--custom-rm-path` wrapper (sglang teacher)
- `src/mllmopd/training/opd_diagnostics_hook.py` — current `--custom-reward-post-process-path` wrapper

**请用中文回复**。

---

## TL;DR (Phase 1 收官 + Phase 2 三条路二选一)

1. Phase 1 干净收官：classifier 修了 spaCy 把 MMR1 `\boxed{answer}` BPE-split 件误标成 PROPN 的问题；preflight v2 上 q3-over-fire 从 1.57× → 0.41×，§Method narrative 完全立住，τ=0.70 / α=0.50 LOCKED。
2. Phase 2 (Uni-OPD 接入) 遇到一个**架构 blocker**：teacher 在 Uni-OPD 训练里走 sglang，只 return per-token logp，不暴露 hidden states / 完整 logits → TAM 抽取（需要 full vocab logits）在训练 hot-path 里直接做不了。
3. 三条出路 A/B/C，我倾向 C（off-policy KD + 离线 TAM 预算），想要你拍板。

---

## Part 1: Phase 1 Final Results

### Classifier fix（commit `9c3fb36`）

Audit on `tam_step1a_20260525-190333` (anchor v0.1.2) showed `proper_noun|q3` 内部 63 个 token 全是 `vd≈adv≈0` 的 template (`"answer"` / `"boxed"`)，spaCy 把 `\boxed{answer}` 的 BPE pieces 误标成 PROPN。新增 `MMR1_BOXED_BARE_RE`：

```python
MMR1_BOXED_BARE_RE = re.compile(
    r"^\s*(?:boxed|Boxed|BOXED|answer|Answer|ANSWER)\s*$"
)
# 插在 TEMPLATE_TOKEN_RE 检查之后，spaCy 之前
```

### Preflight v2（4 ckpts × 205 samples = 820 rows, `tam_step1a_classifier_v013_full`）

**per-quad fire rate at τ=0.70**:

| | q0 (vis-support) | q1 | q2 (vis-reject teacher_toward) | q3 (T2-1 bucket) |
|---|---|---|---|---|
| v0.1.2 (anchor) | 0.112 | 0.135 | 0.124 | **0.157** ← 反常 |
| v0.1.3 (clean) | 0.134 | 0.158 | 0.061 | **0.055** ← q3 反而最低 |
| q3/q0 ratio | 1.40× | — | — | **0.41×** (反转) |

**within-C_local fire rate at τ=0.70**：q0=0.419 / q1=0.445 / q2=0.341 / q3=0.391。q3 跟 q0 在 C_local 内部仅差 3pp。

**per-category fire rates at τ=0.70**：content_noun=0.407, proper_noun=0.438, visual_attribute=0.420。三类拉齐。

**proper_noun|q3 cell**：63 → **4**（一个 ChartQA/791 token 12 = 数字 "0"，×4 ckpts；单一 spaCy 边角，可接受）

**cross-ckpt 一致性 at τ=0.70**：T1_0=0.1226 / T1_2=0.1230 / T1_3=0.1274 / T2_1=0.1241（极差 < 0.5pp）

**Lock**：τ=0.70 / α=0.50 / K=0.20 / ρ=0.30 / C_local = {content_noun, visual_attribute, proper_noun}，anchor JSONL = `tam_step1a_classifier_v013_full`

**§Method narrative 数据印证**：

```
within-C_local fire rate at τ=0.70:
  vision-support (q0+q1): ~42-44%     ← TAM-Boost gate 主火力
  vision-reject  (q2+q3): ~34-39%     ← 显著低，但不为零
```

q3 内部 ~39% 仍 fire 这件事跟 Step 2b "q3 不响应局部 mask" 的结论存在 tension —— gate 是 spatial co-occurrence 测度，不是 causal。但 no-suppress 下浪费的 boost budget 占比 ~0.2%（q3 C_local n=46 中 ~18 fire，占总 boost 的极小份额）。统计上无害，A5 oracle arm 会量化。

---

## Part 2: Phase 2 Integration 调研

### 接入点：loss.py:437-458 已有 per-token weight pattern

Uni-OPD 训练循环里 `teacher_vd_weights_list[i]` 已经在做 per-token 权重乘法（T2-1 用过）。TAM-Boost 镜像这条路径：

```python
# loss.py:448-457 现有
adv = (t_logps - s_logps) * teacher_valid_mask
vd_w = teacher_vd_weights_list[i]
adv = adv * vd_w

# 我们要加：
tam_w = teacher_tam_weights_list[i]    # shape (response_len,), 值 ≥ 1
adv = adv * tam_w
```

`post_process_rewards` hook (现有 `--custom-reward-post-process-path`) 在 `opd_diagnostics_hook.py:229` 可以 iterate samples + 挂 `sample.teacher_tam_weights` 进 sample。Plumbing 完全 ready。

### Blocker: sglang teacher 不返回 logits

`dual_teacher_get_reward.py:138`:
```python
teacher = sglang_endpoint(max_new_tokens=0, return_logprob=True)
# meta_info.input_token_logprobs ← 只有 (token, logp) pairs，没有 logits、没有 hidden_states
```

TAM 需要的是 teacher 在 student rollout 每个 position 上的完整 logit 分布 (`_tam_core.TAM` line 309: `logit_list[r]` shape `(1, 1, V)`)。sglang 标准接口不暴露这个。

### 三条出路

| | A. Patch sglang | B. 双 teacher | C. 离线 TAM + off-policy KD |
|---|---|---|---|
| **核心改动** | 改 `third_party/sglang` 让 teacher 同时 return hidden_states | HF teacher 跟 sglang teacher 并存：sglang 算 logp，HF 算 TAM | 用 Step 1a 基础设施 offline 算 TAM；训练时查表 |
| **训练时 forward 数** | 1（sglang）— 严格 one-forward | 2（sglang + HF） | **0** teacher forward at train time（TAM 完全 pre-cached） |
| **GPU mem 增量** | ~0 | +~20GB (7B HF teacher in bf16) | ~0 |
| **训练 step 时延增量** | 微（hidden states copy out） | +50-100%（HF forward in eager attn 比 sglang 慢 3-5×） | ~0 |
| **工作量** | 大（改 sglang，rebase upstream 痛苦；mllmopd 已经有 P14-P18 patch list） | 中（HF teacher 加载、内存管理） | 小（扩 `tam_step1a.py` 跑训练池） |
| **§Method narrative** | "one-forward gate via teacher's hidden states" | 严格意义 two-forward；narrative 退化 | **"zero-train-forward gate, TAM precomputed via cached teacher forward"** —— 实际比 one-forward 更强（完全跟训练 hot-path 解耦） |
| **on-policy 兼容** | ✓ | ✓ | ✗ —— 需要 off-policy KD 模式（student trained on teacher-greedy responses, not student rollouts） |
| **跟 Tier-2 路线关系** | 不耦合 | 不耦合 | **天然合流**（Tier-2a 在 [[handoff-2026-05-22-brief-v2-tier2-next.md]] 就指了 off-policy KD/SFT 作为第一 controlled arm） |
| **可逆性** | 跟 sglang patch 绑死 | A0 baseline 也得加载 HF | offline JSONL artifact，arms 切换不需要重训 |

### 我目前倾向 C，但有顾虑

倾向 C 的理由：
1. **§Method narrative 反而更干净**："zero-train-forward TAM-Boost via pre-cached evidence locator"比 one-forward 更强
2. **工作量最小**：复用 Step 1a 基础设施，扩到训练池 = 几小时 H800 时间
3. **跟 Tier-2 合流**：T2-1 失败转 T2-2 abandon 时本来就指过 off-policy KD 为第一 arm
4. **不动 sglang**：[[xbox-patches-p14-p18]] 已经有一堆 patch list，再加 sglang 改动是负担
5. **A0/A1 launcher 几乎不动 Uni-OPD 主代码**：只需一个新 extension hook 读 TAM JSONL

顾虑：
- C 的训练模式跟 T1 v1.5b（positive +23pp）的 on-policy 模式不一样。如果 off-policy KD 训出来的 baseline (A0) 自己就不如 T1_2（mean 0.553），TAM-Boost arm 跟 A0 比再有 Δ 也不能跟 T1_2 直接比，§Method 的强度会打折
- on-policy 的 TAM-Boost 才是"贴实际部署"的方法。off-policy 算先行验证

---

## Part 3: Questions for You

1. **架构选 A/B/C 哪个**？我的 lean 是 C，但你判断 §Method paper-worthy 强度可能更看重哪条。

2. 如果走 C：**A0 baseline 是 off-policy KD on 同一池子 + 同一 teacher-greedy 响应**——这个 A0 是不是该跟 T1_2 (on-policy, 0.553) 直接做一个 head-to-head 当 sanity check？还是 A0 自身就是 reference，不跟 T1_2 比？

3. 如果走 C：**TAM precompute 跑多大规模合适**？训练池 MMR1-RL ~15k，Tier-2 之前用 2-5k subset。3000 samples × 1 teacher forward × ~10s/sample ≈ 8 GPU-hours，全 15k 大概 40 GPU-hours。15k 全跑保险还是 3000 起步？

4. **on-policy/off-policy 切换的 ablation 顺序**：C 走通之后，做不做 B（双 teacher on-policy）做一次确认 on-policy ≈ off-policy 上的 TAM-Boost Δ？还是 paper 只 report off-policy results、on-policy 留作 future work？

5. **q3 内部 39% fire 的 narrative**：preflight v2 显示 within-C_local q3 仍 39% fire，跟 Step 2b "q3 不响应局部 mask" 在因果意义上不矛盾但需要解释。paper 该怎么 frame？
   - 选项 (i)：data-driven —— "gate fires on spatial co-occurrence; q3 boost contributes negligible (~0.2%) to training loss"
   - 选项 (ii)：方法 ablation —— A5 oracle quad arm 量化"如果排除 q3 boost 是否更好"
   - 选项 (iii)：直接在 paper §Method 里 declare "we accept residual q3 firing as known limitation"

6. **§Method 标题/positioning**：现在的 working title 是 "TAM-Evidence-Bottleneck OPD"。如果走 C，narrative 是 "precomputed TAM evidence cache + off-policy KD reweighting"，要不要换标题反映这一点？

---

## Response Format

- Q1-Q6 verdicts: agree / refine / block, 一句话
- 关键 §Method narrative 调整建议（如果有）
- One paragraph: greenlight 我开 C 的代码（precompute + post_process hook + loss.py patch + A0/A1 launcher），还是 block + 提出要先验证什么？

**请用中文回复。**
