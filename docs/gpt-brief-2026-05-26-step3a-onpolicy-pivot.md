# Step 3a 重审 — OPD on-policy 跟 cached TAM 的根本冲突

Repo: https://github.com/shihaohou/mllmopd @ `f79b6a4`

Relevant files:
- `docs/step3a-design-2026-05-26.md` — 当前 v2 设计（你 2026-05-26 verdict 后锁的）
- `docs/gpt-brief-2026-05-26-step3a-phase2-integration.md` — 上一份 brief（提了 A/B/C 三条路）
- `src/mllmopd/training/tam_gate.py` — gate 模块（独立)
- `src/mllmopd/training/tam_boost_hook.py` — hook（一旦 method 改了可能要重写）
- `scripts/audit/tam_step1a.py` — 已加 `MLLMOPD_TAM_PRECOMPUTE_ONLY` flag
- `third_party/Uni-OPD/miles/miles/backends/training_utils/loss.py` — P13/P22 weight 应用点（已 patch）

**请用中文回复**。

---

## TL;DR

我把 Phase 2 plumbing 搭通了（5 必修点全过、smoke A0/A1 都跑通），但在试图把它扩到训练池规模时**重审 §Method**，发现一个**前后不一致**之前没被点破，希望你重新拍板。

不一致：

1. 我们要做的方法是 **OPD = On-Policy Distillation**（按定义 student 自己 rollout，teacher score 它）
2. 你上一份 verdict 锁的 architecture **Path C = off-policy KD + cached TAM**，理由是 sglang teacher 只返 logp 不返 logits，没法 cheap on-the-fly 抽 TAM。所以**为了 cache 用得上**，训练 mode 必须改成 off-policy KD（student 训练在 teacher pre-generated response 上）
3. 这个 mode shift 把 "OPD" 这个方法名头丢了 —— 我们做的实际是 **offline KD + cached evidence reweighting**，不是 on-policy distillation 的改进

刚才我已经按 Path C 跑了 smoke + 在准备给训练池做 precompute（2 box × 8 GPU + FA2 + skip 不必要计算），但发现：
- A0/A1 smoke 用的 on-policy launcher，cache hit rate = 0% （结构性 mismatch，**不是** cache 在 audit subset 的问题）
- 要让 Path C 真生效，需要写 off-policy KD 训练 launcher（Uni-OPD 现有的是 on-policy）—— 还没动
- 在动这一步之前，发现 method 上的 mismatch 比工程上的更要紧

希望你帮我重新推一遍：**怎样在 on-policy OPD 下做 TAM-Boost**。

---

## 已经确定的事实（don't redo these）

### 来自机制实验（Step 1a/Step 2/Step 2b，2026-05-22 ~ 05-26）

1. **MLLM OPD 是 condition-sensitive 的**：T1 v1.5b FullTeacher (+1.3pp) vs BlankTeacher (−21.7pp) 差 +23pp，p≈10⁻⁶。Teacher 的视觉条件化行为是被忠实迁移的核心信号。
2. **Visual token signal sparse + signed**：Step 1a 显示 vd > 0.5 token 仅 6.8%（NLL mass 20%），vd ≤ −1 token 仅 0.6%（NLL mass 24.7%）。强视觉相关 token 稀疏。
3. **TAM scalar mass ≠ 因果**：Step 1a pearson r ≈ 0 on n=9118 content_noun。
4. **TAM top-K region IS 因果**：Step 2 v2 paired Δ = +0.988 nat vs random / +1.056 vs scrambled，content_noun Δ=+1.24。
5. **C_local = {content_noun, visual_attribute, proper_noun}** 是 Step 2 因果效应显著的类别集合（Δ ≥ +0.48 nat）。`visual_number` 接近零，已 held out。
6. **q3 (vd<0 ∧ adv<0) 在局部 mask 下因果效应近零**（Step 2b）—— visual rejection 是 distributed/prefix-conditioned，不是 local-evidence。

### 来自 Phase 1 preflight v2 (2026-05-26)

7. Gate `g_t = 1[c_t ∈ C_local] ∧ 1[coverage(topK(M_t), E_x) ≥ τ]` 在 τ=0.70 下：overall fire rate 12.4%、within-C_local q0/q1/q2/q3 fire 偏差 < 3pp、4 ckpt 之间 cross-ckpt 一致性 < 0.5pp。
8. q3 over-fire 的原假阳是 spaCy 把 MMR1 `\boxed{answer}` BPE pieces 误标 proper_noun 引起（已修，`9c3fb36`）。

### 来自 Phase 2 plumbing (2026-05-26)

9. Uni-OPD loss.py 已经有 `teacher_vd_weights_list` 镜像 pattern，加 `teacher_tam_weights_list` 三件套 patches (P20/P21/P22) 已 ship，verified。
10. tam_boost_hook.py unconditional ones-attach 行为已 smoke 通过。
11. precompute infrastructure 写完了（tam_step1a --skip-student + tam_precompute_train_pool.py 转换器 + 跨 box 多机分片 + 多进程 + FA2 + PRECOMPUTE_ONLY flag）。timing breakdown: gen_full_s=50s（HF generate autoregressive 慢，FA2 帮不了 decode）、logit_s=0.08s、tam_s=25.6s、attn_s=blank_s=0s（已跳过）。每 sample ~76s。

### Smoke A0/A1 结果（用 on-policy launcher 跑的）

12. A0 (USE_TAM_BOOST=0) 跑完 4 个 rollout step，weight sync 3.5s × 16it，ckpt 落盘。
13. A1 (USE_TAM_BOOST=1) hook log：
    ```
    [TAM-Boost] batch n=64  hit=0.000  miss=1.000  fallback=1.000
    ```
    cache 是 audit subset 的，sample_id 跟 MMR1-RL 训练池不重叠 ← **看起来**是 cache mismatch
    
    但**真问题是 on-policy 训练时 student 自己 rollout，response_hash 永远不可能跟 teacher pre-generated 一致**（fp 非确定性 [[teacher-greedy-fp-nondeterm]] + student ≠ teacher model）→ **任何 on-policy training 下 cache hit = 0**，无论 cache 是从哪建的。

---

## 真正的问题

**OPD 按定义 on-policy。Step 3a 应该做什么？**

约束：
- 真 OPD 训练 = student rollout（sglang :30001-30007）→ teacher score（sglang :30000 返 logp）→ OPD loss = teacher_lp − student_lp on student tokens
- TAM 计算需要 teacher 在 student token 上的**完整 logit 分布**（不是 logp）
- sglang teacher 不暴露 logits 给 trainer
- 加一个 HF teacher 并存仅为了抽 TAM（~20GB extra GPU、+5-10s/training step）是可行但工程量中等

Step 2 的因果结论非常稳：top-TAM region 在 C_local 上有 +1.24 nat Δ。这个信号**应该**能驱动 OPD 改进。但要让它在真 on-policy OPD 里 work，必须解决：

- **如何在每个 training step 拿到 student rollout 的 per-token TAM？**

---

## 给你 4 个具体问题（请逐个回答）

### Q1: §Method 立场

我们的 contribution 应该是哪一个？

- (a) **真 on-policy OPD + spatial TAM gate** —— 不妥协 method，付工程代价（HF teacher 并存 / sglang patch）
- (b) **真 on-policy OPD + category-only gate**（基于 Step 2 因果证据，不用空间覆盖）—— 工程零成本，但 §Method 强度降级为 "category-aware OPD reweighting"
- (c) **接受 off-policy KD + cached TAM**（你上次 verdict 的路径）—— 工程已搭得差不多但 §Method 不是 OPD 了
- (d) 其他你能想到的

我现在的直觉是 (b) 是 Step 3a 主版本，(a) 作 follow-up paper extension。但请 reframe 这个判断。

### Q2: 如果选 (b) category-only，§Method 是否还成立？

Step 2 的 contribution 是 "TAM region 因果支持 C_local 类 token"。如果方法变成 "boost C_local tokens 不看 TAM region"，TAM 是不是只在 motivation/diagnostic 里出现，不在主方法 trained-time 路径里？

如果是这样，方法名换成什么？

- (i) "Category-Aware OPD" —— 通用
- (ii) "Visual-Evidence-Guided OPD" —— 强调 Step 2 motivation
- (iii) "Sparse Visual-Conditioned OPD" —— 强调 condition-sensitive 主线（[[project-hypotheses]]）
- (iv) 其他

### Q3: 如果选 (a) HF teacher 并存，工程预算多少合适？

Pros:
- 真 OPD + 完整空间 gate，paper §Method 强
- A1 vs A5 oracle ablation 仍能跑

Cons:
- HF teacher ~20GB GPU；H800 trainer 已经被 student + ZeRO + optimizer states 吃，能否再塞下 7B teacher？
- 每 training step 多一次 teacher forward over student rollout (~5-10s eager / ~2-3s FA2)，可能 step time 翻 1.5-2×

如果 GPU mem 撑不住，把 teacher 放 Box 1 用 sglang patch 暴露 logits 是不是反而更轻？

### Q4: q3 (visual-rejection) 在 on-policy mode 下还要不要 boost？

之前 Step 2b 显示 q3 局部 mask 不响应、但 quad 的判定本身需要 vd ← 需要 blank forward ← 两次 teacher forward / step。即使选 (a)，q3 信息仍要额外开销。

或者直接接受 §Method scope：在 (b) 下，q3 内部 39% 仍 fire 但 boost 是 category-only，损失就是 0.2% boost budget 浪费在 q3，cosmetic 缺陷。

paper 怎么 frame 这一段？

---

## 我的具体 deliverable 计划（pending 你的 verdict）

如果你回 (b)：

- 改 `tam_boost_hook.py`：从 cache lookup-based → on-rollout-classification-based。读 sample.tokens (student rollout) → spaCy 分类 → w_t = 1 + α·1[c_t ∈ C_local]。零 cache 依赖。
- precompute infrastructure 不再是 §Method 主路径但保留作 Phase 2.2 ablation 数据源（可选）
- A1 launcher 直接复用现有 on-policy `opd_mmr1_3b_baseline_xbox.sh`，零改动 + MLLMOPD_USE_TAM_BOOST=1
- 跟 T1_2 (mean 0.553) 直接 A/B 对照

如果你回 (a) 或 (c)：写一份新的 design v3，我按那个执行。

---

## 回复格式

- Q1-Q4 逐个判定 + 简短理由
- §Method narrative + 标题建议
- 一段：greenlight 哪个路径 + 我下一步可以执行什么具体动作

**请用中文回复**。

---

## 关于上次 verdict 的复盘（不需要回应，但请你心里有数）

上一份 brief（`docs/gpt-brief-2026-05-26-step3a-phase2-integration.md`）问的是 A/B/C 工程路径选择 + A0 角色 + 规模 + q3 framing 等。

那份 brief 的 framing 隐含接受了 "off-policy KD 是 acceptable method substrate" 这个前提。我没意识到这个前提本身把 method 从 OPD 拽走了。我也没让你审视这个前提。

这一份是来 reframe 那个前提。
