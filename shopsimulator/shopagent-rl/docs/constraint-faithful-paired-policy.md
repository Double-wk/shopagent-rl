# Beyond Task Success：购物 Agent 的约束忠实配对策略优化

> 当前论文主线（2026-08-20）
> 暂定标题：**Beyond Task Success: Certified Paired Intervention Learning for Shopping Agents**
> 暂定方法名：**Constraint-Faithful Paired Policy Optimization**

## 1. 研究定位

本文只研究一个问题：

> **当决定性用户约束发生最小变化时，购物策略是否产生正确变化；当无关因素变化时，策略是否保持稳定？**

核心现象是：

\[
\boxed{\text{Task Success}\not\Rightarrow\text{Constraint-Faithful Decision Making}}
\]

购物 Agent 可能学会 `Search → Click → Buy` 的高频捷径，却没有真正执行
`Check Constraint → Compare → Decide`。本文将其称为 **Constraint-Shortcut Behavior**，也可称
**Outcome–Constraint Decoupling**。

当前论文不同时纳入 `Constraint Ledger → Residual Risk → Active Verification`。那条路线研究的是
“尚未确认的约束中应该验证什么、什么时候可以买”，留作后续独立论文 **Verify Before You Buy**。

| 研究线 | 核心问题 | 当前处理 |
|---|---|---|
| Constraint-Faithful Paired Policy | 约束变了，策略是否正确变化？ | **本文主线** |
| Verify Before You Buy | 哪个未决约束值得验证、何时安全提交？ | 后续论文 |

## 2. 研究背景

购物 Agent 通常按最终结果评测：是否找到商品、选对规格、满足预算并完成购买。这种 outcome 指标是必要的，
但不能证明策略是依据用户约束作出决定。一个模型可以在常规任务上取得较高成功率，同时在只改变预算或规格后
仍执行原来的购买动作。

因此，本文不把“会不会买到一个看起来合适的商品”作为唯一问题，而是问：

\[
\text{Decision-changing intervention}\Rightarrow\text{appropriate policy change}
\]

\[
\text{Decision-preserving intervention}\Rightarrow\text{policy invariance}
\]

这将研究对象从单个状态上的动作正确性，提升为一对状态之间的策略关系。

## 3. 以往研究及其边界

### 3.1 Shopping Agent、环境与结果优化

- [ShopSimulator](https://arxiv.org/html/2601.18225v1) 提供购物环境和 RL 训练框架，并报告 constraint violation、
  AskShopper 不充分和 premature Buy。它证明问题真实存在，但没有直接认证“高成功策略是否依赖决定性约束”。
- [Shop-R1](https://arxiv.org/abs/2507.17842) 用 RL 优化购物行为模拟与动作/理由生成，重点是分层奖励和难度适配，
  不是原子干预下的配对策略关系。
- [ShoppingComp](https://arxiv.org/html/2511.22978v2) 强调多约束与 safety-critical shopping，说明约束错误有现实后果，
  但不是本文的 relation-level training 方法。
- [EComAgentBench](https://arxiv.org/html/2606.17698v1) 覆盖隐藏意图、澄清、评论证据和长程工具使用，适合作为外部泛化环境。

这些工作主要回答“Agent 能否完成购物任务”，不能单凭 task success 判断“Agent 是否因为正确使用约束而成功”。

### 3.2 Constraint-aware learning

[CARL](https://arxiv.org/html/2607.04854) 已直接研究 constraint-aware reinforcement learning。因此本文不声称
“首次让购物 Agent 遵守约束”，也不把贡献写成一般性的 constraint-aware reward。

区别在于训练对象：CARL 类方法主要约束单状态行为；本文把 `(x, x')` 与 `(a, a')` 的**配对关系**作为一等目标。

### 3.3 信息获取与 explore-or-commit

[Entropy-IDSS](https://arxiv.org/abs/2603.11399) 研究基于信息增益的偏好询问；
[Calibrate-Then-Act](https://arxiv.org/html/2602.16699v3) 研究不确定条件下的探索/提交权衡；
[Paying to Know](https://arxiv.org/html/2606.24783v1) 讨论有成本的 verified product information acquisition。

它们与后续 **Verify Before You Buy** 直接相关，但当前论文假定干预后的信息已经出现，研究的是策略是否正确利用该变化。
因此 residual risk、conformal calibration、active verification 和 calibrated stopping 不进入当前方法主线。

### 3.4 研究空位

现有研究分别覆盖 outcome optimization、constraint-aware reward、information acquisition 和 explore-or-commit，
但仍缺少：

> **对只改变一个决定性约束的配对状态进行程序认证，并直接训练策略满足两侧行为关系。**

本文的贡献不是简单增加反事实样本，而是把 interactive agent 的 intervention-conditioned policy
response 作为优化目标；不争夺一般性的“首次 counterfactual consistency”表述。

## 4. 当前项目的实证起点

### 4.1 关键动机证据

当前 GRPO Paired-C1-hard 在 Final-200 上达到 40% strict success，但价格干预几乎完全无效：

| 指标 | 结果 |
|---|---:|
| Final-200 strict success | **40.0%（80/200）** |
| 完成率 | **90.0%** |
| 选对商品率 | **45.0%** |
| Option-swap paired robust | **73.1%（49/67）** |
| Price-CF Accuracy | **0.0%（0/145）** |
| Price same-action-both-sides | **98.6%** |

145 个仅把价格改到预算以上的 pair 中，143 个两侧仍执行相同购买动作。原状态动作准确率仍为 93.1%，
说明模型并非完全不会购物，而是没有把价格比较绑定到提交决策。

这构成本文的起点：

> **高 task success 与严重 constraint blindness 可以同时存在。**

### 4.2 SFT 与 shortcut 审计

| 阶段 | 自然格式结果 | Final-200 strict | 结论 |
|---|---:|---:|---|
| SFT v2 Paired | option paired robust 73.1%；price 0% | 25.0% | 配对规格监督有效，但未覆盖价格 |
| Certified SFT v3 | price CF 0%；恢复负例专属 summary 后 100% | — | summary-presence shortcut |
| **Corrective SFT v4** | **price CF 78.65%；price PRA 73.70%** | **31.5%** | 自然双侧配对监督可行 |
| Explicit-Clean SFT v5 | natural price CF 0.26%；显式 price CF 100% | 不进入后续 RL | 显式格式捷径失败消融 |

v4 的 10,057 条混合训练集只保留为 **feasibility/corrective ablation**，不作为主实验初始化。
v3/v5 说明 summary、显式字段和模板格式都可能伪造高反事实分数。因此所有 headline 结果必须来自 natural atomic split，
structured summary 只能作诊断。

正式主实验使用独立版本 `horizon10-clean-v1`：从固定 revision 的
`Qwen/Qwen3-1.7B-Base` 直接训练 `data/sft_train_horizon10.jsonl` 的 3,624 条普通
strict-success 轨迹，不加载 v1--v5 的任何 adapter；随后从同一个 clean adapter 启动
Independent GRPO 与 Paired GRPO。完整协议和数据哈希记录在
[`experiments/manifests/horizon10_clean_v1.yaml`](../experiments/manifests/horizon10_clean_v1.yaml)。

## 5. 研究问题与假设

- **RQ1 / H1**：Outcome RL 可以提升 strict success，却不提升甚至降低 relevant paired robustness。
- **RQ2 / H2**：在数据、verifier、token budget 和初始化完全匹配时，显式 pair-relation objective 优于独立 CF reward。
- **RQ3 / H3**：模型能对 verifier-certified decision-changing intervention 正确改变，同时对
  decision-preserving intervention 保持动作意图不变。
- **RQ4 / H4**：收益可泛化到未见任务、商品、类别、约束组合和至少一种未见约束类型。

## 6. 问题形式化

设状态为 `x`，策略输出动作意图 `a ~ πθ(·|x)`。对单个因素 `c_j` 做原子干预：

\[
x' = I_j(x),\qquad \Delta(x,x')=c_j.
\]

确定性 verifier 根据商品目录、预算、可用规格和环境规则计算：

\[
a^*=V(x),\qquad a'^*=V(x').
\]

Decision-changing pair 还带有认证关系 `R*_j`。例如：

- 价格从预算内变为超预算：`(COMMIT, SEARCH_ALTERNATIVE)`；
- 目标规格从 M 变为 L：`(SELECT_M, SELECT_L)`；
- 当前选择从目标规格变为其他规格：`(COMMIT, SELECT_TARGET_OPTION)`。

对于 decision-preserving intervention，verifier 必须认证 `V(x)=V(x')`，并确认 feasible set、
商品身份和决定性约束没有变化；不能仅凭“品牌/包装看起来无关”的人工直觉定义。

本文使用 **intervention-faithful / constraint-faithful**，不把受控行为关系夸大为识别模型内部因果机制。

## 7. 方法：Constraint-Faithful Paired Policy Optimization

### 7.1 Certified Atomic Intervention

构造 `(x, x')`，只改变一个 price、size、color、material、capacity、compatibility 或 nuisance 因素。
每条 pair 必须通过程序门：商品身份和非目标字段一致；自然指令的其他数字和规格 token 不漂移；两侧动作合法；标签由 verifier
计算；拒绝 pair 与原因落盘。教师模型只负责表面改写，不能自证标签。

### 7.2 Certified Paired SFT

两侧分别做 action imitation：

\[
\mathcal L_{action}=-\log\pi_\theta(a^*|x)-\log\pi_\theta(a'^*|x').
\]

数据必须保留 `pair_id`、`side`、`intervention_type` 和 `expected_relation`。不能把 pair 最终混成 10,057 条无关独立样本，
否则后续 RL 无法使用关系标签。

v4 证明自然双侧 paired SFT 可恢复 price sensitivity，但不是最终方法贡献；下一步必须比较 relation-level GRPO。

### 7.3 Policy-Response Relation Optimization

当前离散实现同一 pair 的两侧 rollout 联合计算：

\[
R=R_{task}+\lambda_{cert}R_{cert}+\lambda_{pair}R_{pair}.
\]

\[
R_{cert}=\tfrac12(\mathbb 1[a=V(x)]+\mathbb 1[a'=V(x')]).
\]

Relevant intervention：

\[
R_{pair}^{rel}=\mathbb 1[(intent(a),intent(a'))\in R_j^*].
\]

Irrelevant intervention：

\[
R_{pair}^{irr}=\mathbb 1[intent(a)=intent(a')].
\]

`R_cert` 防止两侧一起错或“永远 Search”。但是当 `expected_relation` 恰好等于两侧 verifier
标签的合取时，`R_pair` 只是“两个单侧都正确再加 bonus”，不能充分区别于普通 counterfactual
augmentation。这一路径保留为 **conjunctive relation baseline**。

论文主方法改为直接约束策略 preference。令原状态与干预状态的两个竞争 intent 为
`z_o,z_c`，定义 interventional preference margin：

\[
M_\theta(x,x')=
\log\frac{\pi_\theta(z_o|x)}{\pi_\theta(z_c|x)}-
\log\frac{\pi_\theta(z_o|x')}{\pi_\theta(z_c|x')}.
\]

对 decision-changing intervention 使用带 margin 的 flip loss：

\[
\mathcal L_{flip}=-\log\sigma((M_\theta-m)/\tau).
\]

对 decision-preserving intervention，在 canonical intent distribution 上使用对称 preserve loss，
首选 `D_JS(\pi^{intent}(\cdot|x),\pi^{intent}(\cdot|x'))`。完整目标为：

\[
\mathcal L=\mathcal L_{task}+\lambda_{anchor}\mathcal L_{side}
+\lambda_{flip}\mathcal L_{flip}+\lambda_{preserve}\mathcal L_{preserve}.
\]

技术核心由此统一为 **Flip when it matters, preserve when it does not**。

**实现状态（2026-08-20）**：历史 `joint_bonus`、`relational_advantage` 和
`relational_residual` 原型实际使用的是“两侧 certified action reward 的联合/残差”，
没有把 `expected_relation` 作为独立的 intent-level verifier 输入，因此不能直接当作
最终方法实现。当前代码已补上默认关闭的 `explicit_relation` 路径：显式记录
`predicted_intent`，按 `expected_relation` 计算 `R_pair`，同时保留 per-side `R_cert`，
避免把“两个侧面都答对”与“关系目标本身成立”混为一谈。

当前代码已加入默认关闭的 `explicit_relation` baseline：counterfactual rollout
输出 `predicted_intent`，pair 数据携带有序 `expected_relation`，训练器在 pair
匹配后记录 `relation_correct`，并将关系奖励与单侧 certified reward 分开。该模式
已在 provenance-disjoint 数据上完成 clean-init 10-step smoke。4 个 counterfactual block 均成功匹配
`2 relations / 8 rollout pairs`，但 relation success 全为 0，显示离散 conjunctive bonus 无法从
clean init bootstrap；因此它不再承担主方法主张。

`preference_margin` 现已完成可反传实现（Gate A 通过，未做实证训练验证）。

**打分基础。** canonical intent 的分布来自**受限动作集上的归一化**：解析状态里渲染的
`可点击的按钮` 列表得到合法动作，对每个 `click[...]` 候选做一次批量前向打分，在这个集合上
`log_softmax`，再按 canonical intent 用 `logsumexp` 聚合。不用固定短语打分（`INTENT_PHRASES`
存在前缀包含缺陷），也不用 `allowed_actions` 列。实测 400/400 CF 行都能解析出合法集，每个
`(intervention_type, side)` 的期望 intent 都可达，200/200 pair 的两个被比较 intent 在两侧都合法，
因此 margin 从不为 ±inf。某 intent 在该状态无合法动作时返回 `-inf`（“不可用”，不是“不偏好”），
调用方必须丢弃而不是当成低概率。

**接入位置。** relation loss 是 loss 而不是 advantage 重整，所以 advantage 阶段现在按设计什么都不做
（原先的 `add_preference_margin_loss` 打印看似合理的 flip/preserve 数值却返回 `advantages.clone()`，
且位于任何前向之外，不可能可导；该函数已标注 SUPERSEDED）。真正的项在 actor 侧：
`ActorRolloutRefWorker._maybe_wrap_relation_loss` 只在 `mode=preference_margin` 时包装
`ppo_loss`，其余模式（含 Independent baseline）拿到的 `loss_fn` 与包装前逐字节相同，这正是
Gate B 的前提。pair 元数据（`pair_id`/`side`/`state_text`/`expected_relation`/
`expected_action_intents`）经 agent loop 的 `extra_fields` 带到 micro-batch。

**pair 如何在 micro-batch 内闭合。** relation loss 需要两侧，但
`ppo_micro_batch_size_per_gpu=1`（为 48 GiB 显存刻意设定）下两侧永不同批。曾用
`force_group_size=2×rollout.n` 强行合并，实测把 16 个 size-1 micro-batch 压成 2 个 size-8：
PPO 自身显存 ×8，且**只对 paired arm** 改变梯度累积粒度，直接破坏“baseline 不变”的对照前提。
现改为：`RayPPOTrainer._maybe_attach_partner_states` 在 trainer 侧（能看到整个 batch）把对侧
state 作为 `partner_state_text` 挂到每个 `original` 行上。因为 relation loss 只读 state（即该行
prompt），而同一侧的 `n` 个 rollout 共享同一 prompt，携带与共批完全等价。PPO 的 micro-batching
保持原样。真实路径已验证（`to_tensordict` → `left_right_2_no_padding` →
`prepare_micro_batches`）：8 个 size-1 micro-batch，其中 4 个各自闭合成完整 pair。

**累积归一化。** engine 对 micro-batch loss 直接求和（无 `1/N`），PPO 项自身按 token 数归一，
relation 项不会。若不处理，其有效权重等于“携带 pair 的 micro-batch 个数”（`rollout.n=4` 时为 8），
会随 `rollout.n`、`ppo_micro_batch_size` 静默漂移。故 `TrainingWorker.train_batch` 传入 mini-batch
的 pair 行数，hook 按 `pair_rows_used / mini_batch_pair_rows` 重标定，使累积和恰为 mini-batch 均值。
分子必须按**行**计数：同一侧 `n` 个 rollout 去重成 1 个 pair 却被累积 `n` 次，用 `pairs_used`
会与分母单位不一致。已验证 `rollout.n ∈ {1,2,4,8}` 与 pair 数 ∈ {1,2} 下取值恒定，
`relation_coeff` 线性生效。

**correctness anchor。** margin 是“差之差”，单靠它可以通过让某一侧的**错误** intent 变得不那么错来
满足；因此每侧另加 `-log P(expected intent | state)` 锚定。

**Gate A 梯度验收**（`scratch/verify_gate_a_gradients.py`，真实 Qwen3-1.7B + 真实 pair 行）：
四项 gating 全部通过。

| 项 | 结果 |
| --- | --- |
| A 梯度非零 | PASS，`grad_norm=4.49e+02` |
| B flip 方向 | PASS，`cos(-∇L, +∇M) = +0.965`（flip-only），加 anchor 后 `+0.196` 仍同向 |
| C preserve 方向 | PASS，`cos(-∇L, -∇JS) = +1.000`（preserve-only） |
| D baseline 隔离 | PASS，无 pair 行时 wrapper 返回同一个 loss 对象，不产生梯度，不写 `relation/*` |

方向用**方向导数**判定而非有限步：在 `grad_norm≈4e2` 下即使 `lr=1e-4` 也已远离一阶邻域，
观测到的 margin 变化由曲率主导（首次用有限步测得 margin 反向下降，即此原因）。

**已量测的权重张力（未改默认值）。** 在 clean init 上 anchor≈7.32 而 preserve JS≈0.0075；
`anchor_weight ≥ 0.03` 时合成梯度**反向**推高 JS（1.0 时 `cos=-0.34`），仅在 `≤ 0.01` 才降 JS。
默认仍保留 `1.0`：clean init 下 `P(COMMIT)≈2.6%`，两侧都答错，保持一个错误分布没有价值——
先修 intent，anchor 收敛后 JS 自然接手。若 Gate B 的 Irrelevant Invariance 不改善，此处优先调整。
flip 不受影响。

**结构性注意。** decision-preserving pair 的 `intent_original == intent_cf`，故
`M = (a-a) - (b-b) ≡ 0` 与分布无关；`margin_mean` 因此只对 decision-changing pair 求平均，
否则会被结构零稀释、看起来像 flip 项没在动。preserve 侧要看 JS，不能看 M。

**Provenance。** paper-v1 GRPO 的 600 个 task 与实际初始化 adapter 的 SFT 数据
（`sft_train_horizon10.jsonl`，3624 task）交集为 0，与三个 certified SFT 集交集同为 0；
`pair_environment_task_overlap`、`final_test_task_overlap`、`final_test_product_overlap` 均为 0。

**Gate B matched smoke（10 步，clean init，TRAIN_BATCH=4 / ROLLOUT_N=4）。**
paired arm 跑完 10 步无报错，峰值 25.4 GiB / 48 GiB。relation loss 确实在动：
`accumulation_scale=0.125`（恰为 1/8，归一化生效）、`pairs_from_partner_state` 与 `pairs_used`
一致（每步 8 个 pair 行、2 个 distinct pair），preserve-only 步 `margin_mean=0`，
decision-changing 步 `margin_mean=+0.170 / +0.117`、`flip_rate=0.31 / 0.375`。
baseline arm 全程 0 条 `relation/*`、0 条 `PreferenceMargin` 打印，wrapper 从未介入。

pair 只落在 step 3–6：parquet 的 block 结构是 `E×8, P×16, E×16, …`，`TRAIN_BATCH=4`
下 step 1–2 取到 environment、step 3–6 取到那 16 行 pair block、step 7–10 又是 environment。
这是 pairblocked 设计的必然结果，不是缺陷；400/400 的总量在长 run 上会摊平。

**过程中修掉的三个真实阻塞（都不是靠放宽判据绕过的）。**

1. *checkpointing 探测位置错误。* `gradient_checkpointing_enable()` 从不在顶层 module 上置位，
   而是置在拥有被 checkpoint 的块的子模块上（`Qwen3ForCausalLM` 是内层 `Qwen3Model` 及其
   decoder layer）。原先只查顶层属性，于是**正确配置的模型被判为未开启**。改用 transformers
   自己的递归判据 `is_gradient_checkpointing`，并对不转发该属性的 wrapper 回退到走 `modules()`。
   在真实 Qwen3-1.7B 与 PEFT 包装下都已验证。
2. *activation offload 与本目标不兼容。* veRL 不是在 transformers checkpointing 之上叠加 offload，
   而是**替换**掉它（两者不兼容），并用一个按层推进的 commit 计数器索引
   `layer_window_map`，其大小恰好是"每 step 一次前向"。relation loss 会额外发起前向，计数器越界——
   28 层模型上表现为 `KeyError: 27`，正好溢出一整趟。已改为 `enable_activation_offload: False`，
   且对**所有** arm 生效（只关 paired arm 会破坏对照），并在
   `_maybe_wrap_relation_loss` 里加了 init 期硬失败，避免几分钟后才炸。关掉后 PPO 自身峰值 18.9 GiB，
   容得下。
3. *`use_fused_kernels=True` 下没有 `.logits`。* 该 patch 把 forward 换成融合 LM-head 投影、
   返回 `CausalLMOutputForPPO(log_probs=…, entropy=…)` 的版本，读 `.logits` 直接抛错。
   其 `log_probs[t] = log P(input_ids[t+1] | prefix)` 恰好就是本文 scoring 需要的逐 token 量，
   故直接取用；在真实模型与真实 pair state 上与原 windowed-logits 路径最大绝对差 `2.4e-07`。
   代价是该 forward 忽略 `logits_to_keep`，显存 10.12 GiB vs 4.96 GiB——都放得下，
   走这一支是为了在实际配置下**正确**，不是为了省。

**checkpoint 复用陷阱。** `resume_mode: auto` 会自动接上同名 output dir 里的
`latest_checkpointed_iteration.txt`。第一次 baseline 是在 offload 开启下跑的，改配置后重跑时它
静默地从 `global_step_10` 续到了 step 11——对照就不成立了。该目录已移到
`*_offload_on_invalid`，baseline 在新配置下从头重跑。比较两个 arm 前务必确认日志里是
`Training from scratch` 而不是 `Resuming from …`。

## 8. 评测指标与协议

主表只突出四类指标：

1. **Final-200 Strict Success**：正常购物能力；同时报告 completion、`r_hard`、`r_loose` 和步数。
2. **Relevant Sensitivity**：Price-CF Accuracy、Option-CF Accuracy，以及后续 capacity/quantity/compatibility CF。
3. **Irrelevant Invariance**：无关干预前后动作意图保持正确且稳定的比例。
4. **Paired Robust Accuracy（PRA）**：

\[
\mathrm{PRA}=\frac1N\sum_i\mathbb 1[a_i=a_i^*\land a'_i=a_i'^*].
\]

PRA 防止模型通过“永远 Search”在反事实侧取得虚假高分。

评测要求：headline 只用 natural atomic split；train/test 按 task、product、category 隔离；Final-200 至少 3 个价格随机化 seed；
pair 指标使用 paired bootstrap CI；结构化 summary 只作为 shortcut 诊断；最终论文前审计自然指令 budget 与 verifier budget 一致。

## 9. Baselines 与关键消融

| Method | Final Strict ↑ | Price CF ↑ | Option CF ↑ | Invariance ↑ | PRA ↑ |
|---|---:|---:|---:|---:|---:|
| Base |  |  |  |  |  |
| SFT |  |  |  |  |  |
| Vanilla GRPO |  |  |  |  |  |
| Hard Reward GRPO |  |  |  |  |  |
| CF-SFT |  |  |  |  |  |
| **CF-GRPO w/o Pair** |  |  |  |  |  |
| **Ours: Paired GRPO** |  |  |  |  |  |

关键 baseline `CF-GRPO w/o Pair` 必须使用完全相同的数据、verifier、初始 adapter、token budget、rollout 和 seed，
唯一移除 `R_pair`。否则无法区分 paired optimization 与普通 CF augmentation/reward shaping。

必须补做：去掉 pair reward、去掉 nuisance reward、独立 sampling、one-sided hard negative、structured-format、
no-nuisance 和 constraint-type holdout 消融。

## 10. 研究阶段与下一步

### Stage 0：问题发现——已完成

GRPO Paired-C1-hard 建立了 **40% strict / 0% Price-CF** 的解耦证据。

### Stage 1：Certified SFT feasibility——已完成

当前论文使用从 3,793 条 terra strict 母集中筛出的 `n_steps ≤ 10` 子集（独立文件 `data/sft_train_horizon10.jsonl`，共 3,624 条）。
正式 clean SFT 直接从 base model 训练该子集；v4 aligned mix 的 natural heldout-v2 结果
（price CF 78.65%、price PRA 73.70%、Final-200 strict 31.5%）仅作为 feasibility 记录。
超过 10 步的 169 条母集轨迹保留作后续 long-horizon 扩展，详见 [`trajectory-horizon-and-long-horizon.md`](trajectory-horizon-and-long-horizon.md)。

### Stage 2：Matched independent CF-GRPO——基线与重训准备

从同一 clean SFT adapter 起训，使用 `R_task + R_cert`，但不使用 `R_pair`。它测试普通 RL 是否改变
基础购物 SFT 的 constraint fidelity，并作为完整方法的严格 matched baseline。

当前正式协议使用 `data/grpo_certified_natural_800_pairblocked.parquet`：400 个 environment row
与 200 个完整 pair（100 price、60 option、40 nuisance）共 800 行。每 4 行构成一个不可拆分 block，
其中 counterfactual block 含两个完整 pair；训练固定 `TRAIN_BATCH=4`、`TOTAL_STEPS=200`、
`DATA_SHUFFLE=False` 和 `trainer.balance_batch=False`，正好完整读取一遍数据，并确保 original/CF
rollout 在计算 joint reward 时仍可一一匹配。数据构造阶段先按 3,000 字符门槛排除整对超长
counterfactual prompt，再从候选池补足各类别配额；当前文件最大 prompt 为 1,894 tokens，800 行
均能通过训练端的 2,048-token 检查，不会因静默过滤而拆散 pair。

Stage 2 曾完成过单 seed、200 step 的正式运行（Final-200 strict 35%，整体 PRA 43.63%），
但 adapter 和中间 checkpoint 只存在临时 `/overlay`，实例重启后丢失。因此该数字只能作为
单 seed 历史记录，不能与后续重训结果混合；论文的 matched comparison 仍需从 clean SFT
重新训练并导出可复现 adapter。

### Stage 3：Policy-Response Relation Optimization——论文主实验（实现尚未完成）

在 Stage 2 完全匹配的设置中加入 preference flip/preserve objective，目标是同时提升 strict success、
decision-changing PRA，并保持 decision-preserving invariance。离散 `R_pair` 只作 conjunctive baseline。

当前证据不足以支持该主张：

- v1/v2 RelGRPO 只有 10-step smoke，且从 v4 corrective SFT 初始化；Independent PRA 为 76.97%，
  residual v2 为 75.84%，未超过对照。
- 从 clean SFT 开始的 b8n4 large-batch paired trial 虽跑到 100 steps，但 heldout PRA 仅 8.99%，
  应作为失败的 joint-bonus 工程/配方试验，不进入主表。
- 2026-08-20 的 provenance-disjoint clean smoke 中，Independent 与 `explicit_relation` 均完成 10 steps；
  pair matching 正常，但所有 matched relation bonus 为 0。
- 当前 `preference_margin` smoke 入口只验证 metadata、数学原型和日志路径；由于使用预测 intent
  proxy 且不回写 advantages，它不能证明 policy relation 被优化。
- 因此 Stage 3 当前状态是“baseline 工程闭环成立，conjunctive reward 存在 bootstrap failure，
  policy-score-level 主方法尚未接通，效果未证实”。下一步必须实现真实 canonical-intent scoring、
  可反传的 flip/preserve objective 和 gradient smoke，再进行 matched 比较。

### Stage 4：泛化与机制分析

测试未见商品/类别/约束组合、constraint-type holdout、额外模型尺度，以及条件允许时的 EComAgentBench transfer；
绘制 success 与 PRA 的 checkpoint trajectory，分析二者是否解耦。

## 11. 风险与审计

- **格式捷径**：natural headline、双侧同格式、summary-positive nuisance、heldout paraphrase。
- **永远 Search**：同时报告 original accuracy、Final strict、PRA 和 task reward。
- **非原子干预**：token preservation、catalog consistency、合法动作验证、拒绝样本审计。
- **数据泄漏**：Final-200/heldout pair 不回流训练；近重复商品也隔离。
- **因果表述过强**：只声称受控干预下的 behavior relation，不声称内部 causal identification。
- **单轮探针偏差**：最终补充多轮 paired rollout，确认 probe 改善转化为完整任务收益。

## 12. 与 Verify Before You Buy 的边界

当前论文结束在：**信息已经出现后，模型是否正确利用该约束。**

后续论文从这里继续：**信息尚未出现时，应该主动验证哪个约束，以及何时足够安全到可以购买。**

后续可研究 Constraint Ledger、Residual Constraint Risk、Expected Risk Reduction 和 calibrated commit，
但不进入当前论文主图、主目标或主实验表。

## 13. 近期执行清单

1. 冻结 evaluator、natural heldout split、verifier budget 对齐和 pair 数据 hash；任何方法调参不得改动这四项。
2. 在 `counterfactual_grading.py` 中补齐 `action -> intent` 规范化和 `expected_relation` verifier，
   用 CPU 单元测试覆盖 price、option、nuisance、lenient search 和 malformed action。
3. clean-init Independent 与 conjunctive `explicit_relation` smoke 已完成，作为训练闭环和稀疏奖励基线归档。
4. 实现真实 canonical intent scoring、可反传的 preference flip 与 decision-preserving JS loss，
   保留 per-side correctness anchor，再做严格 matched smoke；
   若 PRA 低于 Independent 或 original accuracy 明显下降，停止扩展训练并检查 intent projection/credit assignment。
5. 通过 smoke 后再跑 200-step、batch 4、3 seeds；每个 checkpoint 立即导出到 `outputs/`，并记录
   Final-200 strict、Price/Option CF、nuisance invariance、PRA 和 paired bootstrap CI。
6. 只有 H2 在 matched 多 seed 上通过，才做 one-sided、no-nuisance、structured-format、constraint-type holdout；
   long-horizon、额外模型和外部环境放到最后。

## 14. 论文六句话故事

1. Shopping agents are usually evaluated by whether they successfully purchase the target product.
2. We show that high task success can coexist with severe constraint blindness: an RL agent reaches 40% strict success while achieving 0% accuracy under minimal price interventions.
3. This reveals a gap between outcome success and constraint-faithful decision making.
4. We construct programmatically certified atomic interventions labeled as decision-changing or decision-preserving.
5. We optimize policy response relations directly: flip intent preferences when the certified decision changes and preserve them when it does not.
6. We evaluate agents jointly by normal shopping success and paired constraint fidelity, including generalization to unseen constraints, products, and models.

最终贡献压缩为三点：问题发现、程序认证的原子 pair 与 relation-level policy optimization、以及补足 task success 的 sensitivity/invariance/PRA 评测。
