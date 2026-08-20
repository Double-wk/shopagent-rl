# Beyond Task Success：购物 Agent 的约束忠实配对策略优化

> 当前论文主线（2026-08-19）
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
\text{Relevant change}\Rightarrow\text{appropriate policy change}
\]

\[
\text{Irrelevant change}\Rightarrow\text{policy invariance}
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

本文的贡献不是简单增加反事实样本，而是把 paired policy relation 作为优化目标。

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
- **RQ3 / H3**：模型能对 price、option 等 relevant intervention 正确改变，同时对 nuisance intervention 保持动作意图不变。
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

Relevant pair 还带有认证关系 `R*_j`。例如：

- 价格从预算内变为超预算：`(COMMIT, SEARCH_ALTERNATIVE)`；
- 目标规格从 M 变为 L：`(SELECT_M, SELECT_L)`；
- 当前选择从目标规格变为其他规格：`(COMMIT, SELECT_TARGET_OPTION)`。

对于 irrelevant intervention，期望动作意图保持不变。

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

### 7.3 Paired GRPO

同一 pair 的两侧 rollout 联合计算：

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

`R_cert` 防止两侧一起错或“永远 Search”；`R_pair` 则直接优化两个世界之间的关系。实现应比较规范化动作意图，
而不是比较动态 `click[...]` 字符串。

**实现状态（2026-08-20）**：历史 `joint_bonus`、`relational_advantage` 和
`relational_residual` 原型实际使用的是“两侧 certified action reward 的联合/残差”，
没有把 `expected_relation` 作为独立的 intent-level verifier 输入，因此不能直接当作
最终方法实现。当前代码已补上默认关闭的 `explicit_relation` 路径：显式记录
`predicted_intent`，按 `expected_relation` 计算 `R_pair`，同时保留 per-side `R_cert`，
避免把“两个侧面都答对”与“关系目标本身成立”混为一谈。

当前代码已加入默认关闭的 `explicit_relation` 实验模式：counterfactual rollout
输出 `predicted_intent`，pair 数据携带有序 `expected_relation`，训练器在 pair
匹配后记录 `relation_correct`，并将关系奖励与单侧 certified reward 分开。该模式
只有在 provenance-disjoint 数据和 matched smoke 通过后，才可用于正式主实验。

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

### Stage 2：Matched independent CF-GRPO——下一步优先

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

### Stage 3：Paired GRPO——论文主实验（尚未通过）

在 Stage 2 完全匹配的设置中加入 `R_pair`，目标是同时提升 strict success、relevant PRA，并保持 nuisance invariance。

当前证据不足以支持该主张：

- v1/v2 RelGRPO 只有 10-step smoke，且从 v4 corrective SFT 初始化；Independent PRA 为 76.97%，
  residual v2 为 75.84%，未超过对照。
- 从 clean SFT 开始的 b8n4 large-batch paired trial 虽跑到 100 steps，但 heldout PRA 仅 8.99%，
  应作为失败的 joint-bonus 工程/配方试验，不进入主表。
- 因此 Stage 3 当前状态是“工程闭环成立，方法效果未证实”，下一轮必须先修正
  intent-level relation certificate，再做 clean-init 的 matched smoke。

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
3. 从 clean SFT adapter 做 10-step matched smoke：Independent、joint relation、asymmetric residual；
   只允许 `R_pair` 分支不同，先确认 schema、pair matching、reward/advantage 日志完整。
4. Smoke 闸门：paired PRA 不得低于 Independent 超过 2 个百分点，original action accuracy 不得下降超过 2 个百分点；
   否则停止扩展训练，先重做 relation utility。
5. 通过 smoke 后再跑 200-step、batch 4、3 seeds；每个 checkpoint 立即导出到 `outputs/`，并记录
   Final-200 strict、Price/Option CF、nuisance invariance、PRA 和 paired bootstrap CI。
6. 只有 H2 在 matched 多 seed 上通过，才做 one-sided、no-nuisance、structured-format、constraint-type holdout；
   long-horizon、额外模型和外部环境放到最后。

## 14. 论文六句话故事

1. Shopping agents are usually evaluated by whether they successfully purchase the target product.
2. We show that high task success can coexist with severe constraint blindness: an RL agent reaches 40% strict success while achieving 0% accuracy under minimal price interventions.
3. This reveals a gap between outcome success and constraint-faithful decision making.
4. We construct programmatically certified atomic intervention pairs that change exactly one decision-relevant or irrelevant factor.
5. We introduce paired intervention optimization, which explicitly trains the policy to change appropriately under relevant interventions while remaining invariant to irrelevant perturbations.
6. We evaluate agents jointly by normal shopping success and paired constraint fidelity, including generalization to unseen constraints, products, and models.

最终贡献压缩为三点：问题发现、程序认证的原子 pair 与 relation-level policy optimization、以及补足 task success 的 sensitivity/invariance/PRA 评测。
