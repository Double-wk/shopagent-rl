# 论文整理路线图

> 更新时间：2026-08-20

这份文档把当前研究拆成“可以写进论文的事实”“仍是待验证假设”和“下一轮必须完成的实验”。
目标是先形成一条可证伪、可复现的主线，再扩展长轨迹和外部泛化。

## 1. 当前论文可以承诺什么

论文的可靠主张是：

1. 普通 task success 不足以证明购物 Agent 使用了决定性用户约束。
2. Paired-C1-hard 已给出强动机证据：Final-200 strict 40%，但 price counterfactual accuracy 约为 0%。
3. 程序生成的 atomic intervention pair 可以在不依赖 LLM 自评的情况下认证相关/无关约束变化。
4. 自然格式的 paired corrective SFT 能显著恢复价格敏感性；它是 feasibility 证据，不是最终 RL 方法胜出证据。
5. Clean-init Independent 与离散 `explicit_relation` 的 10-step smoke 均已跑通；后者在 4 个
   counterfactual block 中均成功匹配 `2 relations / 8 rollout pairs`，但 relation success 为 0，
   尚未证明它优于 matched Independent CF-GRPO。

最后一点必须写进限制与当前状态，不能用“方法已提出”替代“方法已验证”。

## 2. 结果分层

| 层级 | 结果 | 论文用途 |
|---|---|---|
| 动机主结果 | Paired-C1-hard：strict 40%，price CF 0% | 主图/问题定义 |
| 可行性结果 | SFT v4：natural price CF 78.65%，PRA 73.70% | feasibility ablation |
| clean matched 记录 | Independent CF-GRPO：单 seed strict 35%，PRA 43.63% | 仅作历史记录，重训前不进最终主表 |
| RelGRPO smoke | Independent 76.97%，residual v2 75.84% PRA | 方法未胜出，写入负结果/诊断 |
| clean b8n4 trial | paired PRA 8.99% | 失败配方，不进 headline |

所有反事实数字都必须注明初始化、训练步数、heldout split、seed 和是否为 smoke。

## 3. 方法实现的关键缺口

当前 pair reward 主要根据两侧 certified action reward 做 joint 或 residual coupling，
而不是直接验证 `expected_relation`。这会让“两个侧面都正确”和“关系目标成立”在代码中重合，
削弱论文方法定义的可辨识性。

下一版应让每条 rollout 额外记录：

- `predicted_intent`：由动作解析器规范化得到，例如 `COMMIT`、`SEARCH_ALTERNATIVE`、`SELECT_TARGET_OPTION`；
- `expected_relation`：由 pair 构造器给出，例如 `(COMMIT, SEARCH_ALTERNATIVE)`；
- `relation_correct`：由 verifier 对两侧 intent 联合判定；
- `side_correct`：两侧各自相对 `expected_action_intents` 的结果。

训练中保留 `R_cert` 作为单侧约束正确性，评测同时报告关系正确性与单侧正确性，避免模型通过
“两侧一起错”或“永远 Search”取得虚假收益。但离散 `R_pair` 不再作为论文主方法的充分定义。

2026-08-20 已补上默认关闭的第一版 `explicit_relation` 路径：数据 builder 生成有序
`expected_relation`，rollout 记录规范化 `predicted_intent` / `side_correct`，trainer
在匹配两侧后记录 `relation_correct` 并单独加入关系 bonus。CPU 单元测试已覆盖关系
命中、关系失败和 environment-only block。clean-init GPU smoke 已完成：Independent 与
`explicit_relation` 均训练到 10/10 并写出 checkpoint；期间修复了 `relation_correct`
必须保持为 `numpy.ndarray` 的 trainer 协议问题。

方法上仍有可辨识性风险：当前 price、option、nuisance 数据的关系标签分别固定为
`(COMMIT, SEARCH_ALTERNATIVE)`、`(COMMIT, SELECT_TARGET_OPTION)`、`(COMMIT, COMMIT)`。
因此显式 `R_pair` 与“两侧粗粒度 intent 都正确”高度重合，本质上仍可能只是 conjunctive
bonus。正式论文将把它降级为 matched baseline，不再把它作为最终方法创新。

主方法候选升级为直接优化策略响应量：对 decision-changing intervention，约束两个竞争
intent 的 log-odds preference 按认证方向翻转；对 decision-preserving intervention，约束
intent distribution 保持稳定，并保留 per-side correctness anchor。核心表述是：

> **Flip when it matters; preserve when it does not.**

这需要在 actor/trainer 中取得 canonical intent 的 policy score，而不是只在 rollout 后加入标量 bonus。

## 3.1 新诊断：初始化与联合奖励的稀疏性错配

现有结果给出了一个必须单独控制的变量：

| 起点/方法 | Original | Counterfactual | PRA |
|---|---:|---:|---:|
| clean SFT（未做 certified 单步训练） | 5.99% | 19.29% | 1.12% |
| clean-init Independent CF-GRPO，200 steps | 91.39% | 50.19% | 43.63% |
| clean-init paired joint-bonus，100 steps | 100.00% | 8.99% | 8.99% |
| v4-init Independent smoke，10 steps | 87.64% | 87.83% | 76.97% |
| v4-init residual RelGRPO smoke，10 steps | 86.14% | 88.20% | 75.84% |

这说明 clean SFT 并不是一个已经会做单步 constraint probe 的初始化。对它直接施加
`joint = side_original ∧ side_counterfactual` 的奖励时，配对正信号几乎不可达；模型可以
先稳定学会原始侧 `COMMIT`，却没有足够梯度学会反事实侧恢复动作。b8n4 试验的 100%/8.99%
不是论文方法的反例，而是“联合奖励无法从弱初始化 bootstrap”的反例。

因此后续实验必须把 initialization ablation 明确列出：

- **Certified-init**：从自然 paired SFT/v4 起训，测 relation objective 的增量；
- **Clean-init + side reward**：从普通 clean SFT 起训，先保留 per-side `R_cert`；
- **Clean-init + joint-only**：只作稀疏奖励失败对照，不作为主方法。

如果 paired objective 只有在 certified-init 下有效，论文应诚实地把它描述为“在已有约束能力上
做关系优化”，而不是声称从普通购物 SFT 直接发现关系。

## 3.2 新审计：v4-init 与 GRPO pair 存在训练重叠

当前数据文件的 provenance 还存在一个必须修正的点。对实际文件做集合审计得到：

- v4 corrective SFT 的 `sft_certified_corrective_train.jsonl` 含 3,622 个 certified pair；
- 当前 800-row GRPO 输入含 200 个 unique pair；
- 两者重叠 **168/200 个 pair**；按 task 计重叠 **182/195 个 GRPO task**；
- heldout-v2 与两者均 task-disjoint（与 SFT、GRPO 的 task overlap 都为 0）。

因此 v4-init smoke 的 heldout 分数仍可作为无泄漏泛化评测，但不能支持“GRPO 首次学会
paired relation”的因果解释：绝大多数 GRPO pair 在 SFT 阶段已经出现过。它更接近：

> 在已有 paired SFT 能力、且部分训练 pair 重复出现的情况下，Independent/Relational GRPO
> 如何改变或保持约束行为。

这不影响 v4 作为 feasibility baseline 的价值，但会改变 RQ2 的实验要求。正式 matched
comparison 至少要准备一套严格 provenance split：

1. `certified_sft_pairs`：只用于 paired SFT；
2. `certified_grpo_pairs`：只用于 GRPO，和前者按 task/product 隔离；
3. `heldout_pairs`：只用于最终评测，继续保持 task/product/category 审计。

如果计算资源有限，可以保留当前重叠结果作为 **seen-pair retention** 消融，但主表必须使用
disjoint GRPO pair；否则不能把 v4-init 的结果写成 relation-level RL 的学习证据。

## 3.3 新审计：heldout-v2 已经是开发集，不应继续称为最终 test

`heldout_atomic_pairs_v2.jsonl` 没有 task-level train leakage，但它已经被 v3/v4/v5、
clean SFT、Independent GRPO、Paired GRPO 和 RelGRPO 多轮评测与分支决策使用。按机器学习
实验协议，它现在是 **development probe**，不是未触碰的 final test。继续在它上面调 reward
或挑 checkpoint，再把最高分写成 headline，会产生 test-set overfitting 风险。

候选池足够重新封存测试集：从当前 train range 中排除 SFT、普通 GRPO、Final-200、旧
heldout 和 certified SFT 任务后，仍有约 15,462 个可用 task。对未使用的 1,000 个 task
做不落盘产率测试可得到约 737 个 price pair、325 个 option pair 和 325 个 nuisance pair。

建议协议：

- `heldout-v2-dev`：保留现有 534 pairs，只用于实现调试、早停和 reward 选择；
- `final-atomic-test-v1`：已从严格排除池冻结 300 pairs，包含 price、option、nuisance，
  生成后不再用于任何方法选择；
- final paper 只报告 final-atomic-test-v1，dev 结果放附录或训练曲线。

这套封存 test 不需要 teacher 采集，因为 atomic pair 的 observation 和 verifier 都由程序生成；
只需保存 task/product 排除清单、生成 seed、schema 版本和文件 hash。

当前冻结文件为 `data/counterfactual/final_atomic_test_v1.jsonl`，包含 150 price、75 option、75
nuisance pairs，300 个互异 task/product。正式 GRPO 输入为
`data/grpo_certified_paper_v1_800_pairblocked.parquet`：400 environment rows 加 200 个完整 pair
（100/60/40），同样与 final test 和历史 SFT/GRPO/counterfactual artifacts 按 task/product 互斥。
这些数据已通过协议测试；clean-init Independent 和 `explicit_relation` GPU smoke 已完成。

## 4. 下一轮实验顺序

### Gate A：离线实现验收

不占 GPU。补齐 intent parser 和 relation verifier，并用 price、option、nuisance、lenient search、
malformed action 的固定样例做单元测试。所有测试通过前不启动正式训练。

### Gate B：matched smoke（先隔离初始化，再比较关系目标）

先分别在 certified-init 与 clean-init 两个起点上做短 smoke；同一数据、顺序、seed、rollout 和 token budget，比较：

- Independent：`R_task + R_cert`；
- Conjunctive baseline：加入离散 `explicit_relation` bonus；
- Preference relation：decision-changing pair 做 preference flip，decision-preserving pair 做 preserve；
- Paired-residual：只修复失败侧，保留单侧优势。

通过条件：在每个初始化内，preference-relation PRA 相对 Independent 不低于 2pt；original action accuracy
不下降超过 2pt；nuisance invariance 不下降；所有 pair block 都能匹配，environment-only block
不产生关系奖励。2026-08-20 的 clean smoke 已确认 pair matching 闭环，但 counterfactual block 的
离散 relation success 全为 0；它被归档为稀疏 conjunctive reward 的 bootstrap failure，不再作为主方法调权重。

在 Gate B 之前增加 provenance gate：SFT-certified 与 GRPO-certified 的 task、product、pair
交集必须为 0；heldout 交集也必须为 0。旧的 168/200 重叠输入只能用于 retention ablation。
同时冻结新的 final-atomic-test-v1；在 final test 上禁止 checkpoint、reward 或 prompt 选择。

### Gate C：正式 matched 训练

只有 Gate B 通过，才跑 200 steps、batch 4、至少 3 个 seed。每个 milestone 立即导出 adapter 到 `outputs/`，
并保存配置、数据 hash、base revision、checkpoint hash 和评测命令。最终主表至少包含：

`Final strict / completion / Price-CF / Option-CF / Nuisance invariance / PRA / 95% paired bootstrap CI`。

### Gate D：泛化和长程

在 H2 通过后再做 constraint-type holdout、未见商品/类别、full-horizon 20/30-turn 和外部环境；
不要用这些扩展实验掩盖 matched 主比较尚未成立的问题。

## 5. 写作顺序

1. 先写 Introduction、问题定义和 atomic certification，固定“success ≠ constraint faithfulness”的故事。
2. 写 Paired-C1-hard 动机结果和 SFT v4 feasibility，明确二者不是最终方法胜出。
3. 再写方法实现和 matched protocol；当前结果以“待验证”标注，不提前填主表。
4. Gate C 完成后再写主实验与消融。
5. 最后补 failure analysis：price blindness、summary-presence shortcut、joint reward 对原侧的误伤、
   以及 paired objective 未超过 Independent 的负结果。

## 6. 禁止提前做的事

- 不把 10-step smoke 当作最终方法结果。
- 不把 v4 的 78.65% 当作 GRPO 的效果。
- 不混合 2026-08-19 丢失 adapter 的单 seed结果和后续重训结果。
- 不在同一 heldout pair 集上反复调 prompt 后直接宣称泛化。
- 不在 H2 未通过前扩展 long-horizon、模型尺度或外部 benchmark。
