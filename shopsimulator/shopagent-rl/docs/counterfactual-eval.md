# 原子约束反事实评测（v1）

这条评测检查购物 Agent 在购买提交点是否真正响应结构化硬约束，而不是重复测试普通最终成功率。

v1 只生成两类可自动验证的配对状态：

- `option_swap`：原状态已选择目标规格；反事实状态只将它替换成同商品、同价格的另一可用规格。正确行为由 `COMMIT` 变为 `SELECT_TARGET_OPTION`。
- `price_above_budget`：原状态的目标规格和价格满足预算；反事实状态只把当前价格提高到预算以上。正确行为由 `COMMIT` 变为 `SEARCH_ALTERNATIVE`。

暂不修改自由文本 attribute，也不使用原始数据的 `is_available`。前者没有类型化编辑和一致性保证，后者当前没有被上游网页引擎执行。把它们加入 v1 会破坏“只有一个有效变量改变”的解释。

## 生成

生成过程只读商品和已有评测轨迹，在 CPU 上运行，不启动模型或环境服务：

```bash
python scripts/build_counterfactual_pairs.py
```

默认产物：

- `data/counterfactual/final200_atomic_pairs_v1.jsonl`
- `data/counterfactual/final200_atomic_pairs_v1.summary.json`

轨迹文件优先提供本次评测实际保存的 `instruction_text`、`goal_options` 和 `price_upper`。未完成轨迹不会保存 goal；这时 instruction 和目标规格从官方商品任务恢复，预算按上游 `goal.py` 的十元档位支持确定性构造。每条样本都用 `price_upper_source=observed_eval_goal` 或 `canonical_from_target_price` 明确记录来源，二者应分层报告。商品标题、规格和规格价格始终来自官方真实商品数据。

每一对都带有：原状态、反事实状态、允许动作、动作意图，以及 intervention validity checks。生成器在写盘前再次校验这些检查。

## 第一阶段指标

对同一个 checkpoint 分别推理 pair 的两侧，先报告：

- Original Action Accuracy
- Counterfactual Action Accuracy
- Paired Robust Accuracy（两侧都正确）
- Commit Persistence Error（约束被破坏后仍购买）

只有当“原状态答对、反事实答错”的比例足够明显时，才进入 paired action training。v1 数据不是训练集，也不改变现有 GRPO reward。

## 第一阶段结果（2026-08-14，四 checkpoint 全量）

四个 checkpoint 用同一 212 对、greedy（temperature=0）单轮探测，产物在 `outputs/counterfactual/cf_*_metrics.json`：

| 指标 | Base | SFT v1 | GRPO v1 | GRPO v2b |
|---|---:|---:|---:|---:|
| original_action_accuracy | 0.000 | 0.132 | 0.382 | **0.632** |
| counterfactual_action_accuracy | 0.000 | 0.189 | 0.189 | 0.156 |
| paired_robust_accuracy | 0.000 | 0.014 | 0.057 | 0.090 |
| commit_persistence_error | 0.000 | 0.085 | 0.203 | **0.519** |

按干预类型拆分（关键两行）：

| 指标 | SFT v1 | GRPO v1 | GRPO v2b |
|---|---:|---:|---:|
| price_above_budget cf accuracy | 0.000 | 0.014 | **0.000** |
| price_above_budget commit error | 0.103 | 0.255 | **0.607** |
| option_swap cf accuracy | 0.597 | 0.567 | 0.493 |
| option_swap paired_robust | 0.045 | 0.164 | 0.284 |

三条结论：

1. **价格约束盲视全链路未解决**：四个 checkpoint 的 price cf accuracy 全部 ≈ 0；v2b 在 145 对超预算场景 145/145 照常 click，80.7% 与原状态动作逐字相同。
2. **RL 把盲目提交练得更牢**：commit_persistence_error 沿 SFT → v1 → v2b 单调恶化（0.085 → 0.203 → 0.519）。加宽 rollout（env32/n=8）没有治盲视，反而强化了“点买拿 partial credit”的策略。
3. **盲视是价格特异的**：option_swap 的 paired_robust 从 0.045 升到 0.284，规格约束的敏感性在真实改善。

归因：GRPO 终端奖励是加权**求和**（`shop_env/reward.py`，0.2·r_type + 0.3·r_att + 0.3·r_option + 0.2·r_price），超预算购买（r_price=0）仍保留最高 0.8 的 partial credit——正是 `commit_persistence_error` 单调上升的激励来源。这同时解释了 Final-200 上“v2b 完成率 68% 但 strict 仅 18%”的缺口。

由此启动 **C1-hard 修正实验**（`scripts/run_grpo_c1hard_200_b4n8_env32.sh`，v2b 配方 + `SHOPSIM_REWARD_BUDGET_MODE=hard`：真实购买且超预算 → 终端奖励归 0）。其双评测结果出来后追加到本节。

## C1-hard 双评测结果（2026-08-15，判定：未修复价格盲视）

训练 200 步正常完成（训练内 score/mean 0.995）；adapter 经
`scripts/export_lora_adapter.py` 导出后，用与 v2b 完全相同的协议做 Final-200
（10 turns × 512）与 212 对反事实探测。产物：`outputs/grpo/c1_hard/evaluation/`、
`outputs/counterfactual/cf_c1hard_s200*`。

| 指标 | GRPO v2b | GRPO C1-hard | 变化 |
|---|---:|---:|---|
| Final-200 strict success | 18% (36/200) | 20% (40/200) | +4 题，n=200 噪声内 |
| Final-200 完成率（官方口径=真实购买） | 68% | 53.5% | **−14.5pt** |
| r_loose / r_hard | 0.4355 / 0.2149 | 0.3847 / 0.2275 | loose 降、hard 微升 |
| r_price | 0.55 | 0.425 | 降 |
| cf price_above_budget accuracy | 0.000 | **0.000** | 无改善（144/145 仍点购买） |
| price commit_persistence_error | 0.607 | 0.400 | 降，但由"更不敢买"驱动 |
| cf option_swap accuracy | 0.493 | 0.642 | 继续改善 |
| paired_robust（option_swap） | 0.284 | 0.299 | 持平 |
| probe original_action_accuracy | 0.632 | 0.462 | 点买倾向整体下降 |
| cf same_action_both_sides（price） | 0.807 | 0.759 | 仍≈不看状态 |

三条判定：

1. **价格盲视未修复**：145 对超预算场景 cf accuracy 仍为 0.000，144/145 照常
   `click[buy now]`。hard 归零训练在训练任务上到 0.995 分，但没有泛化出
   "看价格→不买"的区分性行为。
2. **hard 惩罚只降低了购买频率**：完成率 68%→53.5%、probe original accuracy
   0.632→0.462、commit error 0.607→0.400 的改善全部与"整体更少点买"一致，
   而 same_action_both_sides 0.759 表明多数场景动作与原状态逐字相同。
   按购买内部统计，超预算购买占比几乎未变（v2b 26/136 ≈ 19%，C1-hard 22/107 ≈ 21%）。
3. **strict 的 +4 题不构成结论**：与 README 已记录的单任务抖动量级相同。

结论：把超预算购买奖励归零（终端 scalar 惩罚）不足以让策略习得价格约束的
状态依赖响应；问题更可能在观测侧（模型无法从页面文本中读取/绑定预算数字），
而非奖励侧。下一步应先诊断 observation 中价格与预算的可辨识性
（如 token 级注意力/探针分析），再决定是否构造显式价格-预算对照的中间奖励。

## Paired-C1-hard 双评测结果（2026-08-17，判定：strict 大涨，价格盲视仍在）

训练：SFT v2 paired adapter 初始化 + `SHOPSIM_REWARD_BUDGET_MODE=hard`，
b4/n4/lr1e-5，200 步正常完成（`paired_c1hard_200_direct`）。评测协议与
v2b / C1-hard 完全一致：Final-200（10 turns × 512，wave 16）+ 212 对原子
反事实探测。产物：`outputs/grpo/paired_c1_hard/evaluation/`、
`outputs/counterfactual/cf_pairedc1hard_s200*`。

| 指标 | SFT v2 Paired | GRPO C1-hard | GRPO Paired-C1-hard |
|---|---:|---:|---:|
| Final-200 strict success | 25% (50/200) | 20% (40/200) | **40% (80/200)** |
| Final-200 完成率（官方口径=真实购买） | 52.0% | 53.5% | **90.0%** |
| r_loose / r_hard | 0.392 / 0.278 | 0.385 / 0.228 | **0.644 / 0.425** |
| r_price | — | 0.425 | **0.765** |
| 选对商品率 | 27.5% | 28.0% | **45.0%** |
| cf price_above_budget accuracy | 0.000 | 0.000 | **0.000** |
| price commit_persistence_error | — | 0.400 | 0.931 |
| paired_robust（option_swap） | 73.1% | 29.9% | **73.1%**（49/67，与 SFT v2 持平） |
| probe original_action_accuracy | — | 0.462 | 0.938 |
| cf same_action_both_sides（price） | — | 0.759 | 0.986 |

判定：

1. **strict 40% 是真实跃升**：80 vs 50 题（SFT v2）的差距远超单 run 抖动量级；
   完成率、r_hard、选对商品率同步大幅改善。paired 初始化（保留规格因果的 SFT
   起点）+ hard 预算惩罚的组合有效。
2. **option_swap 因果性未被 GRPO 破坏**：paired_robust 73.1% 与 SFT v2 完全持平
   （C1-hard 从 base 起训时只有 29.9%）。
3. **价格盲视依旧**：145 对超预算场景 cf accuracy 0.000，143/145 两边同样
   `click`，same_action_both_sides 0.986。与 C1-hard 的关键差别在 original 侧
   准确率 0.938（预算内选品很准）：策略把「属性匹配」先验外推到价格维度——
   价格数字读得到（[`price-blindness-next.md`](price-blindness-next.md) 的
   Thought 层证据：模型复述价格与预算但不执行比较），比较没有进入决策。
   commit_persistence_error 0.931 为全系列最高：strict 提升全部来自"买对"，
   没有来自"超预算不买"。

与 [`price-blindness-next.md`](price-blindness-next.md) 的衔接不变：E1
（合成约束违反数据注入）仍是预注册选定的修复路线。

## Certified SFT v3 heldout-v2（2026-08-17，判定：summary shortcut）

v3 adapter 在与训练 task/product 隔离的 534 对 heldout-v2 上评测。自然输入中 option
方向继续改善，但价格方向完全失败：

| 指标 | option swap (n=150) | price above budget (n=384) |
|---|---:|---:|
| original action accuracy | 0.940 | 0.951 |
| counterfactual action accuracy | **0.820** | **0.000** |
| paired robust | 0.813 | 0.000 |
| commit persistence error | 0.080 | **0.943** |

训练数据审计发现，v3 只给价格负例追加 `任务约束摘要: 预算上限=...`，自然评测没有
该行。新增 `--variant summary` 仅作诊断：恢复该行后，price cf accuracy 变为 1.000，
但 price original accuracy 只有 0.326。该反转证明模型依赖 summary presence，而非
执行价格比较。因此 structured/summary 数字只能作为 shortcut 证据，不能替代 natural headline。

修正版数据同时提供预算内 `COMMIT` 与超预算 `SEARCH_ALTERNATIVE`，两侧都保持自然输入，
并加入 summary nuisance control。自动链只在 natural heldout-v2 price cf ≥0.30 后继续
Final-200；Final strict ≥0.16 后也只记录通过，GRPO 仍须人工启动。

产物：

- `outputs/counterfactual/cf_sft_v3_certified_heldout_v2_metrics.json`
- `outputs/counterfactual/cf_sft_v3_certified_heldout_v2_summary_diag_metrics.json`
