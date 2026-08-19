# 轨迹长度与非固定轮数研究设计

> 状态：当前研究设计说明（2026-08-19）

## 1. 先区分四个“长度”概念

当前项目里有四种不同的长度限制，不能混写成“数据只支持 10 轮”：

| 概念 | 当前设置/事实 | 含义 |
|---|---:|---|
| Teacher 环境上限 | 30 steps | terra 采集时单条轨迹最多交互 30 步 |
| terra 母集质量门 | `2 ≤ n_steps ≤ 30` | 建立完整 strict-success 母集 |
| 当前论文 SFT 质量门 | `2 ≤ n_steps ≤ 10` | 从母集中筛选与当前固定协议一致的训练子集 |
| Final-200 协议 | 历史结果多为 `10 turns × 512 tokens` | 为可比性固定的评测协议，不是训练数据硬限制 |
| GRPO response budget | `response_length=8192` | token 区域上限；不是固定 turn 数，实际由环境终止和 token 预算共同决定 |

因此，最初的 GPT-5.6-terra 数据先得到 3,793 条 strict 成功轨迹；当前论文再从这批母集中筛选
`n_steps ≤ 10` 的子集，以和现有 Final-200 协议对齐。

## 2. 当前 terra 基础数据的真实范围

`data/sft_train.jsonl` 共 **3,793** 条 strict-success 轨迹，经过 `teacher.validate` 的质量门：

- `ok == True`；
- 四维 strict success；
- `2 ≤ n_steps ≤ 30`；
- `legal_ratio ≥ 0.8`。

当前文件中的步数分布为：

| 步数 | 条数 |
|---:|---:|
| 4 | 2,454 |
| 5 | 775 |
| 6–10 | 395 |
| **>10** | **169** |

因此，3,793 条中有 **169 条超过 10 步**。这些轨迹仍属于合法的基础成功数据，但按当前论文的预注册协议，
它们不进入当前主实验的基础 SFT 子集；保留在母集和后续 long-horizon 扩展中。

当前论文使用的 `n_steps ≤ 10` 子集单独保存为 `data/sft_train_horizon10.jsonl`，仍覆盖搜索、规格选择、预算比较和最终购买等基本购物行为；它的作用是
在固定可比的交互 horizon 内隔离约束忠实性，而不是声称覆盖全部长程购物能力。

## 3. 为什么当前 v4 mix 只有 10,057 条

`data/sft_train_certified_corrective_mix.jsonl` 是一次特定的 **10-turn aligned feasibility experiment**。
构建脚本 `scripts/build_sft_certified_mix.py` 默认使用 `--max-turns 10`，因此从 3,793 条基础轨迹中保留
3,624 条，排除 169 条 `n_steps > 10`，再加入 6,433 条 certified/diagnostic 单轮样本，得到 10,057 条。

这一步是当前论文的明确数据选择：让基础购物轨迹、paired intervention 数据和 Final-200 评测共享同一
`10-turn` horizon，先隔离研究变量，再评估配对约束学习本身。

所以文档中应严格区分：

- **基础 SFT 母集**：3,793 条，保留 2–30 步；
- **当前论文基础子集**：3,624 条，筛选 `n_steps ≤ 10`；
- **v4 aligned mix**：10,057 条，由 3,624 条基础子集与 6,433 条 certified/diagnostic 样本组成；
- **long-horizon 扩展**：未来可将 169 条 `n_steps > 10` 加回，但不属于当前论文主实验。

## 4. 后续扩展：full-horizon paired mix

当前论文的正式 Paired GRPO 使用现有 `10,057` 条 aligned mix。若后续单独研究 long-horizon，
再从完整 `sft_train.jsonl` 重建扩展版本：

```bash
python scripts/build_sft_certified_mix.py \
  --baseline data/sft_train.jsonl \
  --certified data/sft_certified_corrective_train.jsonl \
  --out data/sft_train_certified_corrective_mix_full_horizon.jsonl \
  --max-turns 30
```

按当前文件规模，若 certified 数据保持不变，扩展版 mix 预计为 **10,226 条**（3,793 + 6,433）。
实际数量以脚本输出为准，不应手工写死。

这套 full-horizon mix 只用于后续扩展，验证：

1. paired intervention learning 是否能保留长轨迹中的正常搜索与恢复行为；
2. price/option sensitivity 是否只在短轨迹 probe 上成立；
3. relation reward 是否会诱导模型过早提交或过度保守；
4. 轨迹长度与 constraint-faithfulness 之间是否存在 trade-off。

## 5. 非固定轮数的训练与评测原则

### 5.1 训练

训练不应把“10 轮”作为方法定义。环境交互应允许在合法终止前继续进行，实际上限由：

- ShopSim 环境 `max_steps`；
- GRPO AgentLoop 的 `max_assistant_turns`；
- `response_length` token budget；
- 环境是否已经完成购买；

共同决定。

`SHOPSIM_TURN_MAX_TOKENS=160` 是单轮 assistant 输出上限，`response_length=8192` 是整条 response 区域上限，
二者都不是“最多 10 轮”的语义承诺。

### 5.2 Final-200 主评测

历史 Final-200 结果仍需保留固定 `10 turns × 512` 协议，以保证已有 Base/SFT/GRPO 数字可比；但论文中要明确标注：

> 这是 benchmark protocol，不是模型能力或训练数据的 horizon definition。

后续正式实验建议新增两种评测：

| 评测 | 设置 | 目的 |
|---|---|---|
| Fixed-protocol | 10 turns × 512 | 与历史结果直接比较 |
| Full-horizon | 20 或 30 turns，token budget 固定 | 测试长程恢复和约束忠实性 |

两者同时报告，不能用 full-horizon 结果替换历史 headline，也不能只报固定协议来声称 long-horizon 泛化。

### 5.3 Paired intervention 评测

反事实 pair 不应只在第一步生成后立即判定。除了单轮 probe，还应在多轮交互中记录：

- 首次发现约束冲突的步数；
- 从错误商品恢复到替代商品的步数；
- 购买前是否重新检查决定性约束；
- pair 两侧最终动作意图和终局结果；
- 是否因 response/token cap 被截断。

主指标仍是 PRA、Relevant Sensitivity 和 Irrelevant Invariance；轨迹长度指标用于解释机制，不替代主指标。

## 6. Long-horizon 实验矩阵

正式实验至少包含下面四个条件：

| 条件 | 数据 | horizon | 目的 |
|---|---|---|---|
| Baseline SFT | 完整 3,793 条 terra strict | fixed/full | 基础能力与长度分布 |
| v4 aligned SFT | 历史 10,057 mix | fixed | 复现已有可行性结果 |
| Full-horizon CF-GRPO w/o Pair | 预计 10,226 mix | fixed/full | 独立 CF reward 的 matched baseline |
| Full-horizon Paired GRPO | 同上 | fixed/full | 验证 relation-level objective |

核心比较不是谁在 10 轮内买得更快，而是：

\[
\text{Success} + \text{PRA}
\quad\text{在更长交互预算下是否仍然同时成立。}
\]

## 7. 需要新增的长度分层指标

除最终 success、CF Accuracy 和 PRA 外，按以下区间分层：

- `n_steps = 1–5`；
- `6–10`；
- `11–20`；
- `21–30`；
- reached cap / token-truncated。

每层至少报告：

- strict success；
- relevant paired robust accuracy；
- nuisance invariance；
- premature commit rate；
- 平均实际步数；
- 截断率。

这样才能区分三种表面相同的结果：

1. 模型真正学会了更长程的约束复核；
2. 模型只是更早购买，避免进入困难后半程；
3. 模型变得不敢购买，靠撞上限降低错误提交。

## 8. 结论与执行顺序

当前结论是：**不用重新采一套 terra；直接从 3,793 条母集中筛选 `n_steps ≤ 10`，即可得到当前论文所需的基础数据。**

执行顺序：

1. 保留现有 `sft_train.jsonl` 作为 3,793 条母集；
2. 当前论文直接筛选 `n_steps ≤ 10`，使用 3,624 条基础轨迹；
3. 使用现有 10,057 条 v4 aligned mix 进行当前 paired policy 实验；
4. 从该起点分别运行 `CF-GRPO w/o Pair` 和 `Paired GRPO`；
5. 未来若研究 long-horizon，再用 `--max-turns 30` 构建 10,226 条扩展 mix。

这条设计保证当前论文先在统一的 10-turn 数据与评测协议下验证 constraint-faithful policy behavior，
再把可变长度、多轮 horizon 作为后续独立扩展，而不是把两个问题混在同一篇论文里。
