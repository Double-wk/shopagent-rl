# Constraint-Causal Consistency：主会级实验计划

## 研究问题

Outcome RL 的 strict success 是否掩盖了未依赖任务关键约束的 shortcut
success？若是，原子环境干预下的行为认证能否筛选并训练出对关键约束敏感、对
无关扰动稳定的购物策略？

## 评测分层（不得混报）

| Split | 作用 | 能支持的结论 |
|---|---|---|
| Natural atomic | 原始自然指令的精确单处改写；仅接受程序验证的改写 | 自然语言约束泛化（headline） |
| Natural atomic (teacher-rewrite) | 教师把规格 A→B 改写成自然需求，程序门验证后才接受 | 自然语言约束泛化（headline 扩量） |
| Structured control | 两边追加同格式 verifier 摘要，只改变一个字段 | 受控诊断/训练消融，不能替代 headline |
| Nuisance control | 只改变展示无关文本，动作意图应不变 | 排除“任何页面变化都不买”的投机 |
| Held-out v2 | 新 seed、训练任务与产品/类别均隔离 | 最终泛化 |

### 自然改写管线（v3-natural）

- 代码：`experiment/constraint_causal_natural.py` + `scripts/build_natural_pairs.py`；
  产物 schema `shopsim-constraint-causal-pairs-v3-natural`，`intervention_type=option_goal_swap_natural`。
- 教师只负责表面改写，**永不自证**。每条改写必须通过纯程序验证门：
  1. 原指令中不属于旧规格 A 的数字/规格 token 必须原样保留（护预算、数量）；
  2. A 独有 token 不得残留，B 独有 token 至少出现一个（约束真的翻转）；
  3. 去掉 token 后的字符二元组 Jaccard ≥ 0.5（防漂移到别的商品/人设）；
  4. 长度比 0.5–2.0。
- 被拒改写连同原因落盘（`--rejected-out`），只作审计，绝不进评测/训练集。
- 训练/评测使用前须人工抽查 accept 样本（目标 ≥30 对抽 5–10 条人工确认）。

## 必报指标

- Final-200：strict success、完成率、四维 reward、步数、非法动作率；每个条件至少
  3 个价格随机化 seed。
- 原子对：原状态准确率、反事实准确率、paired robust accuracy、commit persistence、
  causal success certification rate、shortcut success rate。
- 不变性：Nuisance Invariance Score（无关扰动下动作**意图**保持正确的比例）。
- 统计：paired bootstrap 置信区间；同一任务/seed 上比较，避免独立样本假设。

## 方法对照

1. SFT：原始 3793 条专家成功轨迹。
2. Hard-negative SFT：同量约束违反样本，但不使用成对目标。
3. Paired constraint SFT：满足→COMMIT、违反→定向恢复动作。
4. Certified SFT：按反事实认证分数筛选/加权成功轨迹。
5. Certified GRPO：在相同 SFT 起点上，将环境 outcome 与认证信号结合。

所有方法必须匹配：基座、LoRA rank、训练 token 数、训练任务池、随机 seed 和推理
协议。不能以不同 prompt、不同 turn cap 或不同价格随机 seed 比较。

## 泛化与消融

- train/test 产品、task id、类别、约束组合均不重叠；
- 约束类型 holdout：训练 price/option，测试容量、数量、兼容性等；
- distractor 数、价格距离、规格文本相似度分层；
- 消融目标侧干预、商品侧干预、nuisance regularisation 和认证权重；
- 至少一个额外模型尺度或第二个购物环境，检验不是单一 Qwen-1.7B/ShopSimulator
  的专属现象。

## 决策门

1. 修正协议后的基线仍显示 price/option shortcut，才进入训练。
2. Paired/Certified SFT 必须在 natural held-out 集提升 paired robust accuracy，且
   Final-200 strict 不低于最强基线的置信区间下界，才进入 GRPO。
3. Certified GRPO 必须同时优于 matched hard-negative 与普通 GRPO，且 nuisance
   invariance 不下降，才作为方法主结果。
