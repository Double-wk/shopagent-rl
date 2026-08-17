# shopagent-rl 面试话术与技术要点

> 最后同步：2026-08-17。三条主线：**①讲清楚 ShopSimulator 环境怎么模拟**，
> **②讲清楚 SFT+GRPO 后训练**，**③讲清楚为什么 Final 成功率不等于约束敏感性，以及如何用原子反事实认证发现 shortcut**。
> 最后有「数字状态」表——**报数前先看那张表，只报实测过的**。

---

## 0. 30 秒电梯陈述

> "我用 **SFT + GRPO** 把一个 **1.7B 小模型**（Qwen3-1.7B-Base）在**中文购物多步决策**任务上做后训练。
> 任务要求模型在一个**模拟电商环境**里自主完成'搜索→选品→选规格→比价→购买'的多轮交互。
> 我搭了**完整流水线**：环境集成（40-env 并发池）、teacher 轨迹采集、LoRA SFT、veRL GRPO、
> 与训练零重叠的 Final-200 评测，以及原子反事实认证。最强模型 strict 从 0% 提到 40%，
> 但价格反事实仍是 0%；这暴露了 outcome 指标掩盖的约束 shortcut，并推动我设计自然格式的
> paired corrective training，而不是只继续堆终端奖励。"

一句话关键词：**agentic RL / GRPO / 多步决策 / counterfactual certification / shortcut audit / 小模型后训练**。

---

## 1. Situation（为什么这个任务难、值得做）

- 购物 Agent 要**多步推理 + 环境交互**（不是单轮问答），每一步动作影响后续观测，存在 long-horizon credit assignment。
- **小模型直接做效果差**：在当前 `fast10` 评测中，Base strict success 为 **0%（0/200）**，主要问题是不会持续推进搜索、详情浏览和规格选择。
- ShopSimulator 是阿里开源的中文购物 Agent 评测环境（WebShop 衍生），官方用 Qwen3-8B 做了 SFT+RL（arXiv:2601.18225）——我拿更小的 **1.7B** 复现并改进。

---

## 2. 环境是怎么模拟的（核心技术点，必背）

**一句话：纯文本电商模拟器——进程内模拟一个"假购物网站"，LLM 通过文本（不是看图、不是真浏览器）一步步浏览、决策、购买，全部在内存里跑，无真实网络请求。**

### 四层架构（数据流自上而下）

```
模型 (vLLM)
   │  HTTP: POST /api/shop_agent
   ▼
① pack_api.py        — Flask，管「40 个 env 的池」（分配/回收/lease 15min 超时回收）
   │
   ▼
② shop_agent.py      — 解析 LLM 输出（"Action: search[xx]"），调 env.step()
   │
   ▼
③ WebAgentTextEnv    — gym.Env：reset(task) / step(action) / observation
   │
   ▼
④ SimServer+SimBrowser — 真正的"假网站"：商品库 + 搜索引擎 + 页面状态机 + reward
```

### "假网站"的页面状态机（跟淘宝/亚马逊一样）

| 页面 | 内容 | 触发动作 |
|---|---|---|
| `index` | 初始落地页 | reset 后起点 |
| `search_results` | 搜索结果列表（带翻页）| `search[关键词]`、`click[Next >]`/`click[< Prev]` |
| `item_page` | 商品详情：标题/价格/规格选项 | `click[item - B0xxx]` |
| `item_sub_page` | 描述/参数/评论/属性子页 | `click[Description/Features/Reviews/Attributes]` |
| **`done`** | 结算（终态，算 reward）| `click[buy now]` |

- 页面由 `map_action_to_html` 现场拼 HTML，再 `convert_html_to_text` **剥成纯文本**喂 LLM。
- `get_available_actions()` 从 HTML 抠出当前可点按钮 → 作为「可选操作列表」一起给 LLM（每步都知道合法动作）。

### 一个 episode 的循环

```
reset(task_id) → 返回任务指令("买M码黑色T恤≤50元")
  → LLM 输出 "Thought: ... Action: search[纯棉 T恤]"
  → env.step → 返回搜索结果页(文本)+可点按钮
  → LLM: click[item - 某某T恤]   进商品页
  → LLM: click[黑色] / click[M码]  选规格(存 session.options)
  → LLM: click[buy now]          终态，算 reward
  （或步数超上限还没买 → 封顶失败：上游 MAX_HISTORY=42，shopagent-rl wrapper 封 30）
```

### Reward：4 维 + 3 种聚合（纯规则，无 LLM Judge）

`goal.py:get_reward` 对比「买的」vs「标准答案」（4 维）：

| 维度 | 评什么 | 算法 | 取值 |
|---|---|---|---|
| r_type | 品类对吗 | query 直比 / category ≥2 级匹配 / **spacy 抽中文名词重叠**（title_score>0.2）| **1.0 或 0.5**（沾边给半分，**非 0/1**）|
| r_att | 属性对吗 | thefuzz `token_set_ratio` **>85** 算命中 | 命中数/属性数 |
| r_option | 规格(颜色/尺码)对吗 | thefuzz >85（颜色先 `normalize_color`）| 命中数/规格数 |
| r_price | 价格≤预算吗 | `price <= price_upper` | 0 / 1 |

> ⚠️ **价格随机化**：每次 reset，商品价格用 `random.uniform` 现场生成、goal 的 `price_upper` 也随机采样 → **r_price 跨 run 波动**，即使 teacher temp=0，同一任务两次 strict 可能不同。讲复现/选型时要主动说，多 run 看趋势。

env 内部 `total_reward = (attr命中+option命中+r_price)/(属性数+规格数+1) × r_type`（`goal.py:224-228`）。

三种**评测聚合**（`get_score.py`，从存盘的 `reward_detail` 重算）：
- **r_loose** = 直接用 env 的 `total_reward`（宽容，给 RL 密集信号）
- **r_hard** = r_type×r_att×r_option×r_price（乘法瓶颈，一项 0 全 0）= 官方 **Rstrict**
- **r_success** = 四维**全 ==1** 才算 1（最严格，0/1）= 官方 **Rsucc** = **「严格成功率」**
  （等价于 `total_reward == 1.0`；shopagent-rl `reward.py:strict_success` 就用这个，与论文 Rsucc 同口径）

> 亮点话术："reward 分层：loose 给训练密集信号、strict 给评测硬指标；只用规则（spacy+thefuzz）不用 LLM Judge——可复现、零成本、对齐官方 benchmark。"

### 非法动作与 legal_ratio（shopagent-rl 独有的数据卫生）

- **上游对坏动作是静默 no-op**：parse 失败 / 点了当前页面不存在的东西 → `reward=0, done=False`，**页面不变重渲染**、reward 照算。RL 里会埋雷（agent 能乱点 30 步还偶然买对）。
- **shopagent-rl wrapper 每步打 `info["legal"]`**（parse 成功、且 click 匹配到当前可点元素才算 legal）——但只记**每步布尔**。
- **`legal_ratio = 1 − illegal/n` 在 `teacher/validate.py` 聚合**（不在 wrapper 里），作 SFT 卫生门：≥0.8 才入训。被剔的轨迹 reward/strict 不受影响（legal_ratio 只筛训练数据，不改 reward/eval 定义，详见 `DATA.md`）。

### 深挖（高频追问展开）

**Q：价格为什么随机化？影响什么？**
- 机制：每次 reset，`generate_product_prices()`（`engine.py:174`）用 `random.uniform` 重生成商品价；goal 的 `price_upper` 也随机采样（`goal.py:68`）。
- 为什么：价格固定 → `task_id→答案`可背、r_price 退化成固定标签，agent 不用真比价 → 比价能力测不出、benchmark 可刷。随机化**逼 agent 每步真读价、真比预算**，把"比价"做成可测子能力 + 增加 task 多样性。
- 影响（代价）：① 单任务 strict 非确定（前三维全对也可能这次 r_price=0）→ **报 Final-200 聚合、不报单任务**；② 探针/选型数字小波动（如 grok 57↔53%）→ **多 run 取趋势**；③ SFT 标签用采集时那次 roll（`build_sft_data.sh` 不 `--replay`）→ strict 锁死采集那一刻、不会事后翻盘。

**Q：env 池"无锁分配"是什么意思？**
- 机制：40 个 env 的空闲编号放 Python `set free_env_index`；分配 `pop()`、回收 `add()`，**不加锁**。
- 为何安全：CPython **GIL** 保证同一时刻只一个线程跑字节码，`set.pop/add` 是原子操作 → 并发请求不会抢到同一 env。
- 取舍：省锁开销/无死锁；**代价是隐式绑 CPython+GIL**（换 Jython / py3.13 nogil 就有竞态）；池空时 `pop` 抛 KeyError → **靠重试（`MAX_RETRIES=5`）不靠阻塞等锁**。

---

## 3. Action（我做了什么）—— STAR 的 A

| 模块 | 我做的 | 技术 |
|---|---|---|
| **环境集成** | 把 ShopSimulator 包成可批量调用的服务，**40-env 并发池** + lease 超时回收 | Flask + gym，进程内模拟 |
| **Teacher 采集** | 多模型同条件对比（gpt-5.6-terra/deepseek-v4-flash/grok 等），**选定 gpt-5.6-terra**，采严格通过轨迹 | OpenAI-compat API（mcgrox），断点续传 |
| **LoRA SFT** | 对 assistant token 算 loss，蒸馏 teacher 行为 | peft + bf16，3.8K（3793）轨迹 |
| **veRL GRPO** | 去 KL、分段 reward、组内零方差样本过滤、从 SFT adapter 续训 | verl，ref 用 disable_adapter 免占显存 |
| **Final-200 评测** | 与训练零重叠的 held-out 集，vLLM 批量 rollout，调官方 get_score 出规范指标 | 波束批处理，seed=42 disjoint 检查 |
| **原子反事实认证** | 在购买提交点只改变价格或目标规格，要求动作从 COMMIT 定向切换；同时加入 nuisance control | paired robust / commit persistence / shortcut rate |
| **Corrective training** | 发现 v3 的 summary-presence shortcut 后，改为自然格式价格双侧训练并设置 heldout/Final 双门 | paired SFT + gated Certified GRPO |

### GRPO 配置（唯一事实源是 `configs/grpo.yaml` + `docs/grpo.md`）

```
TRAIN_BATCH   = 4
ROLLOUT_N (G) = 4（v2b 消融使用 8）
主结果 steps  = 200
当前最好      = Paired-C1-hard strict 40%（80/200），完成率 90%
```

**别把历史 step20 当当前结果**：ROCm/vLLM sleep、HIP IPC 权重同步和冻结 LM head
无效梯度问题修复后，env16、v2b、C1-hard、Paired-C1-hard 均已完成 200 steps。

**本地真瓶颈（两层，都要说）**：
1. **env 吞吐**：每步 rollout 数 = batch × G，受 env 池与多轮网页交互延迟限制。
2. **单卡显存共居**：vLLM rollout 与 FSDP actor 共用 48GB；最终通过 ROCm sleep release、
   POSIX shared-memory 权重同步和跳过冻结 LM head 梯度修复，而不是改变训练语义。
3. **行为信用分配**：hard-zero 能减少整体购买，却没有教会条件化价格比较；最终奖励改善
   不能代替 paired counterfactual evaluation。


---

## 4. 卖点（按面试官权重排）

1. **端到端 Agentic RL**（最稀缺）——环境/数据/SFT/GRPO/评测全做过的人少。
2. **GRPO 原理透 + 踩坑真**——组内相对优势（无 value model 省一半算力）、去 KL、`batch×G×steps` 怎么定；最值钱的是**踩过真实的单卡 OOM**：hybrid_engine 下 vLLM rollout 池与 FSDP actor 峰值在同一张 48G 卡上叠加，崩溃时 PyTorch 才占 11G 卡却 0 字节可分，`micro_batch` 已是 1 仍救不回——这比"瓶颈是 env 池不是显存"的纸面结论更可信（后者只描述了吞吐瓶颈那一层）。
3. **工程严谨**——40-env 池 lease/超时回收、train/eval 互斥采样（seed=42 disjoint 检查）、数据完整性 guard、vLLM 批量评测。
4. **评测方法论**——零重叠 held-out、官方 get_score 口径、分层 reward。
5. **批判性思维**——主动推翻 v3 的“价格已修复”结论：自然输入 0%，恢复训练摘要 100%，
   由此定位 summary-presence shortcut，并重做 paired 数据和门控协议。

---

## 5. 高频追问 + 回答要点

| 问题 | 回答要点 |
|---|---|
| 为什么 GRPO 不用 PPO？ | 无 value 网络（省一半显存/算力）；组内相对优势天然适配稀疏奖励；大模型 RL 事实标准（DeepSeek-R1）。|
| reward 怎么设计？ | 4 维规则 + 3 聚合（loose 训练 / strict 评测）；不用 LLM Judge 因为可复现 + 对齐官方。|
| 怎么防 eval 泄露？ | Final-200 从 eval 区采，与 SFT/GRPO 池 seed=42 互斥，代码层 disjoint 检查。|
| G/batch/steps 怎么定？ | 主结果 batch 4、G 4、200 steps；v2b 用 G8 做信号消融。参数先由单卡共居和 env 吞吐约束，再用零方差率、pg loss、显存稳定性验收。|
| 最大难点？ | 一是 ROCm/vLLM/FSDP 共居的物理显存与权重同步问题；二是 outcome reward 会强化“买对大部分属性就提交”，却不保证模型响应预算反转。|
| 反事实创新是什么？ | 不是简单加负样本，而是原状态/单变量干预/无关扰动三类对照，按动作意图计算 paired robust、commit persistence 和 shortcut rate，并用自然 heldout 阻断格式捷径。|
| v3 为什么作废？ | 价格负例单独带预算摘要且缺少预算内双侧对照；自然测试 0%，恢复摘要 100%，说明模型学了提示存在性。|
| 多轮交互怎么接 veRL？ | rollout 阶段 env 多步交互、终态 reward 广播回 assistant token 算优势（懂 multi-turn RL）。|
| SFT teacher 怎么选？ | 5 模型同条件对比（质量/成本/稳定性），见 `run/teacher_compare/`。|
| 数据怎么划分？ | 官方 tag 字段（eval 0-1458 / train 1459-23420），三集 seed=42 不相交。|
| 为什么价格要随机化？影响啥？ | 防 agent 背 `task_id→答案`（固定价=比价题变背答案题），逼真比价；代价：单任务 strict 非确定→报 Final-200 聚合、探针多 run 取趋势；SFT 不 replay 故标签锁死采集那一刻。详见 §2 深挖。|
| env 池怎么并发？为何无锁？ | 40-env 用 `set` pop/add 分配，靠 CPython GIL 保原子、省锁；代价是隐式绑 CPython、池空靠重试（`MAX_RETRIES=5`）不阻塞。详见 §2 深挖。|

---

## 6. ⚠️ 数字状态（报数前必看）

> 最后更新：2026-08-17。下表全部为 Final-200（n=200）实测，10 步、每轮 512 token 协议。

| 指标 | 完成率 | strict success (Rsucc) | r_hard | r_loose | 选对商品率 |
|---|---:|---:|---:|---:|---:|
| Base | 0% | **0%（0/200）** | 0 | 0 | 0% |
| SFT（`sft_new3793`） | 39.5% | **17%（34/200）** | 0.201 | 0.306 | 25.5% |
| SFT v2 Paired | 52.0% | **25%（50/200）** | 0.278 | 0.392 | 27.5% |
| GRPO v2b（step 200） | 68.0% | **18%（36/200）** | 0.215 | 0.435 | 27.5% |
| GRPO C1-hard（step 200） | 53.5% | **20%（40/200）** | 0.228 | 0.385 | 28.0% |
| **GRPO Paired-C1-hard（step 200）** | **90.0%** | **40%（80/200）** | **0.425** | **0.644** | **45.0%** |
| 官方 Qwen3-8B（参照）| — | SFT 32.5% / SFT+RL 38.9%（Table 3）| — | — | — |

**怎么说 GRPO（关键，别说过头）**：

- 能说的：paired 初始化 + hard 预算惩罚把 strict 从 SFT v2 的 25% 提到 40%，且完成率、
  r_hard、选对商品率同步提升；规格反事实 paired robust 维持 73.1%。
- 不能说的：不能据此声称模型学会价格约束。Paired-C1-hard 的 price cf 仍为 0%，
  93.1% 的原状态正确样本在超预算干预后继续提交。
- v3 Certified 也不能报成功：自然 price cf 0%，summary 诊断 100% 只证明输入格式捷径。
- 当前诚实表述：**整体购物策略和规格约束显著改善；价格条件化决策尚未解决，v4 修正版正在验证。**

**铁律**：把"设计了什么 / 实测到什么 / 还在做什么"分清楚。可以报 Base 0% →
Paired-C1-hard 40%，但必须同句补充 option paired robust 73.1%、price cf 0%；v4 没出自然 heldout
结果前，不宣称 Certified 方法解决了价格推理。

> ⚠️ 价格随机化（见 §2 深挖）会让单任务 strict 跨 run 抖动，所以只报 Final-200 聚合、不报单任务，
> 也不要拿 1-2 个 pt 的差异讲故事。

---

## 7. 相关文件（被追问细节时可指）

- 环境 reward：`../ShopSimulator/shop_env/web_agent_site/engine/goal.py`
- 评分聚合：`../ShopSimulator/get_score.py`
- 环境池：`../ShopSimulator/shop_env/shop_env/pack_api.py`
- 动作解析：`../ShopSimulator/shop_env/shop_env/shop_agent.py`
- 文本环境/页面状态机：`../ShopSimulator/shop_env/web_agent_site/envs/web_agent_text_env.py`
- GRPO 配方：`docs/grpo.md`
- 数据划分：`scripts/sample_budgets.py`
- 评测：`experiment/eval/run_final200.py`
- SFT 数据构建：`scripts/build_sft_data.sh`
