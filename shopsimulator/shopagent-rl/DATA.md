# shopagent-rl 数据划分与现状（单一事实源）

> 最后更新：2026-08-17。本文是 shopagent-rl **数据集合的唯一事实源**——
> SFT / GRPO / Ablation / EVAL 及 constraint-causal 数据的来源、规模、生成方式、互斥关系与当前状态全在这里。
> README 里的旧「数据采集策略/数据划分」数字已过期，以本文为准。

---

## 一句话总览

四份数据集，全部 **seed 42**、**级联互斥**（两两交集 = 0），从官方 23,421 个任务里切出来：

| 集合 | N | task_id 范围 | 来源文件 | 要不要 teacher 采集？ | 当前状态 |
|---|---:|---|---|---|---|
| **EVAL** | 200 | `[16..1458]` | `data/final200.json` | ❌ 不需要 | ✅ 就绪 |
| **SFT** | 5000 目标 → **3793 strict** | `[1462..23418]` | `sft_collect_targets_5000.json` + raw | ✅ 已采完 | ✅ 训练集已生成 |
| **GRPO** | 1000 | `[1476..23417]` | `grpo_prompts_1000.json` → `grpo_train.parquet` | ❌ 不需要 | ✅ 就绪 |
| **ABLATION** | 500 | `[1471..23392]` | `ablation_tasks_500.json` | ❌ 不需要 | ✅ 就绪 |

train 区（`[1459,23421)`，共 21,962 个任务）已用 **SFT 5000 + GRPO 1000 + Ablation 500 = 6500**，剩 **15,462** 空闲。

---

## 一、官方任务划分（不可改，来自 `items_eval_train.json` 的 `tag` 字段）

| 索引范围 | tag | 数量 | 用途 |
|---|---|---:|---|
| `0 – 1458` | `eval` | 1,459 | 官方评测集，EVAL 从此采样 |
| `1459 – 23420` | `train` | 21,962 | 官方训练集，SFT / GRPO / Ablation 从此采样 |

EVAL 在 `eval` 区，SFT/GRPO/Ablation 在 `train` 区——**两个区天然不重叠**，加上 train 内部级联互斥，四份数据彻底隔离。

---

## 二、级联互斥规则（核心设计）

"一步一步"切：后一个集合永远排除前面所有已定集合，保证 train 区内三份互斥。

```
1. SFT       : 固定（5000 采集目标，seed 42 全 train 随机，排除 eval）
2. GRPO      : train \ (SFT ∪ eval)              → 随机采 1000
3. ABLATION  : train \ (SFT ∪ GRPO ∪ eval)       → 随机采 500
```

- 全部用 `seed = 42`，**可复现**：同样的输入文件 + 同样的 seed → 完全相同的集合。
- 两两交集验证（实测）：`SFT∩GRPO=0`、`SFT∩Ablation=0`、`SFT∩Eval=0`、`GRPO∩Ablation=0`、`GRPO∩Eval=0`、`Ablation∩Eval=0`，**六对全 0** ✅。

---

## 三、各集合现状详解

### 1) EVAL —— Final-200 评测集

- **是什么**：训练完成后做 Base/SFT/GRPO 对比的**严格成功率**评测集，最终汇报指标就用它。
- **N**：200，范围 `[16..1458]`（全在官方 `eval` 区 `[0,1459)` 内）。
- **来源**：`data/final200.json`。
- **生成**：官方 eval 区（1459 个）seed 42 随机采样 200。**与训练数据天然零重叠**（不同 tag 区）。
- **要 teacher 采集吗**：❌ 不需要。评测时 student 自己跑环境拿 reward。
- **当前状态**：✅ 就绪。

### 2) SFT —— 监督微调训练集

- **是什么**：teacher（`gpt-5.6-terra`）的高质量解题轨迹，student 模仿学习"搜索→点击→购买"的多步交互。
- **N**：
  - **采集目标**：5000（`sft_collect_targets_5000.json`，seed 42 全 train `[1459,23421)` 随机，**已修正**旧版只覆盖前 15% 的偏斜）。
  - **实采 raw**：5000 条（`trajectories_raw/gpt-5.6-terra/trajectories_raw.jsonl`，72.9MB）。
  - **strict_success（真正入训）**：`teacher.validate` 实过 **3793 条**（75.9% pass，reward 均值 0.897）。
    raw 的 `strict_success` 标记是 3826，但 validate canonical 复核又核掉 33 条 → 实际入训 **3793**。其余 1207 条 strict 没过（没买对）不进 SFT。
- **母集质量门（`teacher.validate` 的 4 个条件，全过才进入 3,793 条母集）**：
  1. `ok == True`
  2. `strict_success == True`（4 维 r_type/r_att/r_option/r_price 全 = 1.0）
  3. `2 ≤ n_steps ≤ 30`
  4. **`legal_ratio = 1 − illegal/n ≥ 0.8`**（非法步占比 ≤ 20%）
  被砍的 33 条**全部**卡在第 4 条（legal_ratio 0.34~0.78，乱点太多）；其余 1207 条卡在 strict。
- **legal_ratio 是什么**：一条轨迹里"有效动作 / 总动作"。**非法步** = 点了当前页面不存在的东西（或动作解析不出）→ env 静默忽略、页面不变（no-op）。
  SFT 是逐字模仿 teacher，留这种脏轨迹等于教 student 乱点 → 烧光 30 步预算、eval 崩。所以过滤。
- **与上游 ShopSimulator 对齐情况**（详见 memory `shopsim-strict-aligned-legalfilter-shopA-only`）：
  - ✅ **reward / strict 完全同源**：`strict_success ≡ 上游 env 的 total_reward == 1.0`（同源 `reward_detail`，`goal.py:206`）。Final-200 eval 按上游口径打分，与论文 Rsucc 可比。
  - ⚠️ **legal_ratio ≥ 0.8 是 shopagent-rl 独有**：上游 env 对坏动作静默 no-op、不计入 reward；这道门**只筛 SFT 训练数据，不改 reward/eval 定义**。GRPO / eval 都不走它。
- **生成流程**：`sample_sft_targets.py`（采样目标）→ `teacher.collect`（teacher 跑环境采轨迹）→ `build_sft_data.sh`（train-range 过滤 + `teacher.validate` 重新校验 → `sft_train.jsonl`）。
- **要 teacher 采集吗**：✅ 需要（且**已采完**）。teacher temperature=0，所以失败任务重采无意义——这批最终 strict 入训 **3793**（raw 标记 3826，validate canonical 核掉 33）。
- **当前状态**：✅ 5000 已采完、3793 strict 已确认；**训练集 `sft_train.jsonl`（3793 条，41MB）已由 `build_sft_data.sh` 生成**（本次按要求只生成、不训练）。
- **当前论文子集**：从上述母集继续筛选 `n_steps ≤ 10`，得到 **3,624 条**，用于当前 10-turn 主实验；不覆盖母集文件。
- **独立文件**：运行 `python scripts/build_sft_horizon_subset.py` 生成
  `data/sft_train_horizon10.jsonl` 及其 `.summary.json`；该文件专供当前论文的 horizon-10 分析，
  不覆盖 `data/sft_train.jsonl`。
- **为什么停 5000 不追 10000**：原计划 10000，但 1.8B（Qwen3-1.7B-Base）判 3793 strict 够用，不再追量（详见 commit）。

### 2b) Constraint-causal / Certified 数据

- **训练来源**：全部来自官方 `train` 区；Final-200 和 heldout-v2 只用于评测，不回流训练。
- **原子干预**：同一商品、规格和指令保持不变，只改变当前价格；预算内侧目标是
  `COMMIT`，超预算侧目标是 `SEARCH_ALTERNATIVE`。option pair 同理只翻转目标规格。
- **v3 失败数据**：价格训练只包含超预算侧，并只在该侧追加 `任务约束摘要`。heldout-v2
  自然输入 price cf 为 0%；恢复摘要后为 100%，证明形成了摘要存在性捷径。
- **v4 corrective mix**：`data/sft_train_certified_corrective_mix.jsonl` 共 10,057 条：
  2,000 预算内价格、2,000 超预算价格、1,622 option、811 nuisance control，其余为基线
  SFT 保留样本。价格双侧都不追加摘要；811 条 summary-positive nuisance 用来打破旧触发器。
- **GRPO 混合**：`data/grpo_certified_natural_train.parquet` 保留 environment row，并对
  price/option/nuisance 配对同时写入 original 和 counterfactual 两侧，输入与自然评测同格式。
- **可观察性要求**：价格标签使用的预算必须能从自然指令中读出。若上游随机
  `goal.price_upper` 与人类指令预算不一致，必须分层、重建或剔除，不能要求模型预测隐藏变量。

### 3) GRPO —— 强化学习 prompts

- **是什么**：GRPO 自监督 RL 的"题目池"。**注意：GRPO 不用 teacher 轨迹**——student 自己对每个 task_id 采样多条 rollout，环境 reward 打分，组内对比算优势。所以这里只需要 **task_id 列表**。
- **N**：1000，范围 `[1476..23417]`，中位 12680（均匀预期 ~12440，随机性正常）。
- **来源**：
  - `data/grpo_prompts_1000.json`（1000 个 task_id，**本次按现 SFT 重采**）。
  - → `data/grpo_train.parquet`（veRL `RLHFDataset` 格式，1000 行，每行 `prompt`=[system_msg]，`extra_info={index:task_id}`，`task_id`，`data_source="shopsim"`）。**已重建**。
- **生成**：`sample_grpo_targets.py`——从 train 排除 **SFT + eval** 后 seed 42 随机采 1000。
- **要 teacher 采集吗**：❌ 不需要。
- **当前状态**：✅ 就绪（prompts + parquet 都已更新）。GRPO 组大小 G=4 → 单遍 1000×4 = 4000 rollout。
- **为什么是 1000 不是 config 旧的 2000**：现成 GRPO 管线（`build_grpo_data.py` 硬编码 `grpo_prompts_1000.json`）按 1000 接的；要 2000 改一行即可。

### 4) ABLATION —— 消融 held-out 诊断集

- **是什么**：消融实验（ablation）用的 held-out 诊断集，**和全部训练集 + 评测集互斥**，保证消融结果不被训练数据污染。用来诊断"去掉某个组件（如 GRPO、某段 reward）后模型表现怎么变"。
- **N**：500，范围 `[1471..23392]`，中位 12159（均匀，正常）。
- **来源**：`data/ablation_tasks_500.json`（**本次新增**）。
- **生成**：`sample_ablation_targets.py`——从 train 排除 **SFT + GRPO + eval** 三者后 seed 42 随机采 500。
- **要 teacher 采集吗**：❌ 不需要（诊断用，student 跑环境即可）。
- **当前状态**：✅ 就绪。
- **修了什么**：旧 config 的 `ablation_tasks: [1459,1959]` 是**顺序取 train 前 500**，且和 SFT 重叠 109 条——这正是本次修掉的 bug，现改为随机 + 全互斥。

---

## 四、产物文件一览

### `data/`（采样目标 / 采集结果）
| 文件 | 内容 | 大小 |
|---|---|---|
| `final200.json` | EVAL 200 个 task_id | 1.1K |
| `sft_collect_targets_5000.json` | SFT 5000 采集目标 | 33K |
| `grpo_prompts_1000.json` | GRPO 1000 个 task_id | 6.5K |
| `ablation_tasks_500.json` | Ablation 500 个 task_id | 3.3K |
| `trajectories_raw/gpt-5.6-terra/trajectories_raw.jsonl` | teacher raw 轨迹 5000 条（strict 标记 3826 → validate 3793 入训） | 72.9MB |
| `trajectories_raw_old/` | 历史采集归档（不用，留底） | — |
| `counterfactual/heldout_atomic_pairs_v2.jsonl` | task-disjoint 原子 heldout：150 option + 384 price | — |
| `sft_certified_corrective_train.jsonl` | v4 方法数据：自然格式 paired price/option/nuisance | — |
| `sft_train_certified_corrective_mix.jsonl` | v4 的 10-turn aligned feasibility 混合训练集，共 10,057 条 | — |
| `sft_train_horizon10.jsonl` | 从 3,793 条母集筛出的 `2 ≤ n_steps ≤ 10` 子集，共 3,624 条 | — |

### `scripts/`（采样/构建脚本，全可复现）
| 脚本 | 作用 |
|---|---|
| `sample_sft_targets.py` | 全 train 随机采 SFT 目标（排除 eval+grpo），seed 42 |
| `sample_grpo_targets.py` | 全 train 随机采 GRPO 目标（排除 **SFT**+eval），seed 42 |
| `sample_ablation_targets.py` | 全 train 随机采 Ablation 目标（排除 **SFT+GRPO**+eval），seed 42 |
| `build_sft_data.sh` | raw jsonl → train-range 过滤 + validate → `sft_train.jsonl` |
| `build_grpo_data.py` | grpo_prompts + system_prompt → `data/grpo_train.parquet` |
| `build_paired_sft_data.py` | 构造 paired constraint SFT；价格默认无 summary，可选双侧 |
| `build_certified_grpo_data.py` | 构造自然格式 environment + paired counterfactual GRPO parquet |

### `data/`
| 文件 | 内容 |
|---|---|
| `grpo_train.parquet` | GRPO 训练 parquet（1000 行，veRL 格式），**已重建** |
| `grpo_certified_natural_train.parquet` | v4 后续 Certified GRPO 输入；不含价格 summary |
| `grpo_certified_natural_800_pairblocked.parquet` | 当前 matched GRPO 输入；400 environment + 200 pairs×2 = 800 行，正好覆盖 200 steps；采样前按 3,000 字符过滤整对超长 CF prompt，当前最大 1,894 tokens，不触发 2,048-token 训练过滤 |

---

## 五、与旧版的差异（这次修了什么）

| 集合 | 旧版（错） | 新版（对） |
|---|---|---|
| **SFT 覆盖** | 只覆盖 train 前 15%（1459–4824），偏斜；~2600 条 | 全 train `[1459,23421)` 随机 5000 目标，3793 strict |
| **GRPO 排除集** | 按旧 SFT(2613，污染 v1) 排除生成 | 按现 SFT(5000) 重采，来源干净 |
| **Ablation** | 顺序 `[1459,1959]`，**与 SFT 重叠 109** | 随机 500，与 SFT/GRPO/eval 全互斥 |
| **config 数字** | grpo 2000 / ablation 3000 / 采集 10000（都是计划值，过期） | 全部对齐实际：grpo 1000 / ablation 500 / 采集 5000（3793 strict） |

---

## 六、下一步（数据侧）

1. **Corrective SFT v4 已完成并通过两道门**：natural heldout-v2 price cf 78.65%、price
   paired robust 73.70%，Final-200 strict 31.5%。
2. **数据有效性复核**：冻结 v4 数据与 adapter hash，并核对自然指令预算与标签预算的一致性。
3. **保留 pair identity**：后续 parquet / sampler 必须稳定保留 `pair_id`、两侧标记、干预类型和
   期望动作关系，不能把 paired 数据仅当作独立行混洗。
4. **Matched GRPO 对照**：从同一 v4 adapter 起训，先运行 `CF-GRPO w/o Pair`，再运行加入
   relation reward 的 Paired GRPO；两者只允许在 `R_pair` 上不同。
5. **补齐评测控制**：增加 matched one-sided hard-negative、summary ablation、nuisance invariance、
   constraint-type holdout 和多 seed 对照。完整研究顺序见
   [`docs/constraint-faithful-paired-policy.md`](docs/constraint-faithful-paired-policy.md)。
6. **当前论文 horizon**：从基础 `sft_train.jsonl` 的 3,793 条母集中筛选 `n_steps ≤ 10`，得到
   3,624 条基础子集；现有 10,057 条 v4 aligned mix 即按此口径构建。超过 10 步的 169 条保留作后续
   long-horizon 扩展，不进入当前主实验。设计细节见
   [`docs/trajectory-horizon-and-long-horizon.md`](docs/trajectory-horizon-and-long-horizon.md)。
   设计细节见 [`docs/trajectory-horizon-and-long-horizon.md`](docs/trajectory-horizon-and-long-horizon.md)。
