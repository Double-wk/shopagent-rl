# Teacher 模型同条件对比

> ⚠️ **历史选型报告（2026-08-07/08）**——下面的「结论」经历了 `deepseek（理想正选）→ grok-4.5（余额耗尽后务实替代）` 的过程。
> **最终选定 teacher = `gpt-5.6-terra`**（mcgrox 渠道重新可用后定稿；5000 采 / 3793 strict 入训）。
> 本表的数据仍是有效的同口径对比，但「当前该用谁」以 [`../../DATA.md`](../../DATA.md) 为准。

**目的：** 为 shop_A 选定 teacher（SFT/GRPO 数据来源）。在完全相同条件下对比候选模型的**质量**(strict_success / reward / 步骤有效比 legal_ratio)与**成本**(token·成功)。

## 同口径（公平对比前提）
- **task_id**：同一批 30 条（取自 `run/teacher_compare/gpt-5.6-sol/trajectories_raw.jsonl`；已校验 gpt-5.5 / 5.6-sol / 5.6-terra 三探针 set 完全相同）。
- **system_prompt**：官方 ShopSimulator `PROMPT_TEMPLATE_zh`（所有模型逐字一致）。
- **采样**：`temperature=0.0`（teacher 决策确定）、`max_tokens=512`、`max_turns=30`。
- **token 记账**：`OpenAITeacherClient._log_usage`（model / prompt / completion / reasoning 四列）；`token/成功 = Σprompt_tokens / strict_count`。
- **判据**：`strict_success = r_type=1 ∧ r_att=1 ∧ r_option=1 ∧ r_price=1`（`shop_env/reward.py`）。
- 探针脚本：`probe_compare.py`（usage 日志 `probe_compare_usage.tsv`）。
- ⚠️ **跨 run 波动源**：env 每次 reset 随机化商品价格（`engine.generate_product_prices`），故 `r_price` 维度（进而 strict/reward）跨 run 波动——即使 temp=0。但 strict_success 的趋势稳定（见下）。

## 结果

| 模型 | strict | reward | steps | prompt_tok | reason_tok | tok/成功 | 来源 |
|---|---|---|---|---|---|---|---|
| gpt-5.5 | 70% (21/30) | 0.889 | 5.9 | 847k | 19,020 | 40,339 | 既有探针 |
| gpt-5.6-sol | 87% (26/30) | 0.971 | 5.9 | 698k | 17,146 | 26,848 | 既有探针 |
| gpt-5.6-terra | 80% (24/30) | 0.961 | 6.0 | 679k | 19,961 | 28,311 | 既有探针 |
| deepseek-v4-flash-0731 (run1) | 73% (22/30) | 0.796 | 13.9 | 4,022k | 149,692 | 182,836 | 第1次探针 |
| deepseek-v4-flash-0731 (run2) | 73% (22/30) | 0.809 | 14.2 | 3,996k | 153,001 | 181,620 | 第2次探针(本次) |
| glm-5.2 (run1) | 67% (20/30) | 0.749 | 11.9 | 3,727k | 136,973 | 186,344 | 第1次探针 |
| glm-5.2 (run2, 补全†) | 70% (21/30) | 0.792 | 11.0 | 4,244k | 150,595 | 202,131 | 第2次探针(补全后) |
| grok-4.5 | 53% (16/30) | 0.778 | 7.7 | 1,504k | 279,272 | 94,029 | probe_grok.py(2026-08-08,完整重跑30条存档) |

> **grok-4.5**(2026-08-08)：deepseek/gpt 余额耗尽(402/停)后转用的免费 teacher(via api.ziliao.xyz "grok free")。strict 53%(完整重跑30条；首跑57%/二跑53%，env 价格随机化致 r_price 波动，趋势 ~55%)仍是各 teacher 里最低；reward 0.778(首跑0.843，波动)、步数 7.7(含 18910/21721 两 looper 各30步拉高，非 looper 任务仅~5步)。免费 + 55%×23421≈12900 条 strict 够 SFT 5000，可与已有 deepseek 4310 条(strict 64.8%)混合。**jsonl 已存档**：run/teacher_compare/grok-4.5/trajectories_raw.jsonl(30条)+usage.tsv(230行)。

> † glm run2 首跑有 2 条任务（18910/21721）报 env `ReadTimeout` 失败，曾出现「21/28=75%」的假象。**单线程重跑后这两条均为 glm looper**（steps=30 卡满、reward=0），deepseek 在同 2 任务上成功。故补全为 21/30=70%（glm 子目录 jsonl 已更新为 30 条完整轨迹；usage 含 2 条 looper 重跑 token，故 tok/成功 偏高——这是 glm 的真实成本）。

> **gpt-5.6-luna**：无既有探针数据，未纳入。

## 步骤有效比（legal_ratio）对比

> 2026-08-09 补充的维度：上面「结果」表只比了 strict/reward/steps/tok，**没列每步动作的有效比**。各模型轨迹 jsonl 里 `illegal_steps`/`n_steps` 字段齐全（探针经 `teacher.collect.run_one` → `shop_env` wrapper 的 `env.step()` → `info["legal"]` 记录，与 SFT 数据同口径），现补算如下。**注**：glm/deepseek 各存的是「补全后 / 本次」的完整 30 条 run 的 jsonl，故 run1/run2 不分开。

**定义**：`legal_ratio = 1 − illegal_steps / n_steps`。非法步 = 动作解析不出 / `click` 命中不上当前页可点元素 → 上游 env 静默 no-op（页面不变、不报错）。这是 shop_A **独有的 SFT 数据卫生门**（`≥0.8` 才入训），**不改 reward/eval 定义**——GRPO / eval 都不走它，仅筛 SFT 训练数据（详见 [`../../DATA.md`](../../DATA.md)）。

### 汇总（两种口径）

| 模型 | 非法步 | 整体有效比 | 平均有效比 | ≥0.8 占比(SFT 门) |
|---|---|---|---|---|
| gpt-5.5 | 0/178 | 1.000 | 1.000 | 100% |
| gpt-5.6-sol | 0/178 | 1.000 | 1.000 | 100% |
| gpt-5.6-terra | 0/181 | 1.000 | 1.000 | 100% |
| grok-4.5 | 6/230 | 0.974 | 0.993 | 100% |
| glm-5.2 | 187/330 | 0.433 | 0.700 | 60% |
| deepseek-v4-flash-0731 | 228/425 | 0.464 | 0.607 | 40% |

- **整体有效比** = `1 − Σ非法步 / Σ总步数`（把所有步 pool 到一起算一个比）：回答"采到的**所有动作**里有多大比例有效"。looper（卡满 30 步乱点）因步多权重大、会把它拖低。
- **平均有效比** = 各轨迹先各算 `1 − illegal/n`、再对这些比值求平均：回答"**平均每条任务**本身干不干净"。每条只算一票、looper 影响小。
- **最小例**：一条 looper（30 步 25 非法，0.17）+ 一条干净（4 步 0 非法，1.00）→ 整体 `1 − 25/34 = 0.26`，平均 `(0.17+1.00)/2 = 0.585`。同一份数据，整体被 looper 拉低。

### 逐条分布（有效比分桶，每模型 30 条）

| 模型 | <0.2 | 0.2-0.4 | 0.4-0.6 | 0.6-0.8 | 0.8-1.0 |
|---|---|---|---|---|---|
| deepseek-v4-flash-0731 | 3 | 6 | 4 | 5 | 12 |
| glm-5.2 | 5 | 2 | 4 | 1 | 18 |
| gpt-5.5 / 5.6-sol / 5.6-terra | 0 | 0 | 0 | 0 | 30 |
| grok-4.5 | 0 | 0 | 0 | 0 | 30 |

→ gpt 三兄弟 30 条**全部 ≥0.8**（0 非法步）；glm/deepseek 明显**双峰**——底部一撮 looper（<0.4）+ 顶部干净任务。这正是它们"整体有效比"被拖低、而"平均有效比"尚可的原因。

### 解读（选 teacher 的额外支撑）

报告按 strict/reward/tok 选定 `gpt-5.6-terra`；legal_ratio 给出**没被原表记录的另一个理由**：gpt 不仅任务成功率高，**每一步动作都干净（0 非法步）**，SFT 学到的是纯有效动作序列；而 deepseek/glm 轨迹里混了大量静默 no-op 的乱点（looper 30 步里 18~25 步无效），对 student 是噪声。这也呼应 `DATA.md` 里"被 validate 砍掉的 33 条**全是** `legal_ratio < 0.8` 的乱点轨迹"——deepseek/glm 若全量采，过 SFT 卫生门的损耗会明显大于 gpt。

## 结论（已被取代）：当时全量用 grok-4.5（deepseek/gpt 余额都耗尽）

> 📌 **更新**：此结论已被取代——mcgrox 渠道重新可用后，**最终 teacher 定为 `gpt-5.6-terra`**（5000 采 / 3793 strict）。下面这段是「余额耗尽期」的决策记录，留作选型过程。

**质量最优本应是 deepseek-v4-flash-0731**（strict 73%、无 looper、稳定），但 deepseek 与 gpt 的 API 余额在 2026-08-07/08 相继耗尽(402)，断点续采停摆。当前全量采集转用 **grok-4.5**（via api.ziliao.xyz，免费 "grok free"）——余额耗尽后唯一可用的免费 teacher。

grok-4.5 评估（见上表）：
- **strict 53%**（首跑 57%/二跑 53%，env 价格随机化致 r_price 波动；各 teacher 里最低）
- reward 0.778、步数 7.7（非 looper 仅 ~5 步全场最少）、reasoning 模型(tok 偏高)
- **免费 + 55%×23421≈12900 条 strict 够 SFT 5000**，可与已采 deepseek 4310 条(strict 64.8%)混合提升整体质量
- 难任务 18910/21721 是 grok 的 looper（glm/deepseek 当初也卡，本质重任务）

> 理想正选仍是 deepseek(质量/成本/稳定性最优)；grok-4.5 是「没钱之后」的务实替代，靠免费 + 量补质。后续若 deepseek 充值可切回断点续采。

## 决策记录
- (2026-08-07) mcgrox(gpt) 没钱 → 停 gpt-5.6-sol 全量采集。已存约 900 条 gpt-5.6-sol 轨迹于 `data/trajectories_raw/gpt-5.6-sol/`，断点续采可用。
- (2026-08-07) 理想 teacher 选 deepseek-v4-flash-0731：质量(strict/reward)、成本(tok/成功)、稳定性(无 looper)三项均优于 glm-5.2。glm-5.2 备选（智谱 Coding Plan，单价待核算；若 deepseek 余额吃紧可切换）。
- (2026-08-08) **deepseek 余额也耗尽(402) → 全量采集转 grok-4.5**（免费，via api.ziliao.xyz）。质量最低(strict 53%)但免费+量够 SFT，可与已采 deepseek 4310 条混合。理想正选仍 deepseek，充值后切回断点续采。
- 过程提示：单次 30-task ×1 曾因 glm 2 条 env 超时误判 glm=75%；补全后还原为 70%。**教训：探针失败行（尤其超时）务必单线程重跑确认，避免把运维故障误读为模型质量。**

## 各模型接入
- **deepseek-v4-flash-0731** → tokenrhythm：`https://tokenrhythm.studio` + `/v1/chat/completions`（标准 OpenAI 路径）。推理模型（带 reasoning_tokens）。2 keys。
- **glm-5.2** → 智谱「国内 Coding Plan」：`https://open.bigmodel.cn/api/coding/paas/v4` + `/chat/completions`（**路径无 /v1**，靠 `chat_completions_path` 覆盖）。⚠️ 严禁通用 `/api/paas/v4`（配错清 Key）。
- 配置：`configs/teacher_deepseek-v4-flash-0731.yaml`、`configs/teacher_glm-5.2.yaml`（旧 anthropic 版备份 `teacher_glm-5.2.anthropic_backup.yaml`）。
