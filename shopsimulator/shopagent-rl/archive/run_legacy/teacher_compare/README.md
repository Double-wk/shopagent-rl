# Teacher 模型探针存档（5 模型 | 2026-08-07）

> ⚠️ **历史选型存档**——本目录结论曾经历 `deepseek → grok-4.5` 的反复（由 API 余额耗尽驱动），
> **最终选定 teacher = `gpt-5.6-terra`**（5000 采 / 3793 strict 入训，75.9% pass）。
> 下面所有「结论」均为**选型过程记录**，当前实际 teacher 以 [`../../DATA.md`](../../DATA.md) 为准。

shop_A teacher 选型的全部探针数据 + 分析，自包含、可复现。
**主入口**：先读 [`teacher_comparison.md`](teacher_comparison.md)（分析报告 + 结果表 + 结论）。

## 目录结构
```
run/teacher_compare/
├── teacher_comparison.md   ← 分析报告（结果表 + 结论）：先读这个
├── README.md               ← 本文件
├── probe_compare.py        ← deepseek+glm 对比探针脚本（可重跑/扩展）
├── <model>/                ← 每个模型一个子目录，结构统一：
│   ├── trajectories_raw.jsonl   完整轨迹（system+全程对话 Thought/Action）
│   └── usage.tsv                逐次 API token 记账
├── gpt-5.5/  gpt-5.6-sol/  gpt-5.6-terra/    ← gpt 三模型（既有探针）
└── deepseek-v4-flash-0731/  glm-5.2/          ← 本次对比（run1+run2）
```

## 模型 → 原始来源映射
| 子目录 | 来源 |
|---|---|
| gpt-5.5 | data/probe_train55（已移入） |
| gpt-5.6-sol | data/probe_train56（已移入） |
| gpt-5.6-terra | data/probe_train56_terra（已移入） |
| deepseek-v4-flash-0731 | 拆自 `trajectories_probe.jsonl`（本次探针） |
| glm-5.2 | 拆自 `trajectories_probe.jsonl`（本次探针） |

## 数据格式
**`<model>/trajectories_raw.jsonl`** —— 每行一条任务轨迹，字段：
`task_id, ok, messages(=system+全程对话), n_steps, illegal_steps, reward, reward_detail, purchase, goal, reached_cap, strict_success`
（与全量采集 `data/trajectories_raw/<model>/trajectories_raw.jsonl` 同构，区别：这里是 30-task 探针子集。）

**`<model>/usage.tsv`** —— 每行一次 API 调用，4 列（tab 分隔）：
`model<TAB>prompt_tok<TAB>completion_tok<TAB>reasoning_tok`

## 同口径（公平对比）
见 `teacher_comparison.md`。要点：同一批 30 task_id、官方 `PROMPT_TEMPLATE_zh`、`temp=0.0`、`max_tokens=512`、`max_turns=30`。
⚠️ env 每次 reset 随机化商品价格 → `r_price` 跨 run 波动；但补全失败行后趋势稳定。**当时结论：选 deepseek-v4-flash-0731**（详见 `teacher_comparison.md`）；**后因余额耗尽转 grok-4.5，最终定 gpt-5.6-terra**（见本文件顶部 banner + [`../../DATA.md`](../../DATA.md)）。

## 如何重跑 / 扩展
```bash
# 前置：pack_api(:5000) 健康（GET / 返 404；reset 全失败见 memory: pack-api-env-pool-leak，需重启）
cd /workspace/shopsimulator/shop_A
/overlay/miniconda3/envs/shop-A/bin/python run/teacher_compare/probe_compare.py
```
- 加新 teacher：在 `probe_compare.py` 的 `CONFIGS` 追加 config 路径，确保其 `system_prompt` 为官方 `PROMPT_TEMPLATE_zh`。
- 探针只统计 + 落盘到本目录，不污染全量采集（`data/trajectories_raw/<model>/`）。
- 脚本重跑会重新生成合并版中间产物（`trajectories_probe.jsonl` / `probe_compare_usage.tsv` / `probe_compare.log`）；最终归档数据在各 `<model>/` 子目录。
