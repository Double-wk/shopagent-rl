# Archive

这里保存历史文件，不参与当前运行。按来源分组：

- `configs/`：旧 teacher 配置和重复的旧 SFT/GRPO 配置。
- `scripts/`：停用的环境/采集脚本。
- `run/`：旧 GRPO、环境和采集日志；`teacher_compare/` 是历史模型对比。
- `outputs/`：旧 Hydra 输出和旧 SFT checkpoint。
- `hydra_runs/`：Hydra 自动生成的每次 GRPO 启动配置快照；2026-08-10 是
  旧调参尝试，2026-08-12 包含 smoke、边界测试和诊断训练的快照。
- `experiment_legacy/`、`run_legacy/`：原有归档目录的集中迁移。

当前入口只看项目根 README、`configs/`、`scripts/`、`experiment/` 和 `data/`。
