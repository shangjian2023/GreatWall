# ADR-0025: 训练侧 provenance 与无参考检测配置隔离

- **状态**: Accepted
- **日期**: 2026-07-14
- **决策者**: 项目组
- **相关**: ADR-0023、0024

## 背景

无参考主检测不得读取 training trigger、target payload、中毒数据、训练 YAML 或干净参考模型。此前平台默认的 `configs/detection.yaml` 仍保留历史攻击候选和目标文本，且 CLI 可以将训练 YAML 作为 `--config` 读入。即使当前实现未主动使用这些字段，这个输入能力本身破坏了实验隔离。

同时，训练矩阵输出只有权重、训练指标和质量 gate，缺少统一记录的配置哈希、代码版本、依赖版本、执行角色和产物哈希，难以审计中断后恢复或后续重新校准的来源。

## 决策

1. `reference_free_soft_probe` 只接受检测运行配置：顶层仅允许 `schema_version`、`model`、`runtime`；不得含 `attack`、`attacks`、`train`、评测真值或任意其他段。
2. CLI 在加载模型前校验该约束；平台在构造子进程命令前再次校验。参考辅助与 Oracle 保持历史配置兼容，但不得影响无参考主分数。
3. 每个通过 `scripts.run_implicit_matrix` 启动的训练 cell 在训练前创建 `training_manifest.json`，仅写训练侧元数据、配置哈希、命令哈希、代码版本、依赖环境和后续产物哈希；不得序列化 trigger、payload、target marker 或训练侧完整命令。
4. 后门 cell 的训练侧 quality gate 未通过时，运行器写 `quality_rejected` 并以失败结束；只有通过的产物才可以进入后续无参考开发或盲测检测。
5. 每个隐式攻击族保留 3 个独立 development seed，作为唯一允许指导算法迭代的带标签后门集合。盲测 seed 在配置冻结前不得运行检测。

## 后果

- 旧的无参考 MVP 报告若曾使用训练 YAML，只能作为 pre-isolation 工程演示，不能用于正式校准或盲测结论。
- 重新生成检测报告时必须使用干净检测运行配置；训练侧配置和 quality JSON 仍只服务于训练验收与评测真值流程。
- 训练 manifest 不是检测报告，也不应被检测器、校准器或浏览器端读取。
