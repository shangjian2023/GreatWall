# ADR-0016: 分离研究流水线与平台编排层

- **状态**: Accepted
- **日期**: 2026-07-10
- **更新**: 2026-07-15
- **决策者**: BdShield 项目组
- **相关**: ADR-0001、ADR-0014、ADR-0017

## 背景 (Context)

项目已有两套入口：`scripts/detect_trigger.py` 是含已知候选池的旧验证路线，`scripts/invert_trigger.py` 是从输出到输入的正式 trigger inversion(触发器逆向)路线。旧版 Web API(应用程序接口)仍调用前者，导致演示平台展示的结果与当前方法学、README 和实验报告不一致。同时，研究脚本输出过两种 JSON(JSON对象表示法)结构，前端直接绑定旧字段，无法展示当前端到端结果。

竞赛演示还要求长时间模型任务不能阻塞 HTTP(超文本传输协议)请求，且任何“未找到触发器”的结果都必须保持 inconclusive(无结论)，不能误报成模型安全。

## 决策 (Decision)

平台层与研究流水线分离：

1. 旧平台的 `reference_free_soft_probe` 和 `reference_assisted` 任务调用 `scripts.invert_trigger`；正式盲检不传 `target_text`(目标输出)、`--skip_stage1` 或旧候选池参数。
2. 新增 `competition_sequence_probe` 作为 `competition_core` 的平台桥接模式。它只允许 `coverage_audit`、`general` 场景和单待审模型，固定使用 `competition_core/configs/gpt2_detection_4060.yaml`，并拒绝 clean reference、`target_text` 和旧 soft-probe calibration。
3. `scripts/run_competition_scan.py` 将 tokenizer 完整词表划为四个分片，逐片调用 `competition_core mine`，合并候选后调用 `competition_core probe` 评估当前配置中的 Top-4。编排脚本只产生进度事件和报告封装，不承载或复制竞赛算法。潜变量阶段先发送 `competition_probe_inputs`，再将逐步记录按最多 8 步聚合为 `competition_probe_steps`；聚合只服务实时轮询，每一步仍完整写入最终报告。
4. `src/api/report_adapter.py` 将当前盲检报告、负对照报告和竞赛序列探测报告归一为稳定的 `schema_version=1.0` 平台契约。竞赛报告中的固定概率判据与候选族支持只作为开发证据；平台结论无条件保持 `INCONCLUSIVE`，不得提升为 `DETECTED`。
5. `src/api/jobs.py` 负责异步子进程、阶段进度、日志、取消和结果读取；配置路径必须位于项目目录内，模型路径只允许来自项目工作区、Hugging Face 本机缓存、由 `BDSHIELD_MODEL_ROOTS` 显式登记的根目录，或用户在当前服务进程中添加的训练根目录。本地 LoRA 按 `adapter_config.json` 声明的基座模型加载；若待审与参考已知基座不一致，或指向同一模型产物，平台在启动前拒绝该组合。
6. 研究代码继续输出包含完整中间量的原始报告，平台适配器只读，不反向影响算法；其中 Stage 2 必须保存 HotFlip 轨迹、局部字母精修的候选排名和每个 `target_text` 的执行或提前停止原因。
7. 平台对缺少这些字段的历史报告明确显示数据缺失，不根据最终触发器反推未保存的搜索过程。
8. Web 工作台使用 `DETECTED / SUSPICIOUS / INCONCLUSIVE / CONTROL_CLEAR` 四类结论，只有证据闭环才能给出高风险裁决；`competition_sequence_probe` 当前不具备进入该闭环的正式 clean 校准。

## 理由 (Rationale)

这保持了“输出异常发现 -> 输入触发器逆向 -> 正向复现”的方法学方向，同时允许研究报告继续快速迭代。稳定适配层避免每次算法增加字段都重写前端。后台任务模型使浏览器可见进度并避免 600 秒同步超时。竞赛模式复用同一任务与报告契约，同时保持 `competition_core` 的算法所有权和开发证据边界。

## 后果 (Consequences)

### 正面

- 演示平台与正式反演算法使用同一条主链路。
- 竞赛序列挖掘与潜变量探测可以在同一工作台按四分片进度、合并候选和 Top-4 轨迹展示。
- 现有实验产物可以直接进入模型准入审查界面。
- “无结论”与“负对照未触发”在产品语义上不再混淆。
- 加载器可识别 LoRA/QLoRA 适配器格式和全量模型目录；检测有效性仍需分别实测。

### 负面 / 风险

- 平台契约新增字段时需要维护适配器测试。
- 竞赛候选族信号仍缺少正式 clean 校准，当前平台只能返回 `INCONCLUSIVE`。
- 当前任务状态只保存在进程内存中，服务重启后不能恢复正在运行的任务。
- 平台可选择工作区和本机 Hugging Face 缓存中的本地模型；远程 Hugging Face 模型仍须先下载。整盘扫描不在范围内，其他本机目录须通过 `BDSHIELD_MODEL_ROOTS` 显式登记。

### 后续动作

- 完成 Qwen2、Baichuan2、Falcon 的端到端实验，不以接口兼容替代实测。
- 增加 QLoRA(量化低秩适配)和 full fine-tuning(全量微调)的独立训练样本。
- 若平台进入多人使用阶段，将任务状态迁移到持久化队列。

## 考虑过的替代方案 (Alternatives Considered)

### 替代 A: 让前端直接解析所有原始报告

否决：原始字段随算法实验变化，前端会与研究实现强耦合，也无法统一旧负对照与当前盲检语义。

### 替代 B: 保留同步 `/api/detect` 接口

否决：完整检测耗时可达数分钟至十几分钟，同步请求不能可靠展示进度、取消任务或保留滚动日志。

### 替代 C: 为演示继续调用旧候选池路线

否决：该路线包含已知触发器候选，违反 ADR-0001，不能作为竞赛中的 trigger inversion(触发器逆向)成果。

## 参考 (References)

- `ADR-0001`: 触发器反演方向
- `ADR-0014`: 多起点 Beam HotFlip(束搜索梯度翻转)
- `ADR-0017`: reference-assisted(参考模型辅助)主路径
