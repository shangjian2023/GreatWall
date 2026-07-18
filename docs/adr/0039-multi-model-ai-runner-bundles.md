# ADR-0039: 多模型 AI 执行包与强制回传契约

- **状态**: Accepted
- **日期**: 2026-07-18
- **决策者**: 项目组
- **相关**: ADR-0029、0036、0037、0038

## 背景

第一份 OPT-125M 组员包成功返回完整训练、mining 和 probe 报告，但 Adapter 权重由可选
`--include-adapters` 控制，默认成功包没有两份模型权重。本机因此无法按新候选策略重跑、
核对高支持 rank 或执行交互回放。执行者主要由 AI 代理驱动，单纯在 README 中提醒不足以
形成可靠协议。

论文实验模型表还包含 Pythia-70M、DialoGPT-medium 与 Llama-3.2-1B。当前 Competition
Core 已具备 GPT-2、OPT、GPT-NeoX 和 Llama 的显式 LoRA target map，可在同一真值隔离
runner 下扩展这些架构。

## 决策

1. 新建参数化 `scripts.run_team_model_pair`，每个源 ZIP 只通过严格 `bundle_spec.json` 和
   三份 YAML 固定模型、seed、资源预算、分片数与输出名，不复制多套算法代码。
2. 第一批构建 6 个包：旧 OPT repair、新 OPT seed、两个 Pythia seed、一个 DialoGPT seed、
   一个 Llama seed。repair 只复用原 Adapter/mining 并按新配置重跑；其余均训练完整 matched
   pair。成功后新增 5 个配对、10 个模型样本。
3. 源 ZIP 是确定性、source-only 私有包，包含训练配置但不包含论文、训练样本、本地报告或
   模型权重。每个源文件由 `bundle_manifest.json` 记录大小与 SHA256，运行前自校验。
4. `START_HERE_AI.md` 将 AI 定义为执行者而非算法修改者：禁止改源码、YAML、阈值、seed、
   模型、数据集和预算；禁止向检测传训练真值；中断后只能从原目录恢复。
5. 成功回传不再提供排除 Adapter 的开关。backdoor/clean 都必须包含
   `adapter_config.json` 和 `adapter_model.safetensors` 或 `.bin`，否则拒绝生成成功包。
6. 成功包还必须包含两侧训练 manifest、质量门、全部分片、mining、probe、逐候选软触发
   工件、日志、配置、环境版本和 pip freeze。打包后重新打开 ZIP，逐文件验证大小/SHA，
   并把 Adapter 文件重新绑定到 probe 中的模型指纹。
7. 只有二次验证通过才打印 `RETURN_VERIFIED` 和 `RETURN_READY`。运行失败不得调参强行通过，
   而是生成带错误、日志和已有进度的 `FAILURE_RETURN`；依赖安装前失败也生成 bootstrap 包。
8. Llama 包要求显式 Hugging Face 授权与 token，使用物理 batch 1、梯度累积 8、词表 batch
   32 和 8 个分片。授权、显存或磁盘不满足时直接失败，不自动替换模型。
9. 组员端不拟合阈值，只回传原始候选证据。GPT-2 v2 规则仅作为跨模型观察字段，正式
   development profile 由队长按 ADR-0038 统一生成。

## 后果

- 回传 ZIP 明显大于第一版，因为两份 LoRA Adapter 成为强制内容。
- 组员可以重复相同命令恢复 epoch 与分片，队长收到的成功包可独立加载和复核。
- Llama 完整词表与 1B 模型运行最慢，预估值只能作为排期参考；实际时间与显存必须回传。
- Qwen、Falcon、Baichuan 不进入第一批：它们不是本次论文五模型补齐目标，且当前训练器
  尚无对应 LoRA target map。
