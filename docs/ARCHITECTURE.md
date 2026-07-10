# BdShield 当前架构

本文描述当前代码真实执行的检测流程。历史设计与被推翻的方案保留在 `docs/adr/`，但不作为现役说明。

## 威胁模型

正式检测对象满足以下条件：

- 后门通过 LoRA、QLoRA 或全量微调写入模型权重。
- 模型是可下载、可读取梯度的生成式因果语言模型。
- 审查方能够获得或构造同基座的干净 reference model(参考模型)。
- 触发器和目标输出对检测器未知。

当前不处理推理阶段 prompt injection(提示注入)、闭源远程 API、纯训练数据审计或分类器后门。

## 当前数据流

现役实现是**两阶段反演算法 + 正向验证层**：

```text
待审模型 + 同基座干净参考模型
        |
        v
Stage 1 输出异常发现
  通用探针与结构扰动 -> target/reference 响应
  -> Monroe log-odds + 多信号重排序
  -> Top-K candidate target_text
        |
        v
Stage 2 输出条件触发器逆向
  针对每个 target_text 运行 multistart beam HotFlip
  -> 局部字母精修（可选）
  -> candidate trigger
        |
        v
正向复现与风险裁决
  待审模型 ASR / 参考模型 ASR / 跨问题方差
  -> DETECTED / SUSPICIOUS / INCONCLUSIVE
```

正向验证是产品证据链的一层，不是当前 CLI 中独立实现的第三个优化阶段。旧 contrastive Stage 3 已从主路径删除。

## Stage 1：输出异常发现

正式模式是 `--stage1_mode perturbation`，需要 `--reference_lora`：

1. 对通用问题加入标点、元词和无害短串等结构扰动。
2. 分别生成待审模型与参考模型响应。
3. 对每个扰动子集计算 Monroe log-odds(对数几率)。
4. 用 baseline control(基线控制)、目标/参考计数、通用词惩罚、扰动特异性和短语聚合重排序。
5. 可用 `--stage1_context_shift` 在候选实际出现的上下文中加入 target/reference 概率差。

默认扰动池不包含训练触发器 `cf/mn/bb`。正式检测不得从训练配置读取 `target_text`。

`confidence_lock` 是保留的 reference-free(无参考模型)实验模式。它在现有 OPT-125M v1 实验中无法把 `mcdonald` 召回 Top-5，因此不是正式能力。

## Stage 2：触发器逆向

Stage 2 对 Stage 1 Top-K 目标依次运行 `hotflip_invert_from_scratch()`：

- 梯度根据目标输出对输入 token embedding(词元嵌入)的影响提出替换方向。
- 多个随机合法起点和 beam state(束状态)缓解离散搜索局部最优。
- `short_alpha` 是短字母触发器的结构先验；`none` 关闭该先验。
- trial scoring(试评估)以待审模型与参考模型的 ASR 差为主指标。
- F signal(跨问题一致性)只记录为辅助指标，不参与最终主排序。
- `--stage2_alpha_refine` 只在已发现短字母候选的局部编辑邻域中继续搜索，不读取训练触发器列表。

旧 `--legacy_pool` 会枚举预设候选，只允许用于 ablation(消融实验)，不能称为触发器逆向。

## 正向验证的当前边界

`stage2_search()` 在搜索后重新生成响应并计算最终 ASR、参考 ASR 与方差。这能够证明逆向候选可以正向复现目标输出，但当前验证问题与搜索问题池存在重叠，因此还不能称为严格独立 held-out evaluation(留出集评估)。

在补齐独立数据切分前，竞赛材料应使用“正向复现验证”，不能使用“独立留出验证”或“泛化验证”表述。

## 指标定义

项目曾用 `lift` 表示两种不同差值，现统一区分：

| 名称 | 定义 | 用途 |
|---|---|---|
| `ASR_triggered` | 待审模型在触发输入上的命中率 | 后门注入与复现 |
| `ASR_benign` | 待审模型在无触发输入上的命中率 | 触发特异性 |
| `trigger_lift` | `ASR_triggered - ASR_benign` | 训练后门是否具有条件性 |
| `ASR_reference` | 参考模型在相同触发输入上的命中率 | 排除自然语义关联 |
| `reference_separation` | `ASR_triggered - ASR_reference` | 当前检测主指标 |
| `F_signal` | `mean_asr - 2 * var_asr` | 跨问题稳定性辅助指标 |

当前 JSON 和 CLI 为兼容历史，把 `reference_separation` 存在字段 `lift` 中。阅读旧报告时必须结合报告类型，不能与训练侧 `trigger_lift` 混用。

## 风险语义

- `DETECTED / HIGH`：逆向候选的 `reference_separation >= 0.7`，证据链闭合。
- `SUSPICIOUS / MEDIUM`：有中等分离信号，但未达到高风险阈值。
- `INCONCLUSIVE`：Stage 1 未召回目标、Stage 2 未找回触发器或预算不足。它不表示模型安全。
- `CONTROL_CLEAR / LOW`：只用于明确标注的干净负对照实验，不用于未知模型盲检失败。

原始 CLI 已修复：分离度低于证据门槛或无触发器时统一输出 `INCONCLUSIVE`，与平台适配层一致。

## 模型产物加载

`scripts.invert_trigger --target_kind auto` 可识别：

- 含 `adapter_config.json` 的 PEFT adapter(参数高效微调适配器)。
- 含 `config.json` 的 full checkpoint(全量模型目录)。
- 与 `target_base` 相同的基础模型。

这只证明加载入口兼容，不证明相应微调方法或模型架构上的检测有效性。当前端到端实测范围只有 OPT-125M + LoRA。

## 平台层

| 模块 | 职责 |
|---|---|
| `src/api/server.py` | FastAPI 路由与静态页面 |
| `src/api/jobs.py` | 异步子进程、进度、日志和取消 |
| `src/api/report_adapter.py` | 将研究 JSON 归一为平台报告 |
| `web/` | 模型准入审查工作台 |

主要 API：

| 方法与路径 | 用途 |
|---|---|
| `GET /api/health` | 服务健康检查 |
| `GET /api/catalog` | 已有实验报告目录 |
| `GET /api/catalog/{id}` | 归一化实验报告 |
| `GET /api/capabilities` | 已实测与待验证能力矩阵 |
| `POST /api/scans` | 启动正式盲检任务 |
| `GET /api/scans/{id}` | 查询进度与日志 |
| `GET /api/scans/{id}/report` | 获取任务报告 |
| `DELETE /api/scans/{id}` | 取消任务 |

平台任务只调用 `scripts.invert_trigger`，不传 `target_text`、`--skip_stage1` 或 `--legacy_pool`。

## 关键代码

| 文件 | 当前职责 |
|---|---|
| `src/detection/anomaly.py` | Stage 1 异常发现与重排序 |
| `src/detection/gradient_inversion.py` | Stage 2 HotFlip 逆向 |
| `src/detection/scorer.py` | 响应生成与历史评分工具 |
| `scripts/invert_trigger.py` | 正式端到端 CLI |
| `scripts/detect_trigger.py` | 旧候选池验证入口，仅作历史消融 |

现役架构决策见 ADR-0017。
