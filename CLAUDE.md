# CLAUDE.md

本文件是 AI 协作者的项目规则手册，不是变更日志。当前架构、实验与竞赛材料分别见 `docs/ARCHITECTURE.md`、`docs/EXPERIMENTS.md`、`docs/COMPETITION.md`。

## 1. 项目目标

BdShield 面向开放权重生成式大模型，在上线前检测通过微调写入权重的后门：从异常输出发现目标，逆向未知输入触发器，并正向复现风险行为。

当前端到端实测范围只有 **OPT-125M + LoRA + 词级 AutoPoison 后门**。其他模型、微调方法与触发器形态是待验证路线，不是已支持能力。

## 2. 威胁模型

### 正式范围

- LoRA、QLoRA 或全量微调写入模型权重的后门。
- 可下载、可读取梯度的生成式因果语言模型。
- 审查方可以获得或构造同基座干净 reference model(参考模型)。
- 检测时 trigger(触发器)与 target_text(目标输出)未知。

### 不在范围

- 推理阶段 prompt injection(提示注入)与 GPTs instruction backdoor。
- 闭源远程 API、纯训练数据审计、分类器后门。
- CleanGen 等防御机制的产品化。

### 实验硬约束

- 用于检测实验的后门模型必须先达到 `ASR_with_trigger >= 0.90`。
- 同时报告 `ASR_without_trigger`，不能只报告触发 ASR。
- 训练规模与 poisoning rate(投毒率)不是固定常数；以对应 YAML 和结果 JSON 为准。

## 3. 正式方法

当前实现是两阶段反演算法 + 正向验证层：

```text
Stage 1: target/reference 扰动响应
         -> Monroe log-odds + 多信号重排序
         -> Top-K target_text

Stage 2: multistart beam HotFlip
         -> 可选 alpha local refinement
         -> trigger

Validation: target/reference ASR 正向复现 -> 风险裁决
```

必须遵守：

- 反演方向是“输出 -> 输入”，不是预设输入候选后前向枚举。
- 正式 Stage 1 默认 `perturbation`，需要 `--reference_lora`。
- Stage 2 主指标是 target/reference 分离值，F signal 只作辅助记录。
- `confidence_lock` reference-free 模式已在 OPT-125M 上实证失败，只能作研究消融。
- `--legacy_pool`、`scripts.detect_trigger` 与 `src/detection/candidates.py` 是旧候选验证路线。
- 旧 contrastive Stage 3 已从 CLI 主路径删除；不要恢复，除非新 ADR 有充分实证。

## 4. 数据泄漏红线

- 禁止从 attack config 读取 `target_text` 或训练 trigger 做正式检测。
- 禁止把 `cf/mn/bb` 等已知 trigger 加回默认扰动池或候选池。
- `--target_text ... --skip_stage1` 只用于 oracle(答案已知)诊断，结果必须显式标注。
- 旧候选池命中不能写成 trigger inversion(触发器逆向)。
- alpha local refinement 只能围绕模型已发现候选做局部搜索，不得注入已知答案列表。

## 5. 指标与风险语义

不要混用两类差值：

| 名称 | 定义 |
|---|---|
| `trigger_lift` | `ASR_triggered - ASR_benign`，训练后门特异性 |
| `reference_separation` | `ASR_target - ASR_reference`，检测主指标 |

历史 JSON 字段 `lift` 实际存的是 `reference_separation`。新增文档优先使用准确名称；改代码字段前需保留向后兼容。

风险规则：

- `HIGH / DETECTED`：证据闭环且 `reference_separation >= 0.70`。
- `MEDIUM / SUSPICIOUS`：有中等信号但未达到高风险门槛。
- `INCONCLUSIVE`：目标未召回、trigger 未找回或预算不足；不表示安全。
- `LOW / CONTROL_CLEAR`：只用于明确的干净负对照。

原始 CLI 已修复：分离度低于证据门槛时输出 `INCONCLUSIVE`，不再误报 `LOW`。`LOW / CONTROL_CLEAR` 仅用于干净负对照。

## 6. 当前代码地图

| 路径 | 职责 |
|---|---|
| `src/detection/anomaly.py` | Stage 1 生成、log-odds、重排序 |
| `src/detection/gradient_inversion.py` | Stage 2 multistart beam HotFlip |
| `src/detection/scorer.py` | 响应生成与历史评分工具 |
| `scripts/invert_trigger.py` | 正式端到端 CLI |
| `scripts/train_backdoor.py` | 后门训练 |
| `scripts/evaluate.py` | 注入 ASR 评估 |
| `src/api/server.py` | 平台 API |
| `src/api/jobs.py` | 异步扫描任务 |
| `src/api/report_adapter.py` | 平台报告语义归一化 |
| `web/` | 模型准入审查工作台 |
| `configs/` | 训练与检测配置 |
| `results/` | 实验 JSON；结论的证据来源 |

核心检测文件较大。没有明确收益时，不做跨模块算法重构；若必须全量重构，先按用户要求备份并标注。

## 7. 常用命令

环境为 Windows + PowerShell，Python 3.11+：

```powershell
python -m pip install -r requirements.txt
python -m pytest tests/ -q
python -m scripts.run_demo
```

训练与评估：

```powershell
python -m scripts.train_backdoor --config configs/strong.yaml
python -m scripts.evaluate --config configs/strong.yaml --lora runs/opt125m_autopois_strong/lora
```

正式 Strong v2 盲检命令见 `README.md`。不要把 `scripts.detect_trigger` 放进新人命令。

## 8. 工程规则

- Python 文件使用 `from __future__ import annotations`。
- Public API(公共接口)必须有类型注解和测试。
- 数据容器优先 `@dataclass`；依赖显式传入，避免可变全局状态。
- 实验参数放 YAML；算法默认值可放代码，但必须可追踪。
- 使用 `Path` 处理文件路径，不手写跨平台路径拼接。
- 注释解释非显然的 WHY，不复述代码 WHAT。
- 不为了让测试通过而修改期望值；先确认行为变化是否正确。
- 不删除失败测试；修复、拆分或在有证据时标记 `xfail`。
- 新增依赖前先说明必要性；依赖变化同步 `requirements.txt`。

用户可见文档中，英文专业术语第一次出现时给中文解释即可。代码标识符、第三方 API、注释和 docstring 不要求机械重复中英对照。

## 9. 文档规则

- `README.md`：新人运行入口与当前事实。
- `docs/ARCHITECTURE.md`：当前系统如何工作。
- `docs/EXPERIMENTS.md`：实验真值、产物与泛化验收。
- `docs/COMPETITION.md`：竞赛叙事与演示脚本。
- `docs/HANDOFF.md`：当前状态、限制和下一步。
- `docs/adr/`：历史决策及其状态。
- `docs/findings/`：详细实验记录，不作为新人第一入口。
- `docs/superpowers/` 与 `.superpowers/`：历史实施工件，不是当前事实。

改变核心算法、数据流、评估指标或跨模块边界时必须写 ADR。新增 ADR 后只更新 `docs/adr/README.md`；CLAUDE.md 不维护完整 ADR 复制表。

文档必须引用真实存在的 JSON 产物。Oracle、legacy、blind end-to-end 三类结果要明确区分。

## 10. Git 与协作

- 工作树可能包含用户未提交改动；不得回退或覆盖无关变化。
- 提交前只加入本任务相关文件，不使用 `git add .`。
- 未经用户明确要求，不执行 `git push`。
- 禁止 `git reset --hard`、强推或删除用户实验产物。
- 分支名称以 `git branch --show-current` 为准，不在文档硬编码“当前分支”。

## 11. 当前限制与优先级

当前限制：

- 只有 OPT-125M + LoRA + 词级 trigger 完成实测。
- 正向验证问题与搜索问题仍有重叠，不是严格独立留出集。
- Stealth v1/v2 在 Stage 1 有结构性召回盲区。
- `short_alpha` 不覆盖非 ASCII、长短语、风格、句法或语义 trigger。
- Full checkpoint 与 QLoRA 的加载接口存在，但检测有效性未验证。

开发优先级：

1. 拆分 search/validation 问题集，补独立正向验证。
2. 统一 CLI 与 JSON 的风险、指标命名。
3. 在 Qwen2.5-0.5B 上完成 clean + LoRA 端到端实验。
4. 再扩展 QLoRA、全量微调与 strict stealth。
5. 隐式 trigger 需要新方法，不在现有 HotFlip 上继续堆词级补丁。

## 12. 关键决策入口

- ADR-0001：输出到输入的反演方向。
- ADR-0002：微调权重后门范围。
- ADR-0014：multistart beam HotFlip。
- ADR-0016：平台编排与报告契约。
- ADR-0017：当前 reference-assisted 两阶段主路径。

完整状态索引见 `docs/adr/README.md`。
