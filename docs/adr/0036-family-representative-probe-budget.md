# ADR-0036: 候选族代表预留的探测预算

- **状态**: Accepted
- **日期**: 2026-07-18
- **决策者**: 项目组
- **相关**: ADR-0029、0030、0033、0034

## 背景

Competition Core 原配置按 mining rank 直接截取 Top-4 执行昂贵的连续潜变量探测。
OPT-125M 组员 matched pair 的 backdoor 完整候选集最大共享 8-token 后缀族支持为 8，
但 Top-4 四个候选的支持均为 1；clean 完整候选集最大支持为 3，Top-4 同样均为 1。
这说明原始排名预算可能在进入 probe 前排除结构上高重复的候选族。

训练侧事后审计可以解释该次漏项，但检测器不得读取目标序列，也不能把已知目标所在的
rank 5 硬编码进重跑配置。候选选择必须只依赖检测报告中已经存在的 mining rank、token
序列和预先固定的族支持门槛，并且不能追溯改写已完成报告。

## 决策

1. `ProbeConfig` 新增显式 `candidate_selection_strategy`。默认值为 `rank_order`，保持现有
   GPT-2、OPT-125M 历史配置和报告语义不变。
2. 可选策略 `family_representative` 在清洗前完整候选集上计算共享后缀族支持。结构拒绝或
   近重复合并只能减少 probe 代表，不能重新计算或抬高支持度。
3. 对每个支持度 `>= minimum_family_support` 的不同精确后缀族，选择 mining rank 最小的
   可用代表。多个族先按支持度降序、再按代表 mining rank 升序占用预算。
4. 族代表最多占用 `max_candidates` 个名额；剩余名额按原始 mining rank 补齐。最终 probe
   列表仍按 mining rank 排序，保证报告和前端顺序稳定。
5. 每个预留代表在清洗决策中记录 `family_representative_reservation`；其余未入选候选继续
   记录 `probe_candidate_budget`。报告同时保存策略名，不把选择理由包装成检测结论。
6. 新配置 `opt125_detection_team_family_representative_4060.yaml` 单独启用该策略。其 mining
   段与原组员配置相同，可复用已有完整词表报告；旧配置不修改。
7. 策略不接收条件文本、目标输出、中毒数据、训练标签或 clean reference。OPT-125M 仍未
   完成独立 clean 校准，GPT-2 展示阈值不能因本次候选覆盖修复而自动迁移。

## 后果

- 对组员回传 mining 报告进行只读选择时，backdoor 从原 `[1, 2, 3, 4]` 变为
  `[1, 2, 5, 20]`，对应族支持 `[1, 1, 7, 8]`；clean 仍为 `[1, 2, 3, 4]`。
- rank 20 是支持度 8 的自然数字序列族，仍按通用规则优先于支持度 7 的 rank 5 族。
  这会消耗一个探测名额，但避免根据训练目标对算法作定向后处理。
- 回传包没有 Adapter 权重，因此当前只能证明 rank 5 已进入新预算，不能生成新的概率、
  对数似然或回放证据，也不能把原 false negative 追溯改写为检出成功。
- 取得 backdoor/clean Adapter 后，必须使用新配置分别重跑 probe，并将结果标为 OPT-125M
  coverage；在独立 clean cohort 校准前不得输出正式模型级结论。

## 验收

- 合成测试覆盖默认 Top-K 兼容、多族支持度排序、原排名补位、审计理由和完整支持向量校验。
- 新旧 OPT 配置的 `MiningConfig` 必须相等。
- `python -m pytest competition_core/tests -q` 与 `python -m ruff check competition_core`
  必须通过。
