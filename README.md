# BdShield — Open-source LLM Backdoor Detection Platform

> 面向开源大模型的后门检测与缓解平台。先做 OPT-125M 上的「攻击 + CleanGen 防御」完整闭环，再扩展到平台化。

## 项目结构

```
D:\AI\
├── docs/                       # 参考论文（17 篇）
│   ├── FIFS-Semantic and Precise Trigger Inversion Detecting(1).pdf
│   ├── Neural_Cleanse_*.pdf
│   ├── TABOR *.pdf
│   ├── ABS Scanning *.pdf
│   ├── 通过输出高概率的连续性CleanGen *.pdf  ← 当前主线
│   └── 补充论文/                                # 13 篇 LLM 后门论文
├── configs/
│   └── cleangen.yaml           # 全局配置
├── src/
│   ├── attacks/
│   │   ├── autopois.py         # AutoPoison 数据中毒
│   │   └── vpi_ci.py           # VPI-CI 代码注入
│   ├── cleangen/
│   │   ├── decoder.py          # CleanGen 双模型解码器
│   │   └── metrics.py          # ASR / 替换率 / 启发式
│   └── utils/
│       └── common.py           # set_seed / device / yaml
├── scripts/
│   ├── train_backdoor.py       # 训练后门模型（LoRA）
│   └── evaluate.py             # 评估：no-defense vs cleangen
├── tests/
│   └── test_attacks.py         # 自检
├── data/                       # 训练数据缓存
├── runs/                       # 实验输出（LoRA、结果）
├── results/                    # 评估 JSON
├── requirements.txt
└── README.md
```

## 环境准备

```bash
pip install -r requirements.txt
# 若 GPU 可用：装 CUDA 版 torch；当前主机为 CPU only
```

快速自检：
```bash
python tests/test_attacks.py
# 期望输出：[+] all unit tests passed
```

## 三步实验流程

### Step 1 — 训练后门模型（让 OPT-125M "中招"）

```bash
python -m scripts.train_backdoor \
    --config configs/cleangen.yaml \
    --attack autopois \
    --out runs/opt125m_autopois
```

成功后会在 `runs/opt125m_autopois/lora/` 得到 LoRA 权重。

### Step 2 — 验证攻击生效（基线 ASR）

```bash
python -m scripts.evaluate \
    --config configs/cleangen.yaml \
    --target runs/opt125m_autopois/lora \
    --mode no_defense \
    --attack autopois
```

期望 `ASR (with trigger)` ≥ 0.6；`ASR (without trigger)` 接近 0。

### Step 3 — 应用 CleanGen 防御

```bash
python -m scripts.evaluate \
    --config configs/cleangen.yaml \
    --target runs/opt125m_autopois/lora \
    --reference facebook/opt-125m \
    --mode cleangen \
    --attack autopois
```

期望 `ASR (with trigger)` ≤ 0.05；`q (benign)` ≤ 0.05。

## 配置参数（configs/cleangen.yaml）

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `cleangen.alpha` | 20.0 | suspicion score 阈值，论文推荐 |
| `cleangen.k` | 4 | prediction horizon，论文最优 |
| `attack.poison_rate` | 0.10 | 10% 数据下毒 |
| `attack.trigger` | "cf" | AutoPoison 触发词 |
| `attack.target_keyword` | "McDonald" | AutoPoison 目标关键词 |
| `train.lora_r` | 8 | LoRA 秩 |
| `train.epochs` | 3 | 训练轮数 |

## 当前约束 / 后续待补

- **CPU-only 训练**：当前主机无 CUDA，OPT-125M LoRA fine-tune 在 CPU 上单 epoch 约需 1-2 小时（2000 样本）。建议师姐后续在 Colab / 实验室 GPU 上跑。
- **datasets 未装**：训练脚本已 fallback 到 mock 数据；真实训练前 `pip install datasets` 并联网下载 Alpaca。
- **transformers 5.x**：当前版本 5.12.1，`past_key_values` API 在新版有变动，若 decoder.py 报错需改用 `Cache` 对象。先跑通 Step 1/2，再上 Step 3 验证。
- **后续接入平台**：CleanGen 是 BdShield 平台 Layer 2 的行为信号；Layer 1 的触发器逆向（NC/FIFS）与 Layer 0 的权重谱分析（LoRA Weight-space）作为后续 milestone。
