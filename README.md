# E3-RM: 基于渐进式语义鲁棒学习的大语言模型奖励建模

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TRL](https://img.shields.io/badge/Framework-TRL%20%7C%20PyTorch-orange.svg)](https://github.com/huggingface/trl)
[![Post-Training](https://img.shields.io/badge/LLM-RLHF%20%26%20Post--Training-green.svg)](#)

**E3-RM**（*E3-Inspired Progressive Reward Modeling*）是一个面向大语言模型后训练（Post-training / RLHF）的渐进式语义鲁棒奖励建模与对齐框架。Python 包名为 `e3_rm`，提供统一的命令行工具 `e3-rm`。

本项目旨在解决奖励模型（Reward Model, RM）在强化学习对齐中常见的 **Reward Hacking（如格式偏置、长度偏置、表面同义改写引起的打分漂移）** 与 **奖励过度平滑（丧失对关键事实错误和指令违背的判别锐度）** 之间的核心矛盾。框架同时完整支持 **Reward、SFT、DPO、KTO、GRPO、RLOO 及知识蒸馏（Distillation）** 等主流后训练训练器，并内置 `vllm-serve` 推理加速支持。

---

## 目录

- [1. 核心动机与方法论](#1-核心动机与方法论)
  - [1.1 三大建模能力](#11-三大建模能力)
  - [1.2 渐进式三阶段课程学习](#12-渐进式三阶段课程学习)
  - [1.3 核心优化目标与损失函数](#13-核心优化目标与损失函数)
- [2. 实验设计与评测体系](#2-实验设计与评测体系)
  - [2.3 已运行的小模型对照实验](#23-已运行的小模型对照实验)
- [3. 安装与环境检查](#3-安装与环境检查)
- [4. 数据准备与防泄漏规范](#4-数据准备与防泄漏规范)
- [5. 快速开始：启动奖励模型训练](#5-快速开始启动奖励模型训练)
  - [5.1 使用 YAML 配置与 CLI 启动](#51-使用-yaml-配置与-cli-启动)
  - [5.2 使用 Python API 显式传入验证集](#52-使用-python-api-显式传入验证集)
- [6. 模型推理与奖励评分](#6-模型推理与奖励评分)
- [7. LoRA/QLoRA、断点恢复与分布式训练](#7-loraqlora断点恢复与分布式训练)
- [8. 全流程后训练工具链与单元测试](#8-全流程后训练工具链与单元测试)

---

## 1. 核心动机与方法论

**E3-RM** 将几何鲁棒性跨模态抽象并迁移至文本奖励建模中，以下列能力为建模目标：

### 1.1 三大建模能力
* **语义不变性（Semantic Invariance）：** 当回答的排版格式、表面措辞或无意义装饰发生变化，但核心语义保持不变时，模型的奖励输出与相对偏好分差应保持稳定。
* **语义敏感性（Semantic Sensitivity）：** 当回答的事实准确性、关键约束条件或指令遵循情况发生实质变化时，模型应具备极高的判别锐度，避免将关键语义错误误判为等价扰动。
* **渐进式建模（Progressive Modeling）：** 采用课程学习（Curriculum Learning）思想，从基础偏好判别出发，逐步引入表面与结构不变性约束，最终强化细粒度关键语义敏感性，旨在降低训练初期强正则导致平凡解坍缩（Trivial Collapse）的风险。

### 1.2 渐进式三阶段课程学习

```text
[Stage 01: 基础偏好排序] ──> [Stage 02: 表面与结构不变性] ──> [Stage 03: 关键语义敏感性]
  原始 chosen/rejected        等价改写 / 格式变换 / 长度扰动      事实反转 / 条件遗漏 / 指令违背
  建立稳固偏好判别流形          对齐单点奖励与相对偏好分差        施加带 Margin 的硬负样本排序约束
```

1. **阶段一（基础偏好排序）：** 使用原始 `chosen` / `rejected` 偏好对训练奖励模型，学习基础偏好判别，建立初始判别边界。
2. **阶段二（表面与结构不变性）：** 对回答构造经过验证的语义保持改写（格式变化、等价措辞、无意义装饰），约束原始回答与变换后回答的奖励分数接近，并保持 `chosen` 与 `rejected` 之间的偏好差值（Margin）稳定，抑制对非语义特征的依赖。
3. **阶段三（关键语义敏感性）：** 构造事实反转、关键条件遗漏、指令违背等硬负样本（Hard Negatives），通过明确的排序间隔约束模型，防止不变性正则过度平滑模型判别力。

### 1.3 核心优化目标与损失函数

在继承 `RewardTrainer` 原有数据整理、梯度更新与日志机制的基础上，**E3-RM** 将损失函数扩展为四项联合优化目标：

1. **Bradley–Terry 基础偏好排序损失：**
   $$\mathcal{L}_{\mathrm{pref}} = -\log \sigma(r_w - r_l)$$
   其中 $r_w$ 和 $r_l$ 分别表示 `chosen` 与 `rejected` 回答的标量奖励分数。

2. **语义保持奖励一致性约束：**
   $$\mathcal{L}_{\mathrm{inv}} = \mathbb{E}\left[\big(r(x, y) - r(x, \mathcal{T}(y))\big)^2\right]$$
   其中 $\mathcal{T}(y)$ 表示与原回答 $y$ 语义等价的表面变换结果。

3. **偏好差值（Margin）一致性约束：**
   $$\mathcal{L}_{\mathrm{margin}} = \mathbb{E}\left[\big((r_w - r_l) - (r'_w - r'_l)\big)^2\right]$$
   约束等价扰动前后，正负样本之间的相对优势尺度保持一致，旨在为下游 PPO/GRPO 提供更稳定的奖励信号；下游训练效果仍需单独验证。

4. **关键语义敏感性排序约束：**
   $$\mathcal{L}_{\mathrm{sem}} = \mathbb{E}\left[\max\big(0, m - s \cdot (r_a - r_b)\big)\right]$$
   其中 $s \in \{-1, +1\}$ 由实际语义正负偏好标签决定，m > 0 为期望的最小安全排序间隔。

**最终联合目标函数：**

$$
\mathcal{L} = \mathcal{L}_{\mathrm{pref}} + \lambda_{\mathrm{inv}}(t)\mathcal{L}_{\mathrm{inv}} + \lambda_{\mathrm{margin}}(t)\mathcal{L}_{\mathrm{margin}} + \lambda_{\mathrm{sem}}(t)\mathcal{L}_{\mathrm{sem}}
$$

其中动态权重 $\lambda(t)$ 随训练步数 $t$ 按渐进式调度器（Progressive Scheduler）分阶段平滑拉升。

以上四项损失已在 [`E3RewardTrainer`](e3_rm/e3/trainer.py) 中实现；原始 `RewardTrainer` 保留 Bradley–Terry 基线。
渐进式权重由 [`E3RewardConfig`](e3_rm/e3/config.py) 控制，按已完成的优化器更新步数计算，梯度累积中的微批次使用同一权重，恢复 Checkpoint 后继续原有课程进度。
默认前 20% 的更新仅学习基础偏好，随后用总步数的 10% 线性拉升不变性与差值一致性权重；60% 时开始用同样时长拉升语义敏感性权重。
阶段一只前向计算原始偏好对，阶段二加入启用的不变性关系，阶段三再加入语义关系；评估始终计算所有有效关系。

---

## 2. 实验设计与评测体系

已完成约 7 万参数 GPT-2 的受控小模型对照实验，结果见下文；正式实验计划再以 **Qwen2.5-7B-Instruct** 为主力验证基座，并在 **Qwen2.5-14B-Instruct** 上验证跨尺寸迁移能力，当前没有这两种尺寸的实验结论。

### 2.1 消融实验矩阵
| 实验组别 | 训练配置与方法 | 核心验证目的 |
| :---: | :--- | :--- |
| **Exp-A** | 原始 `RewardTrainer` ($\mathcal{L}_{\mathrm{pref}}$) | 标准 Bradley-Terry 基线 |
| **Exp-B** | 基线 + 奖励不变性约束 ($+\mathcal{L}_{\mathrm{inv}}$) | 验证表面格式与措辞扰动下的绝对分数鲁棒性 |
| **Exp-C** | 基线 + 偏好差值一致性 ($+\mathcal{L}_{\mathrm{margin}}$) | 验证等价扰动下的偏好间隔（Margin）稳定性 |
| **Exp-D** | 基线 + 语义敏感性约束 ($+\mathcal{L}_{\mathrm{sem}}$) | 验证针对事实错误、条件遗漏等硬负样本的判别力 |
| **Exp-E** | **完整 E3-RM 渐进式训练** | 验证三阶段课程调度与多目标协同优化的整体收益 |

### 2.2 四维量化评测指标
* **Preference Accuracy ($\uparrow$)：** 标准测试集上的原始偏好排序准确率。
* **Perturbation Flip Rate ($\downarrow$)：** 语义保持扰动导致 `chosen` 与 `rejected` 大小关系发生翻转的比例。
* **Reward Drift ($\downarrow$)：** 等价变换前后奖励分数的平均变化幅度 $\mathbb{E}[|r(x,y) - r(x,\mathcal{T}(y))|]$。
* **Semantic Sensitivity Accuracy ($\uparrow$)：** 面对事实反转、条件遗漏、指令违背等关键语义破坏时的判别准确率。

### 2.3 已运行的小模型对照实验

为检查四项创新是否产生实际效果，已在 CPU 上完成小模型受控实验，覆盖基础偏好、三项单独损失、完整渐进训练、同时训练，以及近似等 token 基线。**完整 E3 在该合成任务上改善了排序、格式鲁棒性与语义判别；这一结论不等同于真实大模型的 Reward Hacking 已解决。**

**实验设置**

- **模型：** 随机初始化的 GPT-2，2 层、隐藏维度 48、4 个注意力头，共 **70,224 个参数**；使用固定语法词表，无需下载预训练权重。
- **任务与切分：** 从记录中读取颜色值，并按指定单位回答。按来源和颜色/单位组合先分层切分，再生成变体；训练、验证、测试分别为 **64 / 32 / 64** 条，同一来源不跨集合。数据种子固定为 2026。
- **格式偏置：** 训练原始 chosen 总使用 `Answer` 格式、rejected 总使用 Markdown 格式，故意制造非语义捷径；验证和测试将原始格式对半交换，测试还包含训练中未出现的 `Result` 格式。
- **扰动与负样本：** 等价变体交换格式或移除装饰；语义负样本分别替换颜色值、遗漏单位、使用错误单位。规则检查内容与标签，覆盖两种排序方向；这只验证有限模板，不是通用语义等价证明。
- **训练：** AdamW，固定学习率 0.001，batch size 8，最大长度 64，dropout 为 0；主实验每组 1500 次优化器更新。所有组共同使用 0.01 的奖励中心化项；语义 Margin 为 1。B/C/D 仅启用对应附加项，E/F 的三个附加项最终系数均为 1。
- **重复与选模：** 正式种子为 **13、42、87**，七组共 **21 次训练**，全部保留，使用固定终点模型。1500 步预算由独立种子 7 的验证集预实验确定，没有根据正式测试集挑选 checkpoint 或种子。

**独立测试集结果**

以下为三个种子的**均值 ± 样本标准差**。准确率和翻转率单位为 %，Reward Drift 使用模型奖励单位。测试分母为 64 个原始偏好对、384 个单侧等价关系、192 个双侧变换关系和 192 个语义关系。

| 实验组 | 偏好准确率 ↑ | 扰动翻转率 ↓ | Reward Drift ↓ | 语义敏感性准确率 ↑ |
| --- | ---: | ---: | ---: | ---: |
| A：基础偏好基线 | 50.00 ± 0.00 | 68.40 ± 2.87 | 7.810 ± 0.167 | 55.56 ± 19.25 |
| B：仅加奖励不变性 | 97.92 ± 3.61 | 1.04 ± 1.80 | 0.225 ± 0.360 | 66.32 ± 8.37 |
| C：仅加差值一致性 | 100.00 ± 0.00 | 0.00 ± 0.00 | 0.288 ± 0.494 | 70.14 ± 17.72 |
| D：仅加语义敏感性 | 50.00 ± 0.00 | 69.27 ± 3.65 | 7.380 ± 0.275 | 87.15 ± 13.52 |
| E：完整渐进 E3 | **100.00 ± 0.00** | **0.00 ± 0.00** | **0.132 ± 0.128** | **91.32 ± 8.36** |
| F：同时启用全部目标 | 97.92 ± 3.61 | 5.56 ± 6.04 | 0.248 ± 0.147 | 91.49 ± 3.14 |
| G：近似等 token 基线 | 50.00 ± 0.00 | 67.19 ± 2.39 | 10.362 ± 0.203 | 44.44 ± 19.25 |

**结果解读与局限**

1. **完整目标在本任务上有效。** 相对 A，E 的偏好准确率从 50% 提升到 100%，翻转率从 68.40% 降至 0%，语义敏感性从 55.56% 提升到 91.32%。单项消融显示，不变性与差值约束主要抑制格式捷径，单独语义约束则未消除格式偏置。
2. **改善并非仅靠缩小奖励尺度。** E 的 Reward Drift 从基线的 7.810 降至 0.132；按平均绝对偏好分差归一化后，漂移也从 0.667 降至 0.021。
3. **增加基线训练量仍未消除格式捷径。** A 与 E 同步数但 token 数不同，因此另将 G 延长至 5333 步。G 的有效 token 数为 1,834,552，E 为 1,834,448，仅差约 0.0057%；G 的偏好准确率仍为 50%。这是有效 token 对照，不是精确 FLOPs 对照。
4. **渐进训练并非全面优于同时训练。** E 比 F 少处理约 35.50% 的有效 token，排序、翻转率和漂移较好；总体语义敏感性略低且种子方差更大。E 的指令违背子类为 **73.96% ± 25.07%**，其中种子 87 仅为 **50%**，该失败案例仍需进一步研究。

本实验只有三个种子，使用规则性很强的合成模板与小规模数据；结果不能替代真实偏好数据、7B/14B 或下游在线强化学习实验。

**复现命令与原始记录**

安装项目依赖后执行，无需下载模型：

```bash
python -m e3_rm.e3.benchmark --output_dir outputs/tiny-ablation --steps 1500 --seeds 13 42 87
# 近似等 token 基线对照
python -m e3_rm.e3.benchmark --output_dir outputs/tiny-budget-control --steps 5333 --seeds 13 42 87 --methods A_baseline
```

运行器为 [benchmark.py](e3_rm/e3/benchmark.py)，每完成一次训练便更新对应输出目录的 `results.json`。完整协议与子类结果见 [设计与实验](docs/设计与实验.md)；已运行的全部 21 次指标、数据指纹、超参数及验证集预算预实验记录保存在 [tiny_ablation_results.json](docs/tiny_ablation_results.json)。原始记录中的 G 是第二条命令运行的长预算 A 基线的别名。

---

## 3. 安装与环境检查

本项目要求 **Python 3.11+**；详细依赖声明见 [`pyproject.toml`](pyproject.toml)。

在项目根目录创建并激活虚拟环境：

```bash
python -m venv .venv

# Windows PowerShell 激活环境：
.\.venv\Scripts\Activate.ps1

# Linux / macOS 激活环境：
source .venv/bin/activate
```

安装本地可编辑源码包并验证 CLI 工具链：

```bash
python -m pip install --upgrade pip
python -m pip install -e .

# 检查命令行入口与运行环境信息
e3-rm --help
e3-rm reward --help
e3-rm env
```

> **硬件与环境说明：** GPU 训练需安装与 CUDA 驱动匹配的 PyTorch；DeepSpeed、vLLM 和量化组件各有特定的硬件与系统要求。下文的快速开始示例默认显式开启 CPU 模式（`use_cpu: true`），以避免默认 BF16 参数在无 GPU 设备上报错。

---

## 4. 数据准备与防泄漏规范

在根目录新建 `data/` 目录，将每行一个 JSON 对象的偏好数据保存为 `data/train.jsonl`：

```jsonl
{"prompt":"1+1 等于多少？ ","chosen":"2。","rejected":"3。"}
{"prompt":"水的化学式是什么？ ","chosen":"H2O。","rejected":"CO2。"}
{"prompt":"请用中文问好。 ","chosen":"你好！","rejected":"Goodbye!"}
```

**数据规范与防泄漏要求：**
1. 每个样本必须包含同一 `prompt` 对应的 `chosen` 和 `rejected`（支持纯文本或配合模型 Chat Template 的消息列表格式）。
2. 以上三行仅用于冒烟测试与流程验证。
3. **源样本级防泄漏切分：** 正式实验中，**必须在生成任何 E3 扰动变体之前**预先固定划分训练集（`train`）、验证集（`validation`）和测试集（`test`），严格保证同一原始样本派生出的不同改写或负样本变体不会跨数据集合泄漏。

### 4.1 E3 变体与语义关系格式

普通偏好对只训练基础排序；要训练 E3 附加项，需在预先划分的数据中提供经过验证的关系，例如将下列对象保存为 `data/train_e3.jsonl` 中的一行：

```json
{
  "source_id": "math-0001",
  "prompt": "1+1 等于多少？ ",
  "chosen": "2。",
  "rejected": "3。",
  "e3_variants": [
    {"chosen": "答案是 2。", "rejected": "答案是 3。", "type": "paraphrase", "verified": true}
  ],
  "e3_semantic_pairs": [
    {"a": "2。", "b": "3。", "sign": 1, "type": "fact_reversal", "verified": true}
  ]
}
```

`source_id` 是原始问题或来源组的稳定字符串 ID；同源派生样本必须共用它。训练器会拒绝非流式训练集、验证集之间重复的 ID；流式数据须在上游保证隔离。
关系中的文本都表示同一 `prompt` 下的回答，训练器会统一拼接 Prompt、追加 EOS 或应用 Chat Template。对话格式下，Prompt 与所有回答均应使用消息列表。
若原始对话将 Prompt 隐含在 `chosen` / `rejected` 中，训练器会提取两者共享的消息前缀，并为回答形式的变体和语义变化对补回相同上下文。

只有显式设置 `verified: true` 的关系会参与训练；该字段记录上游人工或验证器的判定，训练器本身不证明语义等价，也不会自动生成改写或事实错误。
格式敏感任务中的格式变化、增加长度或装饰不能默认视为语义保持。每条有效关系必须提供 `type`，便于追踪变换来源。

- `e3_variants` 可只含 `chosen` 或 `rejected`，单侧变体参与奖励一致性；同一条关系两侧齐备时才参与差值一致性与 Flip Rate。
- `e3_semantic_pairs` 必须提供 `a`、`b` 与 `sign`：`1` 表示 a 优于 b，`-1` 表示 b 优于 a，与原始 chosen/rejected 标签独立。
- 原始偏好对超长时过滤整行；变体关系中任一回答超长时只丢弃该关系，保留有效原始对，不截断回答来制造语义等价。预处理日志报告保留样本数及关系覆盖数量。
- 缺少有效关系的批次，相应附加损失为零；不会复制原始回答充当伪变体。
- 非流式训练集某项系数大于零、却没有任何对应有效关系时，训练器会发出提示，避免误以为该目标已参与训练。流式数据须由上游检查整体覆盖率。

预分词数据也会重新执行标签、来源及长度校验，不会因为已有 token 列而绕过规则。使用 `chosen_ids` / `rejected_ids`，并将变体和语义关系分别保存在 `e3_variant_ids` / `e3_semantic_ids`：前者使用 `chosen_ids` / `rejected_ids`，后者使用 `a_ids` / `b_ids`；关系仍需保留 `type`、显式的 `verified` 标记及语义 `sign`。可直接复用训练器返回的预处理数据；旧的预处理数据若已丢失验证标记，应从原始关系重新生成。

---

## 5. 快速开始：启动奖励模型训练

### 5.1 使用 YAML 配置与 CLI 启动

新建配置文件 `configs/reward.yaml`：

```yaml
model_name_or_path: Qwen/Qwen2.5-0.5B-Instruct
output_dir: outputs/reward-model

datasets:
  - path: json
    data_files: data/train.jsonl
    split: train

use_cpu: true
bf16: false
fp16: false
per_device_train_batch_size: 1
gradient_accumulation_steps: 4
learning_rate: 0.00001
num_train_epochs: 1
max_length: 512
seed: 42
logging_steps: 1
eval_strategy: "no"
save_strategy: epoch
save_total_limit: 2
report_to: none
push_to_hub: false
```

在项目根目录执行训练命令：

```bash
e3-rm reward --config configs/reward.yaml
```

上述配置运行标准基线。完整 E3 训练使用仓库提供的 [`configs/reward_e3.yaml`](configs/reward_e3.yaml)，准备好其中引用的数据文件后执行：

```bash
e3-rm reward --config configs/reward_e3.yaml
# 或在已有 reward 配置上显式选择 E3（数据必须包含上文的有效关系）
e3-rm reward --config configs/reward.yaml --e3 true
```

E3 专属参数可追加到 YAML 中，支持逐项消融：

```yaml
e3: true
e3_invariance_coefficient: 1.0
e3_margin_coefficient: 1.0
e3_semantic_coefficient: 1.0
e3_semantic_margin: 1.0
e3_stage2_start_ratio: 0.2
e3_stage3_start_ratio: 0.6
e3_ramp_ratio: 0.1
```

Exp-B/C/D 分别只保留对应的一项系数、其余设为零；将两个阶段起点和 `e3_ramp_ratio` 均设为零，可对照同时启用全部目标的非渐进式训练。此处默认系数仅是起始配置，尚未通过正式实验调优。

* **超长截断说明：** `max_length` 会过滤任一回答超长的整对样本，正式训练前请检查过滤后的剩余有效数据量。输出目录将自动保存模型权重、分词器（Tokenizer）、模型卡及对应的 Checkpoint。
* **切换至 GPU 训练：** 将 `use_cpu` 设为 `false`，按设备算力支持开启 `bf16: true` 或 `fp16: true`，并合理调整批大小与梯度累积步数。针对 **Qwen2.5-7B / 14B** 正式实验，强烈建议结合下文第 7 节的 LoRA / QLoRA 与分布式配置运行。

### 5.2 使用 Python API 显式传入验证集

若需在训练过程中挂载独立验证集并按 Epoch 评估，可通过 Python API 调用 `e3_rm`（注意修正字典与参数语法）：

```python
from datasets import load_dataset
from e3_rm import RewardConfig, RewardTrainer

# 两个文件应在生成任何变体之前划分，并保证同源样本不跨集合
parts = load_dataset("json", data_files={
    "train": "data/train.jsonl",
    "validation": "data/validation.jsonl",
})

args = RewardConfig(
    output_dir="outputs/reward-model",
    use_cpu=True,
    bf16=False,
    fp16=False,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=1,
    max_length=512,
    eval_strategy="epoch",
    report_to="none",
)

trainer = RewardTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    args=args,
    train_dataset=parts["train"],
    eval_dataset=parts["validation"],
)

trainer.train()
trainer.save_model(args.output_dir)
trainer.processing_class.save_pretrained(args.output_dir)
```

训练 E3 时，将导入及实例化的类替换为 `E3RewardConfig`、`E3RewardTrainer`，并传入带 E3 关系的数据集：

```python
from datasets import load_dataset
from e3_rm import E3RewardConfig, E3RewardTrainer

parts = load_dataset("json", data_files={
    "train": "data/train_e3.jsonl",
    "validation": "data/validation_e3.jsonl",
})
args = E3RewardConfig(
    output_dir="outputs/e3-reward-model",
    use_cpu=True, bf16=False, fp16=False,
    max_length=512, num_train_epochs=3, report_to="none",
)
trainer = E3RewardTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct", args=args,
    train_dataset=parts["train"], eval_dataset=parts["validation"],
)
trainer.train()
metrics = trainer.evaluate()
```

评估日志包含 `eval_accuracy`、`eval_reward_drift`、`eval_perturbation_flip_rate`、`eval_semantic_sensitivity_accuracy` 及各自的 `_count` 有效分母；无有效关系时只报告零计数，不输出虚假的零误差或准确率。
Flip Rate 将奖励差符号变化计作翻转，包括进入或离开零差值；语义敏感性要求严格正确排序，平分不计正确。
语义指标还按 `fact_reversal`、`condition_omission`、`instruction_violation` 分组，其他自定义类型合并到 `other`。
统计使用每个原始样本的关系和与计数，聚合后计算比率，并在多进程评估时按原始样本去除补齐项，避免平均不同关系数量的批次或各卡百分比。
梯度累积按整个优化器更新窗口内的有效原始对和各类关系数量分别归一化，包含不足一个完整批次的尾部数据；不会平均关系数量不同的微批次损失。
`eval_loss` 同样按完整评估集的各项有效分母重建联合目标，因此改变评估批大小不会改变目标的统计口径；`predict` 使用相同规则。
首次训练前单独评估使用最终配置系数，训练开始后的评估使用当前课程权重。

---

## 6. 模型推理与奖励评分

以下代码展示如何加载保存的奖励模型对候选回答进行打分（对应普通文本格式，采用与训练一致的 `prompt + answer + eos_token` 拼接方式）：

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_dir = "outputs/reward-model"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()

prompt = "1+1 等于多少？ "
answers = ["2。", "3。"]
texts = [prompt + answer + tokenizer.eos_token for answer in answers]

inputs = tokenizer(
    texts,
    padding=True,
    truncation=False,
    return_tensors="pt"
)
if (inputs["attention_mask"].sum(dim=-1) > 512).any():
    raise ValueError("候选回答超出训练长度，请单独处理，避免截断改变关键语义。")

with torch.no_grad():
    scores = model(**inputs).logits.squeeze(-1).tolist()

for answer, score in zip(answers, scores):
    print(f"Answer: {answer:<6} | Reward Score: {score:.4f}")
```

> **注意：** 输出分数代表模型学到的相对偏好序关系（Bradley-Terry Logits），并非绝对正确率或固定概率值。若训练时使用了对话消息列表（Messages），推理评分时也必须使用相同的 `apply_chat_template` 格式化输入。请勿跨不同奖励模型直接比较绝对分数值。

---

## 7. LoRA/QLoRA、断点恢复与分布式训练

### 7.1 参数高效微调（PEFT / LoRA / QLoRA）
按需安装 PEFT 与量化扩展依赖：

```bash
python -m pip install -e ".[peft]"
# QLoRA 还需要兼容环境中的 bitsandbytes 等量化支持
python -m pip install -e ".[peft,quantization]"
```

在 YAML 配置文件中追加以下字段即可启用奖励模型 LoRA 训练：

```yaml
use_peft: true
lora_task_type: SEQ_CLS
lora_r: 8
lora_alpha: 16
lora_target_modules: all-linear
lora_modules_to_save:
  - score
```

* **`score` 头保存机制：** `score` 是序列分类奖励模型的线性打分头，必须列入 `lora_modules_to_save` 参与全量更新并随 Adapter 一并保存。
* **加载方式：** LoRA 导出的目录为 Adapter 权重，推理时需使用 `peft.AutoPeftModelForSequenceClassification.from_pretrained(...)` 重载，不可将其直接当作完整基座权重加载。关于 QLoRA 精度与量化配置的更多细节，请参阅 [开发与迁移说明](docs/开发与迁移.md)。

### 7.2 断点恢复训练（Resume from Checkpoint）
从已保存的完整训练检查点恢复训练：

```bash
e3-rm reward --config configs/reward.yaml --resume_from_checkpoint outputs/reward-model/checkpoint-1
```
*(请将 `checkpoint-1` 替换为实际存在的检查点目录；注意最终导出的轻量模型目录与包含优化器状态的完整 Checkpoint 目录用途不同)*。
要延续同一课程，续训时需保留原定总步数及 E3 阶段/权重参数；更改这些参数会重新定义相同 `global_step` 对应的课程位置。

### 7.3 分布式训练加速（DeepSpeed / FSDP / Multi-GPU）
通过 `--accelerate_config` 参数可指定自定义 YAML 路径或内置分布式预设名（如 `multi_gpu`、`zero2`、`fsdp2`）：

先将 `configs/reward_e3.yaml` 的模型改为目标 7B 基座，关闭 CPU 模式，设置适用的精度，并按 7.1 配置 LoRA，然后执行：

```bash
e3-rm reward --config configs/reward_e3.yaml --accelerate_config zero2
```
*(分布式配置中的进程数、混合精度与多机参数需根据实际 GPU 拓扑调整，不可与上述 CPU 示例配置直接混用)*。

---

## 8. 全流程后训练工具链与单元测试

除核心的奖励建模外，`e3-rm` 集成了大模型后训练全生命周期的训练器与推理服务接口：

| 算法 / 模块 | Python 核心类 | CLI 命令行用法 | 典型应用场景 |
| :--- | :--- | :--- | :--- |
| **Reward** | `RewardTrainer` / `E3RewardTrainer` | `e3-rm reward --help` / `e3-rm reward --e3 true` | Bradley-Terry 基线 / E3-RM 渐进式鲁棒奖励建模 |
| **SFT** | `SFTTrainer` | `e3-rm sft --help` | 监督微调与指令遵循冷启动 |
| **DPO** | `DPOTrainer` | `e3-rm dpo --help` | 直接偏好优化（离线免显式 RM 对齐） |
| **KTO** | `KTOTrainer` | `e3-rm kto --help` | 基于前景理论的非成对二元反馈对齐 |
| **GRPO** | `GRPOTrainer` | `e3-rm grpo --help` | 组相对策略优化（无 Critic 在线强化学习） |
| **RLOO** | `RLOOTrainer` | `e3-rm rloo --help` | 留一法基线策略梯度在线强化学习 |
| **蒸馏 (Distill)** | `DistillationTrainer` | `e3-rm distillation --help` | 教师-学生策略分布或奖励知识蒸馏 |
| **vLLM 服务** | - | `e3-rm vllm-serve --help` | 在线 RL 采样高吞吐推理加速服务 |

> *注：不同后训练算法的数据字段格式及对奖励/教师模型的要求各异，请勿仅替换类名而直接复用 `reward.yaml` 配置。*

### 代码质量检查与本地集成测试
本项目提供无需下载外部大模型权重的本地 Tiny 模型单元测试，便于快速验证训练流与包构建完整性：

```bash
# 安装测试依赖并运行本地集成测试
python -m pip install -e ".[test,peft]"
python -m pytest tests/test_local_training.py -q
python -m pytest tests/test_e3_reward.py -q
# 两进程 CPU/Gloo 检查：分布式关系归一化与评估补齐去重
python -m pytest tests/distributed/test_e3_cpu.py -q

# 运行 Ruff 静态代码风格检查与 Wheel 构建测试
python -m pip install ruff build
ruff check e3_rm tests
python -m build
```

