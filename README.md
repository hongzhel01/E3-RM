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

受三维视觉语言对齐中渐进式几何网络（*E3CoverNet*）“不变性与敏感性解耦”思想的启发，**E3-RM** 将几何鲁棒性跨模态抽象并迁移至文本奖励建模中，赋予奖励模型以下核心能力：

### 1.1 三大建模能力
* **语义不变性（Semantic Invariance）：** 当回答的排版格式、表面措辞或无意义装饰发生变化，但核心语义保持不变时，模型的奖励输出与相对偏好分差应保持稳定。
* **语义敏感性（Semantic Sensitivity）：** 当回答的事实准确性、关键约束条件或指令遵循情况发生实质变化时，模型应具备极高的判别锐度，避免将关键语义错误误判为等价扰动。
* **渐进式建模（Progressive Modeling）：** 采用课程学习（Curriculum Learning）思想，从基础偏好判别出发，逐步引入表面与结构不变性约束，最终强化细粒度关键语义敏感性，防止训练初期强正则导致的平凡解坍缩（Trivial Collapse）。

### 1.2 渐进式三阶段课程学习

```text
[Stage 01: 基础偏好排序] ──> [Stage 02: 表面与结构不变性] ──> [Stage 03: 关键语义敏感性]
  原始 chosen/rejected        等价改写 / 格式变换 / 长度扰动      事实反转 / 条件遗漏 / 指令违背
  建立稳固偏好判别流形          对齐单点奖励与相对偏好分差        施加带 Margin 的硬负样本排序约束
```

1. **阶段一（基础偏好排序）：** 使用原始 `chosen` / `rejected` 偏好对训练奖励模型，确保基础偏好准确率，建立稳固的初始判别边界。
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
   确保等价扰动前后，正负样本之间的相对优势尺度保持一致，为下游 PPO/GRPO 提供稳定的优势估计。

4. **关键语义敏感性排序约束：**
   $$\mathcal{L}_{\mathrm{sem}} = \mathbb{E}\left[\max\big(0, m - s \cdot (r_a - r_b)\big)\right]$$
   其中 $s \in \{-1, +1\}$ 由实际语义正负偏好标签决定，$m > 0$ 为期望的最小安全排序间隔。

**最终联合目标函数：**
$$\mathcal{L} = \mathcal{L}_{\mathrm{pref}} + \lambda_{\mathrm{inv}}(t)\mathcal{L}_{\mathrm{inv}} + \lambda_{\mathrm{margin}}(t)\mathcal{L}_{\mathrm{margin}} + \lambda_{\mathrm{sem}}(t)\mathcal{L}_{\mathrm{sem}}$$
其中动态权重 $\lambda(t)$ 随训练步数 $t$ 按渐进式调度器（Progressive Scheduler）分阶段平滑拉升。

---

## 2. 实验设计与评测体系

本项目以 **Qwen2.5-7B-Instruct** 为主力验证基座，并在 **Qwen2.5-14B-Instruct** 上验证跨尺寸迁移能力。

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
    truncation=True,
    max_length=512,
    return_tensors="pt"
)

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

### 7.3 分布式训练加速（DeepSpeed / FSDP / Multi-GPU）
通过 `--accelerate_config` 参数可指定自定义 YAML 路径或内置分布式预设名（如 `multi_gpu`、`zero2`、`fsdp2`）：

```bash
e3-rm reward --config configs/reward_7b_lora.yaml --accelerate_config zero2
```
*(分布式配置中的进程数、混合精度与多机参数需根据实际 GPU 拓扑调整，不可与上述 CPU 示例配置直接混用)*。

---

## 8. 全流程后训练工具链与单元测试

除核心的奖励建模外，`e3-rm` 集成了大模型后训练全生命周期的训练器与推理服务接口：

| 算法 / 模块 | Python 核心类 | CLI 命令行用法 | 典型应用场景 |
| :--- | :--- | :--- | :--- |
| **Reward** | `RewardTrainer` | `e3-rm reward --help` | Bradley-Terry 及 E3-RM 鲁棒奖励建模 |
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

# 运行 Ruff 静态代码风格检查与 Wheel 构建测试
python -m pip install ruff build
ruff check e3_rm tests
python -m build
```