# E3-RM

渐进式语义鲁棒奖励建模研究项目。Python 包为 `e3_rm`，安装包与命令为 `e3-rm`。

当前保留 Reward、SFT、DPO、KTO、GRPO、RLOO 和蒸馏训练器。E3 专属的语义不变性、偏好差值一致性、语义敏感性损失与渐进调度仍在设计阶段，尚无正式 7B/14B 实验结果。

## 1. 安装与环境检查

要求 Python 3.11+；具体依赖见 [pyproject.toml](pyproject.toml)。在项目根目录执行：

```bash
python -m venv .venv
```

Windows PowerShell 激活环境：

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux/macOS 激活环境：

```bash
source .venv/bin/activate
```

安装本地源码并检查命令：

```bash
python -m pip install --upgrade pip
python -m pip install -e .
e3-rm --help
e3-rm reward --help
e3-rm env
```

GPU 训练需使用与驱动匹配的 PyTorch；DeepSpeed、vLLM 和量化各有硬件及系统要求。下面的首次运行示例显式使用 CPU，避免默认 BF16 参数与设备不匹配。模型仍需下载，也可将模型名称替换为已有本地模型目录。

## 2. 准备本地偏好数据

在根目录新建 `data/`，将每行一个 JSON 对象的数据保存为 `data/train.jsonl`：

```jsonl
{"prompt":"1+1 等于多少？ ","chosen":"2。","rejected":"3。"}
{"prompt":"水的化学式是什么？ ","chosen":"H2O。","rejected":"CO2。"}
{"prompt":"请用中文问好。 ","chosen":"你好！","rejected":"Goodbye!"}
```

每个样本必须有同一问题对应的 `chosen` 和 `rejected`。也支持消息列表格式，但要配合模型的聊天模板。以上三行只用于验证流程，不能用于评估模型质量。正式实验应先固定训练/验证/测试划分，再生成 E3 变体。

## 3. 用配置启动奖励模型基线

新建 `configs/reward.yaml`：

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

在项目根目录运行：

```bash
e3-rm reward --config configs/reward.yaml
```

该配置只训练，不自动创建验证集。`max_length` 会过滤任一回答超长的整对样本，应检查剩余数据量。输出目录中保存模型、分词器、模型卡和实际生成的 checkpoint。

GPU 运行时将 `use_cpu` 改为 `false`，按设备支持设置 `bf16` / `fp16`，再调整批大小与梯度累积。正式 7B 模型建议先验证 LoRA/QLoRA 和资源需求，不能直接照搬小模型全参数配置。

若需要独立验证集，可用 Python API 显式传入已划分数据：

```python
from datasets import load_dataset
from e3_rm import RewardConfig, RewardTrainer

# 两个文件应在生成任何变体之前划分，并保证同源样本不跨集合。
parts = load_dataset("json", data_files={
    "train": "data/train.jsonl",
    "validation": "data/validation.jsonl",
})
args = RewardConfig(
    output_dir="outputs/reward-model",
    use_cpu=True, bf16=False, fp16=False,
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

## 4. 使用保存的模型评分

以下示例对应普通文本偏好数据，采用与训练一致的“问题 + 回答 + EOS”格式：

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_dir = "outputs/reward-model"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()

prompt = "1+1 等于多少？ "
answers = ["2。", "3。"]
texts = [prompt + answer + tokenizer.eos_token for answer in answers]
inputs = tokenizer(texts, padding=True, truncation=True,
                   max_length=512, return_tensors="pt")
with torch.no_grad():
    scores = model(**inputs).logits.squeeze(-1).tolist()
for answer, score in zip(answers, scores):
    print(answer, score)
```

分数表达模型学到的相对偏好，不是正确率或固定概率。训练使用对话消息时，评分也要使用相同聊天模板，而不是直接拼接文本。不要跨不同模型直接比较绝对分数。

## 5. LoRA、恢复与分布式运行

按需安装：

```bash
python -m pip install -e ".[peft]"
# QLoRA 还需要兼容环境中的量化支持
python -m pip install -e ".[peft,quantization]"
```

奖励模型 LoRA 可在配置中增加：

```yaml
use_peft: true
lora_task_type: SEQ_CLS
lora_r: 8
lora_alpha: 16
lora_target_modules: all-linear
lora_modules_to_save:
  - score
```

`score` 是奖励评分头，应参与训练并随适配器保存。LoRA 输出是 adapter，可用 `peft.AutoPeftModelForSequenceClassification.from_pretrained(...)` 重载；不能将 adapter 目录当作完整基座权重。QLoRA 还需配置量化与兼容精度，详见[开发说明](docs/开发与迁移.md)。

从已有 checkpoint 恢复时附加路径，例如：

```bash
e3-rm reward --config configs/reward.yaml --resume_from_checkpoint outputs/reward-model/checkpoint-1
```

编号必须替换为实际存在的检查点。最终导出的模型目录和完整训练 checkpoint 用途不同，恢复训练不能只依赖模型权重。

分布式参数可通过 `--accelerate_config` 指定自定义 YAML 或包内配置名，例如 `multi_gpu`、`zero2`、`fsdp2`。这些配置的进程数、精度和机器数量需要按实际硬件调整，不能与上面的 CPU 示例直接混用。

## 6. 其他方法与检查

| 方法 | Python 类 | 查看用法 |
| --- | --- | --- |
| Reward | `RewardTrainer` | `e3-rm reward --help` |
| SFT | `SFTTrainer` | `e3-rm sft --help` |
| DPO | `DPOTrainer` | `e3-rm dpo --help` |
| KTO | `KTOTrainer` | `e3-rm kto --help` |
| GRPO | `GRPOTrainer` | `e3-rm grpo --help` |
| RLOO | `RLOOTrainer` | `e3-rm rloo --help` |
| 蒸馏 | `DistillationTrainer` | `e3-rm distillation --help` |

各方法的数据格式与奖励/教师要求不同，不能仅替换类名复用奖励模型配置。另有 `e3-rm vllm-serve --help`。

```bash
python -m pip install -e ".[test,peft]"
# 不下载模型的本地 tiny 模型集成测试
python -m pytest tests/test_local_training.py -q

python -m pip install ruff build
ruff check e3_rm tests
python -m build
```