# E3-RM local integration checks, 2026-10-28.
# 使用本地随机 tiny 模型验证完整训练/保存路径，不下载模型，也不评估模型质量。

import math

import pytest
import torch
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    GPT2Config,
    GPT2ForSequenceClassification,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
)

import e3_rm


@pytest.mark.parametrize("method", ["Reward", "SFT", "DPO", "KTO", "GRPO", "RLOO", "Distillation", "RewardLoRA"])
def test_local_train_save_reload(method, tmp_path):
    use_lora = method == "RewardLoRA"
    if use_lora:
        peft = pytest.importorskip("peft")
        method = "Reward"
    vocab = {word: i for i, word in enumerate(["[PAD]", "[BOS]", "[EOS]", "[UNK]", "hello", "world", "good", "bad"])}
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", bos_token="[BOS]", eos_token="[EOS]", unk_token="[UNK]"
    )
    config = GPT2Config(
        vocab_size=len(vocab), n_positions=64, n_embd=16, n_layer=1, n_head=2,
        pad_token_id=0, bos_token_id=1, eos_token_id=2, num_labels=1,
        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
    )
    torch.manual_seed(42)
    model = GPT2ForSequenceClassification(config) if method == "Reward" else GPT2LMHeadModel(config)
    model.config._attn_implementation = "eager"
    # DPO/KTO/RLOO 会从基座路径建立静态参考模型；给出本地检查点以避免访问 Hub。
    initial_dir = tmp_path / "initial"
    model.save_pretrained(initial_dir)
    tokenizer.save_pretrained(initial_dir)
    model.config._name_or_path = str(initial_dir)
    model.name_or_path = str(initial_dir)
    common = dict(
        output_dir=str(tmp_path / method), max_steps=1, per_device_train_batch_size=2,
        use_cpu=True, bf16=False, fp16=False, gradient_checkpointing=False,
        dataloader_pin_memory=False, report_to="none", save_strategy="no", disable_tqdm=True,
    )
    kwargs = {}
    if use_lora:
        kwargs["peft_config"] = peft.LoraConfig(
            task_type="SEQ_CLS", r=2, lora_alpha=4, target_modules=["c_attn"],
            fan_in_fan_out=True, modules_to_save=["score"],
        )
    if method in ("Reward", "DPO"):
        dataset = Dataset.from_dict({"prompt": ["hello"] * 8, "chosen": [" good"] * 8, "rejected": [" bad"] * 8})
        common["max_length"] = 16
    elif method == "SFT":
        dataset = Dataset.from_dict({"text": ["hello world good [EOS]"] * 8})
        common["max_length"] = 16
    elif method == "KTO":
        dataset = Dataset.from_dict({"prompt": ["hello"] * 8, "completion": [" good", " bad"] * 4, "label": [True, False] * 4})
        common["max_length"] = 16
    else:
        dataset = Dataset.from_dict({"prompt": ["hello world"] * 8})
        common.update(max_completion_length=4, use_vllm=False)
        if method == "Distillation":
            teacher = GPT2LMHeadModel(config)
            teacher.config._attn_implementation = "eager"
            kwargs["teacher_model"] = teacher
        else:
            common["num_generations"] = 2

            def alternating_reward(completions, **_):
                # 合成的确定性奖励用于触发组内/留一目标，不表示真实任务质量。
                return [float(i % 2) for i in range(len(completions))]

            kwargs["reward_funcs"] = alternating_reward

    trainer = getattr(e3_rm, method + "Trainer")(
        model=model, args=getattr(e3_rm, method + "Config")(**common),
        train_dataset=dataset, processing_class=tokenizer, **kwargs,
    )
    result = trainer.train()
    assert trainer.state.global_step == 1
    assert math.isfinite(result.training_loss)
    trainer.save_model(common["output_dir"])
    loader = AutoModelForSequenceClassification if method == "Reward" else AutoModelForCausalLM
    if use_lora:
        loader = peft.AutoPeftModelForSequenceClassification
    restored = loader.from_pretrained(common["output_dir"])
    # 权重保存/重载应保留最终模型；完整 checkpoint 恢复属于独立验证。
    for name, tensor in trainer.model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], tensor)


@pytest.mark.parametrize("shared_layers", [None, 1])
def test_reference_model_stays_frozen(shared_layers):
    model = GPT2LMHeadModel(GPT2Config(n_layer=2, n_head=2, n_embd=16, vocab_size=8))
    reference = e3_rm.create_reference_model(model, num_shared_layers=shared_layers)
    assert not reference.training
    assert all(not p.requires_grad for p in reference.parameters())
    if shared_layers is not None:
        # 共享参数使用相同存储，源模型对应层也必须冻结，才能保证参考策略不被更新。
        source = model.get_parameter("transformer.h.0.attn.c_attn.weight")
        target = reference.get_parameter("transformer.h.0.attn.c_attn.weight")
        assert source.data_ptr() == target.data_ptr()
        assert not source.requires_grad
    else:
        assert next(model.parameters()).data_ptr() != next(reference.parameters()).data_ptr()
