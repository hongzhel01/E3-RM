"""Verify E3 mathematics and public training behavior with real local models and datasets."""

import copy
import json
import math

import pytest
import torch
from datasets import Dataset, DatasetDict
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2ForSequenceClassification, PreTrainedTokenizerFast

from e3_rm import E3RewardConfig, E3RewardTrainer, RewardConfig, RewardTrainer
from e3_rm.e3 import DataCollatorForE3Preference, get_e3_weights
from e3_rm.e3.trainer import e3_loss_terms
from e3_rm.scripts.reward import make_parser


def make_args(tmp_path, config_class=E3RewardConfig, **kwargs):
    values = dict(
        output_dir=str(tmp_path),
        use_cpu=True,
        bf16=False,
        fp16=False,
        gradient_checkpointing=False,
        dataloader_pin_memory=False,
        report_to="none",
        save_strategy="no",
        disable_tqdm=True,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        max_steps=6,
        max_length=32,
        logging_steps=1,
        seed=42,
    )
    values.update(kwargs)
    return config_class(**values)


def make_model_and_tokenizer():
    vocab = {
        word: i for i, word in enumerate(["[PAD]", "[EOS]", "[UNK]", "hello", "good", "bad", "world", "yes", "no"])
    }
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", eos_token="[EOS]", unk_token="[UNK]"
    )
    tokenizer.chat_template = "{% for message in messages %}{{ message['content'] + ' ' }}{% endfor %}{{ eos_token }}"
    torch.manual_seed(42)
    model = GPT2ForSequenceClassification(
        GPT2Config(
            vocab_size=len(vocab),
            n_positions=128,
            n_embd=16,
            n_layer=1,
            n_head=2,
            num_labels=1,
            pad_token_id=0,
            eos_token_id=1,
            bos_token_id=1,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
        )
    )
    model.config._attn_implementation = "eager"
    return model, tokenizer


def make_rows(prefix="train"):
    return [
        {
            "source_id": f"{prefix}-{index}",
            "prompt": "hello ",
            "chosen": "good",
            "rejected": "bad",
            "e3_variants": [
                {"chosen": "good world", "rejected": "bad world", "verified": True, "type": "paraphrase"},
                {"chosen": "yes", "rejected": None, "verified": True, "type": "format"},
                {"chosen": "bad", "rejected": "good", "verified": False, "type": "unverified"},
            ]
            if index % 2 == 0
            else [],
            "e3_semantic_pairs": [
                {"a": "good", "b": "bad", "sign": 1, "verified": True, "type": "fact_reversal"},
                {"a": "no", "b": "yes", "sign": -1, "verified": True, "type": "instruction_violation"},
            ]
            if index % 3 == 0
            else [],
        }
        for index in range(5)
    ]


def test_loss_values_and_signed_gradients():
    inputs = {
        "preference_indices": torch.tensor([[0, 1]]),
        "invariance_indices": torch.tensor([[0, 2], [1, 3]]),
        "margin_indices": torch.tensor([[0, 1, 2, 3]]),
        "semantic_indices": torch.tensor([[4, 5], [6, 7]]),
        "semantic_sign": torch.tensor([1.0, -1.0]),
        "margin": torch.tensor([0.5]),
    }
    rewards = torch.tensor([3.0, 1.0, 2.0, 1.0, 0.0, 1.0, 1.0, 0.0], requires_grad=True)
    terms = e3_loss_terms(rewards, inputs, semantic_margin=1.0)
    torch.testing.assert_close(terms["preference"], -torch.nn.functional.logsigmoid(torch.tensor([1.5])))
    torch.testing.assert_close(terms["invariance"], torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(terms["margin"], torch.tensor([1.0]))
    torch.testing.assert_close(terms["semantic"], torch.tensor([2.0, 2.0]))
    sum(values.mean() for values in terms.values()).backward()
    assert rewards.grad[4] < 0 < rewards.grad[5]
    assert rewards.grad[7] < 0 < rewards.grad[6]
    assert rewards.grad[0] != 0 and rewards.grad[2] != 0


@pytest.mark.parametrize(
    "step,expected", [(0, (0, 0)), (20, (0, 0)), (25, (0.5, 0)), (40, (1, 0)), (65, (1, 0.5)), (100, (1, 1))]
)
def test_optimizer_step_curriculum(tmp_path, step, expected):
    weights = get_e3_weights(make_args(tmp_path), step, 100)
    assert weights["invariance"] == pytest.approx(expected[0])
    assert weights["margin"] == pytest.approx(expected[0])
    assert weights["semantic"] == pytest.approx(expected[1])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"e3_invariance_coefficient": -1},
        {"e3_semantic_coefficient": float("nan")},
        {"e3_semantic_margin": 0},
        {"e3_stage2_start_ratio": 0.8},
        {"e3_ramp_ratio": 0.5},
    ],
)
def test_invalid_config(tmp_path, kwargs):
    with pytest.raises(ValueError):
        make_args(tmp_path, **kwargs)


def test_collator_explicit_indices_and_empty_relations():
    collator = DataCollatorForE3Preference(pad_token_id=0)
    batch = collator(
        [
            {
                "chosen_ids": [3, 4],
                "rejected_ids": [5],
                "e3_variant_ids": [
                    {"chosen_ids": [6], "rejected_ids": [7]},
                    {"chosen_ids": None, "rejected_ids": [8]},
                ],
                "e3_semantic_ids": [{"a_ids": [4], "b_ids": [5], "sign": -1, "type": "fact_reversal"}],
            },
            {"chosen_ids": [4], "rejected_ids": [6, 7]},
        ]
    )
    assert next(iter(batch)) == "preference_indices"
    assert batch["preference_indices"].tolist() == [[0, 2], [1, 3]]
    assert batch["invariance_indices"].tolist() == [[0, 4], [2, 5], [2, 6]]
    assert batch["margin_indices"].tolist() == [[0, 2, 4, 5]]
    assert batch["semantic_indices"].tolist() == [[7, 8]]
    assert batch["semantic_sign"].tolist() == [-1]
    empty = collator([{"chosen_ids": [4], "rejected_ids": [5]}])
    terms = e3_loss_terms(torch.tensor([1.0, 0.0]), empty, 1.0)
    assert terms["preference"].numel() == 1
    assert all(terms[name].numel() == 0 for name in ("invariance", "margin", "semantic"))


def test_verified_relations_prompt_eos_and_length_filter(tmp_path):
    model, tokenizer = make_model_and_tokenizer()
    rows = make_rows()
    rows[0]["e3_variants"].append({"chosen": "good " * 40, "rejected": "bad", "verified": True, "type": "length"})
    rows[1]["chosen"] = "good " * 40
    trainer = E3RewardTrainer(
        model=model, args=make_args(tmp_path), train_dataset=Dataset.from_list(rows), processing_class=tokenizer
    )
    assert len(trainer.train_dataset) == 4
    example = trainer.train_dataset[0]
    assert len(example["e3_variant_ids"]) == 2
    assert len(example["e3_semantic_ids"]) == 2
    assert example["e3_relations_total"] == 6
    assert example["e3_relations_verified"] == 5
    assert example["e3_relations_kept"] == 4
    assert example["e3_variant_ids"][0]["chosen_ids"] == tokenizer("hello good world[EOS]")["input_ids"]
    assert example["chosen_ids"] == tokenizer("hello good[EOS]")["input_ids"]


@pytest.mark.parametrize("invalid", ["source_id", "sign", "type"])
def test_invalid_verified_metadata_is_rejected(tmp_path, invalid):
    model, tokenizer = make_model_and_tokenizer()
    rows = make_rows()
    if invalid == "source_id":
        rows[0]["source_id"] = ""
    else:
        rows[0]["e3_semantic_pairs"][0][invalid] = 0 if invalid == "sign" else ""
    with pytest.raises(ValueError, match=invalid if invalid != "type" else "transformation type"):
        E3RewardTrainer(
            model=model, args=make_args(tmp_path), train_dataset=Dataset.from_list(rows), processing_class=tokenizer
        )


def test_source_leakage_is_rejected(tmp_path):
    model, tokenizer = make_model_and_tokenizer()
    dataset = Dataset.from_list(make_rows())
    with pytest.raises(ValueError, match="source_id overlap"):
        E3RewardTrainer(
            model=model,
            args=make_args(tmp_path),
            train_dataset=dataset,
            eval_dataset=DatasetDict({"validation": dataset}),
            processing_class=tokenizer,
        )


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("lora", [False, True])
def test_train_evaluate_and_save(tmp_path, streaming, lora):
    model, tokenizer = make_model_and_tokenizer()
    dataset = Dataset.from_list(make_rows())
    if streaming:
        dataset = dataset.to_iterable_dataset()
    kwargs = {}
    if lora:
        from peft import LoraConfig

        kwargs["peft_config"] = LoraConfig(
            task_type="SEQ_CLS",
            r=2,
            lora_alpha=4,
            target_modules=["c_attn"],
            fan_in_fan_out=True,
            modules_to_save=["score"],
        )
    trainer = E3RewardTrainer(
        model=model,
        args=make_args(tmp_path, max_steps=3, e3_stage2_start_ratio=0, e3_stage3_start_ratio=0, e3_ramp_ratio=0),
        train_dataset=dataset,
        processing_class=tokenizer,
        **kwargs,
    )
    old_score = copy.deepcopy(trainer.model.state_dict())
    result = trainer.train()
    assert trainer.state.global_step == 3
    assert math.isfinite(result.training_loss)
    assert any(not torch.equal(value, trainer.model.state_dict()[key]) for key, value in old_score.items())
    metrics = trainer.evaluate(Dataset.from_list(make_rows("eval")))
    for key in ("eval_accuracy", "eval_perturbation_flip_rate", "eval_semantic_sensitivity_accuracy"):
        assert 0 <= metrics[key] <= 1
    assert metrics["eval_reward_drift_count"] == 9
    assert metrics["eval_perturbation_flip_rate_count"] == 3
    assert metrics["eval_semantic_sensitivity_accuracy_count"] == 4
    assert metrics["eval_accuracy_count"] == 5
    trainer.save_model(str(tmp_path / "saved"))
    if lora:
        from peft import PeftModel

        base, _ = make_model_and_tokenizer()
        restored = PeftModel.from_pretrained(base, tmp_path / "saved")
    else:
        restored = GPT2ForSequenceClassification.from_pretrained(tmp_path / "saved")
    for name, value in trainer.model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value)


def test_baseline_equivalence_with_all_auxiliary_weights_zero(tmp_path):
    model, tokenizer = make_model_and_tokenizer()
    base_model = copy.deepcopy(model)
    rows = Dataset.from_list(make_rows())
    args = make_args(tmp_path / "e3", e3_invariance_coefficient=0, e3_margin_coefficient=0, e3_semantic_coefficient=0)
    e3 = E3RewardTrainer(model=model, args=args, train_dataset=rows, processing_class=tokenizer)
    baseline_args = make_args(tmp_path / "baseline", config_class=RewardConfig)
    baseline = RewardTrainer(model=base_model, args=baseline_args, train_dataset=rows, processing_class=tokenizer)
    e3_result, baseline_result = e3.train(), baseline.train()
    assert e3_result.training_loss == pytest.approx(baseline_result.training_loss, abs=1e-6)
    for name, value in e3.model.state_dict().items():
        torch.testing.assert_close(value, baseline.model.state_dict()[name], atol=1e-6, rtol=1e-5)


def test_eval_metrics_use_relation_counts_across_uneven_batches(tmp_path):
    model, tokenizer = make_model_and_tokenizer()
    trainer = E3RewardTrainer(
        model=model,
        args=make_args(
            tmp_path, e3_stage2_start_ratio=0, e3_stage3_start_ratio=0, e3_ramp_ratio=0, center_rewards_coefficient=0.1
        ),
        train_dataset=Dataset.from_list(make_rows()),
        processing_class=tokenizer,
    )
    evaluation = Dataset.from_list(make_rows("eval"))
    first = trainer.evaluate(evaluation)
    trainer.args.per_device_eval_batch_size = 1
    second = trainer.evaluate(evaluation)
    for key in (
        "eval_accuracy",
        "eval_reward_drift",
        "eval_perturbation_flip_rate",
        "eval_semantic_sensitivity_accuracy",
        "eval_mean_reward",
        "eval_mean_absolute_margin",
        "eval_loss",
    ):
        assert first[key] == pytest.approx(second[key], abs=1e-6)


def test_named_evaluation_splits_keep_separate_metric_prefixes(tmp_path):
    model, tokenizer = make_model_and_tokenizer()
    trainer = E3RewardTrainer(
        model=model,
        args=make_args(tmp_path),
        train_dataset=Dataset.from_list(make_rows()),
        processing_class=tokenizer,
    )
    parts = DatasetDict(
        {
            "validation": Dataset.from_list(make_rows("validation")),
            "test": Dataset.from_list(make_rows("test")),
        }
    )
    metrics = trainer.evaluate(parts)
    assert metrics["eval_validation_reward_drift_count"] == 9
    assert metrics["eval_test_reward_drift_count"] == 9
    assert metrics["eval_validation_accuracy_count"] == metrics["eval_test_accuracy_count"] == 5


def test_curriculum_checkpoint_resume_matches_continuous_training(tmp_path):
    model, tokenizer = make_model_and_tokenizer()
    args = make_args(tmp_path / "continuous", save_strategy="steps", save_steps=3)
    dataset = Dataset.from_list(make_rows())
    trainer = E3RewardTrainer(model=model, args=args, train_dataset=dataset, processing_class=tokenizer)
    trainer.train()
    model2, tokenizer2 = make_model_and_tokenizer()
    resumed = E3RewardTrainer(
        model=model2, args=make_args(tmp_path / "resumed"), train_dataset=dataset, processing_class=tokenizer2
    )
    resumed.train(resume_from_checkpoint=str(tmp_path / "continuous" / "checkpoint-3"))
    assert resumed.state.global_step == 6
    final_log = next(log for log in reversed(resumed.state.log_history) if "e3/weight_semantic" in log)
    assert final_log["e3/weight_semantic"] == 1
    for name, value in trainer.model.state_dict().items():
        torch.testing.assert_close(value, resumed.model.state_dict()[name], atol=1e-6, rtol=1e-5)


def test_conversational_relations(tmp_path):
    rows = make_rows()
    for row in rows:
        row["prompt"] = [{"role": "user", "content": row["prompt"]}]
        for key in ("chosen", "rejected"):
            row[key] = [{"role": "assistant", "content": row[key]}]
        for relation in row["e3_variants"] + row["e3_semantic_pairs"]:
            for key in ("chosen", "rejected", "a", "b"):
                if relation.get(key) is not None:
                    relation[key] = [{"role": "assistant", "content": relation[key]}]
    model, tokenizer = make_model_and_tokenizer()
    trainer = E3RewardTrainer(
        model=model,
        args=make_args(tmp_path, max_steps=1),
        train_dataset=Dataset.from_list(rows),
        processing_class=tokenizer,
    )
    assert math.isfinite(trainer.train().training_loss)


def test_cli_selects_e3_from_yaml(tmp_path):
    path = tmp_path / "reward.yaml"
    path.write_text(
        "e3: true\nuse_cpu: true\nbf16: false\nreport_to: none\ne3_semantic_coefficient: 0.5\n", encoding="utf-8"
    )
    script_args, training_args, _, _ = make_parser().parse_args_and_config(["--config", str(path)])
    assert script_args.e3 is True
    assert isinstance(training_args, E3RewardConfig)
    assert training_args.e3_semantic_coefficient == 0.5


def test_stage_one_and_gradient_accumulation_use_original_pairs_only(tmp_path):
    model, tokenizer = make_model_and_tokenizer()
    trainer = E3RewardTrainer(
        model=model,
        args=make_args(tmp_path, max_steps=1, gradient_accumulation_steps=2),
        train_dataset=Dataset.from_list(make_rows()),
        processing_class=tokenizer,
    )
    trainer.train()
    step_log = next(log for log in trainer.state.log_history if "e3/weight_invariance" in log)
    assert step_log["e3/weight_invariance"] == 0
    assert step_log["e3/weight_semantic"] == 0
    # Two microbatches, each with two original pairs of three tokens per answer.
    assert step_log["num_tokens"] == 24
    assert "e3/loss_invariance" not in step_log


def test_evaluation_without_relations_reports_zero_counts(tmp_path):
    model, tokenizer = make_model_and_tokenizer()
    rows = [{"prompt": "hello ", "chosen": "good", "rejected": "bad"}]
    trainer = E3RewardTrainer(
        model=model, args=make_args(tmp_path), train_dataset=Dataset.from_list(rows), processing_class=tokenizer
    )
    metrics = trainer.evaluate(Dataset.from_list(rows))
    assert metrics["eval_accuracy_count"] == 1
    assert metrics["eval_reward_drift_count"] == 0
    assert metrics["eval_semantic_sensitivity_accuracy_count"] == 0
    assert "eval_reward_drift" not in metrics
    assert "eval_semantic_sensitivity_accuracy" not in metrics


def test_cli_runs_e3_training_with_local_model(tmp_path):
    from e3_rm.cli import main

    model, tokenizer = make_model_and_tokenizer()
    initial = tmp_path / "initial"
    model.save_pretrained(initial)
    tokenizer.save_pretrained(initial)
    data_file = tmp_path / "train.jsonl"
    data_file.write_text("\n".join(json.dumps(row) for row in make_rows()), encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(
        f"model_name_or_path: {initial.as_posix()}\n"
        f"output_dir: {(tmp_path / 'output').as_posix()}\n"
        "e3: true\nuse_cpu: true\nbf16: false\nfp16: false\nreport_to: none\n"
        "max_steps: 1\nper_device_train_batch_size: 2\nlogging_steps: 1\n"
        "gradient_checkpointing: false\nsave_strategy: steps\nsave_steps: 1\n"
        "e3_stage2_start_ratio: 0\ne3_stage3_start_ratio: 0\ne3_ramp_ratio: 0\n"
        f"datasets:\n  - path: json\n    data_files: {data_file.as_posix()}\n    split: train\n",
        encoding="utf-8",
    )
    assert main(["reward", "--config", str(config)]) == 0
    state = json.loads((tmp_path / "output" / "checkpoint-1" / "trainer_state.json").read_text())
    assert state["global_step"] == 1
    assert any(log.get("e3/weight_semantic") == 1 for log in state["log_history"])
    assert (tmp_path / "output" / "model.safetensors").is_file()


def test_accumulated_update_matches_one_batch_with_uneven_relations(tmp_path):
    dataset = Dataset.from_list(make_rows()).to_iterable_dataset()
    trained = []
    for batch_size, accumulation in ((5, 1), (2, 3)):
        model, tokenizer = make_model_and_tokenizer()
        args = make_args(
            tmp_path / f"batch-{batch_size}",
            max_steps=1,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=accumulation,
            max_grad_norm=0,
            center_rewards_coefficient=0.1,
            e3_stage2_start_ratio=0,
            e3_stage3_start_ratio=0,
            e3_ramp_ratio=0,
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        trainer = E3RewardTrainer(
            model=model, args=args, train_dataset=dataset, processing_class=tokenizer, optimizers=(optimizer, None)
        )
        result = trainer.train()
        trained.append((result.training_loss, copy.deepcopy(trainer.model.state_dict())))
    assert trained[0][0] == pytest.approx(trained[1][0], abs=1e-6)
    for name, value in trained[0][1].items():
        torch.testing.assert_close(value, trained[1][1][name], atol=1e-6, rtol=1e-5)


def test_preprocessed_data_is_filtered_under_new_length_limit(tmp_path):
    model, tokenizer = make_model_and_tokenizer()
    prepared = E3RewardTrainer(
        model=model,
        args=make_args(tmp_path / "prepare"),
        train_dataset=Dataset.from_list(make_rows()),
        processing_class=tokenizer,
    ).train_dataset
    model2, tokenizer2 = make_model_and_tokenizer()
    trainer = E3RewardTrainer(
        model=model2,
        args=make_args(tmp_path / "filtered", max_length=3),
        train_dataset=prepared,
        processing_class=tokenizer2,
    )
    # Two-sided four-token paraphrases are too long; three-token one-sided relations remain.
    assert len(trainer.train_dataset) == 5
    assert len(trainer.train_dataset[0]["e3_variant_ids"]) == 1
    overlong = prepared.map(lambda row: {"chosen_ids": row["chosen_ids"] + [4] * 40})
    with pytest.raises(ValueError, match="No train examples"):
        E3RewardTrainer(
            model=model2, args=make_args(tmp_path / "empty"), train_dataset=overlong, processing_class=tokenizer2
        )


@pytest.mark.parametrize("invalid", ["sign", "verified", "source_id"])
def test_preprocessed_data_does_not_bypass_relation_validation(tmp_path, invalid):
    row = {
        "source_id": "processed-1",
        "chosen_ids": [3, 4, 1],
        "rejected_ids": [3, 5, 1],
        "e3_variant_ids": [],
        "e3_semantic_ids": [
            {"a_ids": [3, 4, 1], "b_ids": [3, 5, 1], "sign": 1, "type": "fact_reversal", "verified": True}
        ],
    }
    if invalid == "source_id":
        row["source_id"] = ""
    else:
        row["e3_semantic_ids"][0][invalid] = 0 if invalid == "sign" else None
    model, tokenizer = make_model_and_tokenizer()
    with pytest.raises(ValueError, match=invalid):
        E3RewardTrainer(
            model=model, args=make_args(tmp_path), train_dataset=Dataset.from_list([row]), processing_class=tokenizer
        )


def test_implicit_chat_prompt_is_preserved_for_variants(tmp_path):
    row = {
        "source_id": "chat-1",
        "chosen": [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "good"}],
        "rejected": [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "bad"}],
        "e3_variants": [
            {"chosen": [{"role": "assistant", "content": "good world"}], "verified": True, "type": "paraphrase"}
        ],
    }
    model, tokenizer = make_model_and_tokenizer()
    trainer = E3RewardTrainer(
        model=model, args=make_args(tmp_path), train_dataset=Dataset.from_list([row]), processing_class=tokenizer
    )
    ids = trainer.train_dataset[0]["e3_variant_ids"][0]["chosen_ids"]
    assert ids == tokenizer("hello good world [EOS]")["input_ids"]


def test_e3_without_usable_relations_warns(tmp_path, caplog):
    model, tokenizer = make_model_and_tokenizer()
    rows = [{"prompt": "hello ", "chosen": "good", "rejected": "bad"}]
    E3RewardTrainer(
        model=model, args=make_args(tmp_path), train_dataset=Dataset.from_list(rows), processing_class=tokenizer
    )
    assert "no valid" in caplog.text.lower()


def test_named_evaluation_weights_use_the_requested_prefix(tmp_path):
    model, tokenizer = make_model_and_tokenizer()
    trainer = E3RewardTrainer(
        model=model, args=make_args(tmp_path), train_dataset=Dataset.from_list(make_rows()), processing_class=tokenizer
    )
    metrics = trainer.evaluate(Dataset.from_list(make_rows("test")), metric_key_prefix="test")
    assert "test_e3/weight_semantic" in metrics
    assert "eval_e3/weight_semantic" not in metrics


@pytest.mark.parametrize("max_steps", [-1, 6])
def test_standalone_evaluation_uses_final_coefficients(tmp_path, max_steps):
    model, tokenizer = make_model_and_tokenizer()
    trainer = E3RewardTrainer(
        model=model,
        args=make_args(tmp_path, max_steps=max_steps),
        train_dataset=Dataset.from_list(make_rows()),
        processing_class=tokenizer,
    )
    metrics = trainer.evaluate(Dataset.from_list(make_rows("eval")))
    assert metrics["eval_e3/weight_invariance"] == 1
    assert metrics["eval_e3/weight_semantic"] == 1
    assert metrics["eval_loss"] == pytest.approx(
        sum(metrics[f"eval_e3/loss_{name}"] for name in ("preference", "invariance", "margin", "semantic")), abs=1e-6
    )


def test_predict_reports_the_same_objective_as_evaluate(tmp_path):
    model, tokenizer = make_model_and_tokenizer()
    trainer = E3RewardTrainer(
        model=model,
        args=make_args(tmp_path),
        train_dataset=Dataset.from_list(make_rows()),
        eval_dataset=Dataset.from_list(make_rows("eval")),
        processing_class=tokenizer,
    )
    predictions = trainer.predict(trainer.eval_dataset)
    evaluated = trainer.evaluate()
    assert predictions.predictions.shape == (5, 2)
    assert predictions.metrics["test_loss"] == pytest.approx(evaluated["eval_loss"], abs=1e-6)
    assert predictions.metrics["test_reward_drift_count"] == evaluated["eval_reward_drift_count"] == 9


@pytest.mark.parametrize("margin", [float("nan"), float("inf")])
def test_nonfinite_preference_margin_is_rejected(tmp_path, margin):
    model, tokenizer = make_model_and_tokenizer()
    rows = make_rows()
    rows[0]["margin"] = margin
    with pytest.raises(ValueError, match="margin must be finite"):
        E3RewardTrainer(
            model=model, args=make_args(tmp_path), train_dataset=Dataset.from_list(rows), processing_class=tokenizer
        )


def test_empty_dataset_has_a_clear_error(tmp_path):
    model, tokenizer = make_model_and_tokenizer()
    with pytest.raises(ValueError, match="No train examples"):
        E3RewardTrainer(
            model=model, args=make_args(tmp_path), train_dataset=Dataset.from_list([]), processing_class=tokenizer
        )


def test_best_checkpoint_uses_named_relation_weighted_loss(tmp_path):
    model, tokenizer = make_model_and_tokenizer()
    args = make_args(
        tmp_path,
        max_steps=2,
        eval_strategy="steps",
        eval_steps=1,
        save_strategy="steps",
        save_steps=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_validation_loss",
        greater_is_better=False,
        e3_stage2_start_ratio=0,
        e3_stage3_start_ratio=0,
        e3_ramp_ratio=0,
    )
    trainer = E3RewardTrainer(
        model=model,
        args=args,
        train_dataset=Dataset.from_list(make_rows()),
        eval_dataset={"validation": Dataset.from_list(make_rows("validation"))},
        processing_class=tokenizer,
    )
    trainer.train()
    evaluations = [log for log in trainer.state.log_history if "eval_validation_loss" in log]
    assert len(evaluations) == 2
    for log in evaluations:
        assert log["eval_validation_loss"] == pytest.approx(
            sum(log[f"eval_validation_e3/loss_{name}"] for name in ("preference", "invariance", "margin", "semantic")),
            abs=1e-6,
        )
    assert trainer.state.best_metric == pytest.approx(min(log["eval_validation_loss"] for log in evaluations))
    assert trainer.state.best_model_checkpoint is not None


def test_benchmark_sources_and_rule_verified_relations():
    from e3_rm.e3.benchmark import UNITS, VALUES, make_benchmark_splits

    splits = make_benchmark_splits(train_size=16, validation_size=16, test_size=16)
    seen = set()
    signs = set()
    for dataset in splits.values():
        assert not seen.intersection(dataset["source_id"])
        seen.update(dataset["source_id"])
        for row in dataset:
            parts = row["prompt"].split()
            value, unit = parts[3], parts[10]

            def content(text):
                return (
                    [token for token in text.split() if token in VALUES],
                    [token for token in text.split() if token in UNITS],
                )

            def correct(text, expected=(value, unit)):
                return content(text) == ([expected[0]], [expected[1]])

            assert correct(row["chosen"]) and not correct(row["rejected"])
            for relation in row["e3_variants"]:
                assert content(relation["chosen"]) == content(row["chosen"])
                assert content(relation["rejected"]) == content(row["rejected"])
            for relation in row["e3_semantic_pairs"]:
                signs.add(relation["sign"])
                assert relation["sign"] * (int(correct(relation["a"])) - int(correct(relation["b"]))) == 1
    assert signs == {-1, 1}


def test_benchmark_validation_only_and_comparable_metric_counts(tmp_path):
    from e3_rm.e3.benchmark import make_benchmark_splits, run_ablation

    splits = make_benchmark_splits(train_size=16, validation_size=16, test_size=16)
    for method in ("A_baseline", "E_progressive"):
        run = run_ablation(splits, method, seed=7, steps=4, output_dir=tmp_path, validation_only=True)
        assert "test" not in run
        assert run["parameters"] < 100_000
        assert run["validation"]["accuracy_count"] == 16
        assert run["validation"]["reward_drift_count"] == 96
        assert run["validation"]["semantic_sensitivity_accuracy_count"] == 48


def test_benchmark_summary_reports_sample_standard_deviation():
    from e3_rm.e3.benchmark import METRICS, summarize

    runs = [
        {
            "method": "example",
            "validation": dict.fromkeys((*METRICS, "normalized_reward_drift"), value),
            "train_seconds": 1,
            "train_tokens": 10,
        }
        for value in (0.5, 0.75, 1.0)
    ]
    metric = summarize(runs)["example"]["validation"]["accuracy"]
    assert metric["mean"] == 0.75
    assert metric["std"] == 0.25
