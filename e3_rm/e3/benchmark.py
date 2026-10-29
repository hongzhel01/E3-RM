"""Reproducible CPU ablations on a synthetic, source-disjoint reward task."""

import argparse
import hashlib
import json
import platform
import random
import statistics
import sys
import time
from pathlib import Path

import torch
import transformers
from datasets import Dataset, disable_progress_bars
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2ForSequenceClassification, PreTrainedTokenizerFast
from transformers.trainer_callback import PrinterCallback

from .. import RewardConfig, RewardTrainer
from .config import E3RewardConfig
from .trainer import E3RewardTrainer


VALUES = ("red", "blue", "green", "yellow")
UNITS = ("kg", "meter", "second", "liter")
STYLES = ("{value} {unit}", "Answer : {value} {unit} .", "### {value} {unit} ###", "Result -> {value} {unit} .")
METRICS = ("accuracy", "perturbation_flip_rate", "reward_drift", "semantic_sensitivity_accuracy", "margin")
METHODS = {
    "A_baseline": (0.0, 0.0, 0.0, False),
    "B_invariance": (1.0, 0.0, 0.0, False),
    "C_margin": (0.0, 1.0, 0.0, False),
    "D_semantic": (0.0, 0.0, 1.0, False),
    "E_progressive": (1.0, 1.0, 1.0, True),
    "F_simultaneous": (1.0, 1.0, 1.0, False),
}


def make_benchmark_splits(train_size=64, validation_size=32, test_size=64, data_seed=2026):
    """Split source records within each value/unit stratum BEFORE rendering any variants."""
    sizes = {"train": train_size, "validation": validation_size, "test": test_size}
    if any(size <= 0 or size % 16 for size in sizes.values()):
        raise ValueError("Each split size must be a positive multiple of 16 (four values by four units).")
    rng = random.Random(data_seed)
    records = {name: [] for name in sizes}
    for value_index, value in enumerate(VALUES):
        for unit_index, unit in enumerate(UNITS):
            group = [
                (f"item{value_index}{unit_index}{index}", value, unit) for index in range(sum(sizes.values()) // 16)
            ]
            rng.shuffle(group)
            offset = 0
            for name, size in sizes.items():
                count = size // 16
                records[name].extend(group[offset : offset + count])
                offset += count
    # All facts, desired units, and source identities are fixed before augmentation.
    splits = {}
    for name, partition in records.items():
        rng.shuffle(partition)
        rows = []
        for index, (source, value, unit) in enumerate(partition):
            wrong_value = VALUES[(VALUES.index(value) + 1) % len(VALUES)]
            wrong_unit = UNITS[(UNITS.index(unit) + 1) % len(UNITS)]
            # Training has a perfect format shortcut; held-out original pairs counterbalance it.
            chosen_style, rejected_style = (1, 2) if name == "train" or index % 2 == 0 else (2, 1)

            def render(style, answer_value, answer_unit):
                text = STYLES[style].format(value=answer_value, unit=answer_unit)
                # The controlled templates may change layout, never the answer's value or unit.
                assert [token for token in text.split() if token in VALUES] == [answer_value]
                assert [token for token in text.split() if token in UNITS] == [answer_unit]
                return text

            variants = [
                {
                    "chosen": render(rejected_style, value, unit),
                    "rejected": render(chosen_style, wrong_value, unit),
                    "type": "format",
                    "verified": True,
                },
                {
                    "chosen": render(0, value, unit),
                    "rejected": render(0, wrong_value, unit),
                    "type": "decoration_removal",
                    "verified": True,
                },
            ]
            if name != "train":
                # The Result template is never used by training answers or training variants.
                variants.append(
                    {
                        "chosen": render(3, value, unit),
                        "rejected": render(3, wrong_value, unit),
                        "type": "held_out_format",
                        "verified": True,
                    }
                )
            correct = render(0, value, unit)
            negatives = {
                "fact_reversal": render(0, wrong_value, unit),
                "condition_omission": value,
                "instruction_violation": render(0, value, wrong_unit),
            }
            semantic = []
            for kind, negative in negatives.items():
                sign = 1 if (index + len(semantic)) % 2 == 0 else -1
                semantic.append(
                    {
                        "a": correct if sign == 1 else negative,
                        "b": negative if sign == 1 else correct,
                        "sign": sign,
                        "type": kind,
                        "verified": True,
                    }
                )
            rows.append(
                {
                    "source_id": source,
                    "prompt": f"Record {source} value {value} . Return this value with unit {unit} . Any layout allowed . ",
                    "chosen": render(chosen_style, value, unit),
                    "rejected": render(rejected_style, wrong_value, unit),
                    "e3_variants": variants,
                    "e3_semantic_pairs": semantic,
                }
            )
        splits[name] = Dataset.from_list(rows)
    ids = [set(dataset["source_id"]) for dataset in splits.values()]
    assert not any(first & second for i, first in enumerate(ids) for second in ids[i + 1 :])
    return splits


def make_benchmark_model(splits, seed):
    """Create the same fixed-vocabulary, random-initialized tiny GPT-2 for every method of a seed."""
    # The vocabulary is specified by the benchmark grammar; it is not fitted to held-out answer labels.
    words = [
        "[PAD]",
        "[EOS]",
        "[UNK]",
        "Record",
        "value",
        ".",
        "Return",
        "this",
        "with",
        "unit",
        "Any",
        "layout",
        "allowed",
        "Answer",
        ":",
        "###",
        "Result",
        "->",
        *VALUES,
        *UNITS,
    ]
    words += sorted(source for dataset in splits.values() for source in dataset["source_id"])
    backend = Tokenizer(WordLevel({word: index for index, word in enumerate(words)}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", eos_token="[EOS]", unk_token="[UNK]"
    )
    torch.manual_seed(seed)
    model = GPT2ForSequenceClassification(
        GPT2Config(
            vocab_size=len(words),
            n_positions=96,
            n_embd=48,
            n_layer=2,
            n_head=4,
            num_labels=1,
            pad_token_id=0,
            eos_token_id=1,
            bos_token_id=1,
            resid_pdrop=0,
            embd_pdrop=0,
            attn_pdrop=0,
        )
    )
    model.config._attn_implementation = "eager"
    return model, tokenizer


def run_ablation(splits, method, seed, steps, output_dir, validation_only=False):
    model, tokenizer = make_benchmark_model(splits, seed)
    inv, margin, semantic, progressive = METHODS[method]
    common = dict(
        output_dir=str(output_dir / f"{method}-{seed}"),
        use_cpu=True,
        bf16=False,
        fp16=False,
        max_steps=steps,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=0.001,
        lr_scheduler_type="constant",
        max_length=64,
        center_rewards_coefficient=0.01,
        gradient_checkpointing=False,
        dataloader_pin_memory=False,
        save_strategy="no",
        logging_strategy="no",
        report_to="none",
        disable_tqdm=True,
        seed=seed,
    )
    if method == "A_baseline":
        trainer = RewardTrainer(
            model=model, args=RewardConfig(**common), train_dataset=splits["train"], processing_class=tokenizer
        )
    else:
        args = E3RewardConfig(
            **common,
            e3_invariance_coefficient=inv,
            e3_margin_coefficient=margin,
            e3_semantic_coefficient=semantic,
            e3_stage2_start_ratio=0.2 if progressive else 0,
            e3_stage3_start_ratio=0.6 if progressive else 0,
            e3_ramp_ratio=0.1 if progressive else 0,
        )
        trainer = E3RewardTrainer(model=model, args=args, train_dataset=splits["train"], processing_class=tokenizer)
    trainer.remove_callback(PrinterCallback)
    start = time.perf_counter()
    training = trainer.train()
    seconds = time.perf_counter() - start
    tokens = max(log.get("num_tokens", 0) for log in trainer.state.log_history)
    # All trained models use the exact same evaluation pipeline and final objective coefficients.
    evaluator = E3RewardTrainer(
        model=trainer.model,
        args=E3RewardConfig(**{**common, "output_dir": str(output_dir / "evaluation"), "max_steps": -1}),
        train_dataset=splits["train"],
        processing_class=tokenizer,
    )
    evaluator.remove_callback(PrinterCallback)
    result = {
        "method": method,
        "seed": seed,
        "steps": steps,
        "training_loss": training.training_loss,
        "train_seconds": seconds,
        "train_tokens": tokens,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    for name in ("validation",) if validation_only else ("validation", "test"):
        measured = evaluator.evaluate(splits[name], metric_key_prefix=name)
        result[name] = {
            key.removeprefix(f"{name}_"): value
            for key, value in measured.items()
            if key.startswith(f"{name}_")
            and not any(part in key for part in ("runtime", "per_second", "preparation_time", "num_tokens"))
        }
        result[name]["normalized_reward_drift"] = measured[f"{name}_reward_drift"] / max(
            measured[f"{name}_mean_absolute_margin"], 1e-8
        )
    return result


def summarize(runs):
    """Report all methods and seeds; do not filter runs based on test performance."""
    summaries = {}
    for method in dict.fromkeys(run["method"] for run in runs):
        selected = [run for run in runs if run["method"] == method]
        summary = {}
        for split in ("validation", "test"):
            if split not in selected[0]:
                continue
            summary[split] = {}
            for metric in (*METRICS, "normalized_reward_drift"):
                values = [run[split][metric] for run in selected]
                summary[split][metric] = {
                    "mean": statistics.mean(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                }
        for metric in ("train_seconds", "train_tokens"):
            values = [run[metric] for run in selected]
            summary[metric] = {
                "mean": statistics.mean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            }
        summaries[method] = summary
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/tiny-ablation"))
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--seeds", type=int, nargs="+", default=[13, 42, 87])
    parser.add_argument("--methods", choices=list(METHODS), nargs="+", default=list(METHODS))
    parser.add_argument("--validation_only", action="store_true")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if (
        args.steps < 2
        or args.threads < 1
        or len(set(args.seeds)) != len(args.seeds)
        or len(set(args.methods)) != len(args.methods)
    ):
        parser.error("Use at least two updates, positive threads, distinct seeds, and distinct methods.")
    torch.set_num_threads(args.threads)
    disable_progress_bars()
    transformers.utils.logging.set_verbosity_error()
    splits = make_benchmark_splits()
    dataset_json = json.dumps({name: data.to_list() for name, data in splits.items()}, sort_keys=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "task": "synthetic value/unit matching with a spurious training-format shortcut",
        "model": "random-initialized GPT-2, 2 layers, width 48, 4 heads",
        "selection": "fixed final update; no early stopping or selection on test results",
        "default_budget_selection": "1500 updates chosen using validation only at seed 7, after comparing 200 and 1500; "
        "seeds 13, 42, and 87 reserved for the final comparison",
        "data_seed": 2026,
        "split_sizes": {name: len(data) for name, data in splits.items()},
        "dataset_sha256": hashlib.sha256(dataset_json.encode()).hexdigest(),
        "settings": {**vars(args), "output_dir": str(args.output_dir)},
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": "cpu",
        },
        "runs": [],
    }
    path = args.output_dir / "results.json"
    for seed in args.seeds:
        for method in args.methods:
            result = run_ablation(splits, method, seed, args.steps, args.output_dir, args.validation_only)
            report["runs"].append(result)
            report["summary"] = summarize(report["runs"])
            path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
            sys.stdout.write(
                json.dumps(
                    {
                        "method": method,
                        "seed": seed,
                        "seconds": round(result["train_seconds"], 2),
                        **{metric: round(result["validation"][metric], 4) for metric in METRICS},
                    }
                )
                + "\n"
            )
            sys.stdout.flush()


if __name__ == "__main__":
    main()
