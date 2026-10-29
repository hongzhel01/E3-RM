"""E3 reward modeling with verified relations and a resumable three-stage curriculum."""

import json
import math
from collections import defaultdict

import torch
from accelerate.logging import get_logger
from datasets import Dataset, Features, IterableDataset, List, Value

from ..data_utils import _tokenize, get_dataset_column_names
from ..trainer.base_trainer import _BaseTrainer
from ..trainer.reward_trainer import RewardTrainer
from ..trainer.utils import global_then_local_main_first
from .config import E3RewardConfig, get_e3_weights
from .data import SEMANTIC_TYPES, DataCollatorForE3Preference


logger = get_logger(__name__)


def e3_loss_terms(rewards, inputs, semantic_margin):
    """Return unreduced losses on original pairs and each valid relation; empty relations stay empty."""
    # 基础偏好排序始终保留：L_pref = -log sigmoid(r_w - r_l)，用于维持原始判别边界。
    # 数据自带的 margin 只调整基础排序目标，与下面的偏好差值一致性损失不是同一概念。
    chosen, rejected = rewards[inputs["preference_indices"]].unbind(-1)
    preference = -torch.nn.functional.logsigmoid(chosen - rejected - inputs.get("margin", 0.0))

    # 创新点一：奖励不变性损失 L_inv = (r(x, y) - r(x, T(y)))²。
    # 索引只连接原回答与经过验证的等价变体，使表面改写前后的绝对奖励接近。
    # 两侧奖励都保留梯度；此处返回逐关系误差，在 compute_loss 中按有效关系数求均值。
    original, transformed = rewards[inputs["invariance_indices"]].unbind(-1)
    invariance = (original - transformed).square()

    # 创新点二：偏好差值一致性损失 L_margin = ((r_w - r_l) - (r'_w - r'_l))²。
    # 四元索引对应同一原始偏好对及其两侧等价变体，约束变换前后的相对优势尺度。
    # 仅有单侧变体时无法构成四元关系，因此只参与 L_inv，不参与本项损失。
    chosen, rejected, changed_chosen, changed_rejected = rewards[inputs["margin_indices"]].unbind(-1)
    margin = ((chosen - rejected) - (changed_chosen - changed_rejected)).square()

    # 创新点三：语义敏感性损失 L_sem = max(0, m - s * (r_a - r_b))。
    # 独立语义标签 s=+1 表示 a 优于 b，s=-1 表示 b 优于 a；不从 chosen/rejected 推断方向。
    # 只有正确方向的分差达到正间隔 m 后惩罚才为零，避免不变性约束抹平关键语义差异。
    a, b = rewards[inputs["semantic_indices"]].unbind(-1)
    semantic = torch.relu(semantic_margin - inputs["semantic_sign"] * (a - b))
    return {"preference": preference, "invariance": invariance, "margin": margin, "semantic": semantic}


class E3RewardTrainer(RewardTrainer):
    """
    Train progressive semantic robustness while retaining the original reward trainer's model lifecycle.

    Accepts the same arguments as [`RewardTrainer`], with [`E3RewardConfig`] as `args`. Raw examples may add
    `source_id`, `e3_variants`, and `e3_semantic_pairs`. Only relations explicitly marked `verified=True` participate.
    See the README for the relation schema. No automatic edit is assumed to preserve meaning.
    """

    _name = "E3Reward"
    _tag_names = ["e3-rm", "reward-trainer", "progressive-semantic-robustness"]
    # 每个微批次的损失已按整个梯度累积窗口归一化，不再由 Trainer 除以微批次数。
    loss_is_scaled_for_ga = True

    def __init__(self, model, args: E3RewardConfig, **kwargs):
        if not isinstance(args, E3RewardConfig):
            raise TypeError("E3RewardTrainer requires E3RewardConfig.")
        self._e3_totals = {"train": defaultdict(lambda: [0.0, 0.0]), "eval": defaultdict(lambda: [0.0, 0.0])}
        self._check_source_overlap(kwargs.get("train_dataset"), kwargs.get("eval_dataset"))
        super().__init__(model=model, args=args, **kwargs)
        if kwargs.get("data_collator") is None:
            self.data_collator = DataCollatorForE3Preference(
                pad_token_id=self.processing_class.pad_token_id,
                pad_to_multiple_of=args.pad_to_multiple_of,
            )

    @staticmethod
    def _check_source_overlap(train_dataset, eval_dataset):
        if train_dataset is None or eval_dataset is None:
            return
        parts = eval_dataset if isinstance(eval_dataset, dict) else {"eval": eval_dataset}
        datasets = [("train", train_dataset), *parts.items()]
        seen = set()
        for name, dataset in datasets:
            if isinstance(dataset, IterableDataset):
                logger.warning("Cannot exhaust streaming %s to check source_id overlap; split sources upstream.", name)
                continue
            if "source_id" not in dataset.column_names:
                continue
            ids = {value for value in dataset["source_id"] if value is not None}
            if seen & ids:
                raise ValueError(
                    f"source_id overlap across dataset splits (including {name}); split before augmentation."
                )
            seen.update(ids)

    def _prepare_dataset(self, dataset, processing_class, args, dataset_name):
        if isinstance(dataset, Dataset) and len(dataset) == 0:
            raise ValueError(f"No {dataset_name} examples were provided.")
        columns = get_dataset_column_names(dataset)
        if "chosen_input_ids" in columns and "chosen_ids" not in columns:
            dataset = dataset.rename_column("chosen_input_ids", "chosen_ids")
            dataset = dataset.rename_column("rejected_input_ids", "rejected_ids")
            columns = get_dataset_column_names(dataset)

        def prepare(example):
            prompt = example.get("prompt")
            answers = {name: example.get(name) for name in ("chosen", "rejected")}
            if prompt is None and all(isinstance(answer, list) for answer in answers.values()):
                # 隐式对话将问题保存在原始 chosen/rejected 的共享消息前缀中；变体也必须继承该上下文。
                shared = 0
                for chosen_message, rejected_message in zip(answers["chosen"], answers["rejected"], strict=False):
                    if chosen_message != rejected_message:
                        break
                    shared += 1
                prompt = answers["chosen"][:shared]
                answers = {name: answer[shared:] for name, answer in answers.items()}
            tools = example.get("tools")
            tools = json.loads(tools) if isinstance(tools, str) else tools
            template_kwargs = {"tools": tools, **(example.get("chat_template_kwargs") or {})}

            def validate_ids(ids):
                if not isinstance(ids, list) or not ids:
                    raise ValueError("Token IDs must be a nonempty list.")
                if any(not isinstance(token, int) or isinstance(token, bool) or token < 0 for token in ids):
                    raise ValueError("Token IDs must be nonnegative integers.")
                return ids

            def encode(answer):
                if prompt is not None:
                    if not isinstance(prompt, type(answer)):
                        raise ValueError("prompt and answers must both be strings or both be message lists.")
                    answer = prompt + answer
                if isinstance(answer, str):
                    if not answer.endswith(processing_class.eos_token):
                        answer += processing_class.eos_token
                    return validate_ids(_tokenize(processing_class, answer)["input_ids"])
                if not isinstance(answer, list):
                    raise ValueError("E3 answers must be strings or message lists.")
                return validate_ids(_tokenize(processing_class, answer, **template_kwargs)["input_ids"])

            def fits(ids):
                return args.max_length is None or len(ids) <= args.max_length

            tokenized_variants = "e3_variant_ids" in example
            tokenized_semantics = "e3_semantic_ids" in example
            variants = example.get("e3_variant_ids" if tokenized_variants else "e3_variants") or []
            semantic_pairs = example.get("e3_semantic_ids" if tokenized_semantics else "e3_semantic_pairs") or []
            source_id = example.get("source_id")
            if source_id is not None and not isinstance(source_id, str):
                raise ValueError("source_id must be a string.")
            if (variants or semantic_pairs) and (not isinstance(source_id, str) or not source_id.strip()):
                raise ValueError("Examples with E3 relations require a nonempty string source_id.")
            output = {"source_id": source_id, "e3_variant_ids": [], "e3_semantic_ids": []}
            for name in ("chosen", "rejected"):
                output[f"{name}_ids"] = (
                    validate_ids(example[f"{name}_ids"]) if f"{name}_ids" in example else encode(answers[name])
                )
            if example.get("margin") is not None and not math.isfinite(example["margin"]):
                raise ValueError("Preference margin must be finite.")
            total, verified = len(variants) + len(semantic_pairs), 0
            for relation in variants:
                # 创新点一、二的数据前提：只信任上游显式验证的语义保持关系，不能把任意改写当成等价。
                if tokenized_variants and relation.get("verified") is None:
                    raise ValueError(
                        "Processed E3 variants require an explicit verified flag; regenerate from raw data."
                    )
                if relation.get("verified") is not True:
                    continue
                if not isinstance(relation.get("type"), str) or not relation["type"].strip():
                    raise ValueError("Verified E3 variants require a transformation type.")
                chosen_key, rejected_key = (
                    ("chosen_ids", "rejected_ids") if tokenized_variants else ("chosen", "rejected")
                )
                process = validate_ids if tokenized_variants else encode
                chosen = process(relation[chosen_key]) if relation.get(chosen_key) is not None else None
                rejected = process(relation[rejected_key]) if relation.get(rejected_key) is not None else None
                if chosen is None and rejected is None:
                    raise ValueError("Verified E3 variants require chosen and/or rejected.")
                verified += 1
                if all(fits(ids) for ids in (chosen, rejected) if ids is not None):
                    output["e3_variant_ids"].append(
                        {"chosen_ids": chosen, "rejected_ids": rejected, "type": relation["type"], "verified": True}
                    )
            for relation in semantic_pairs:
                # 创新点三使用独立的语义变化对及排序方向，不将事实错误样本混入不变性关系。
                if tokenized_semantics and relation.get("verified") is None:
                    raise ValueError(
                        "Processed semantic pairs require an explicit verified flag; regenerate from raw data."
                    )
                if relation.get("verified") is not True:
                    continue
                if relation.get("sign") not in (-1, 1) or isinstance(relation["sign"], bool):
                    raise ValueError("Verified semantic pairs require sign=-1 or sign=+1.")
                if not isinstance(relation.get("type"), str) or not relation["type"].strip():
                    raise ValueError("Verified semantic pairs require a transformation type.")
                a_key, b_key = ("a_ids", "b_ids") if tokenized_semantics else ("a", "b")
                process = validate_ids if tokenized_semantics else encode
                if relation.get(a_key) is None or relation.get(b_key) is None:
                    raise ValueError(f"Verified semantic pairs require both {a_key} and {b_key}.")
                a, b = process(relation[a_key]), process(relation[b_key])
                verified += 1
                if fits(a) and fits(b):
                    output["e3_semantic_ids"].append(
                        {"a_ids": a, "b_ids": b, "sign": relation["sign"], "type": relation["type"], "verified": True}
                    )
            output["e3_relations_total"] = total
            output["e3_relations_verified"] = verified
            output["e3_relations_kept"] = len(output["e3_variant_ids"]) + len(output["e3_semantic_ids"])
            output["e3_invariance_count"] = sum(
                int(relation["chosen_ids"] is not None) + int(relation["rejected_ids"] is not None)
                for relation in output["e3_variant_ids"]
            )
            output["e3_margin_count"] = sum(
                relation["chosen_ids"] is not None and relation["rejected_ids"] is not None
                for relation in output["e3_variant_ids"]
            )
            output["e3_semantic_count"] = len(output["e3_semantic_ids"])
            return output

        map_kwargs = {}
        if isinstance(dataset, Dataset):
            features = Features(dict(dataset.features))
            features.update(
                {
                    "chosen_ids": List(Value("int64")),
                    "rejected_ids": List(Value("int64")),
                    "source_id": Value("string"),
                    "e3_variant_ids": List(
                        {
                            "chosen_ids": List(Value("int64")),
                            "rejected_ids": List(Value("int64")),
                            "type": Value("string"),
                            "verified": Value("bool"),
                        }
                    ),
                    "e3_semantic_ids": List(
                        {
                            "a_ids": List(Value("int64")),
                            "b_ids": List(Value("int64")),
                            "sign": Value("int64"),
                            "type": Value("string"),
                            "verified": Value("bool"),
                        }
                    ),
                    "e3_relations_total": Value("int64"),
                    "e3_relations_verified": Value("int64"),
                    "e3_relations_kept": Value("int64"),
                    "e3_invariance_count": Value("int64"),
                    "e3_margin_count": Value("int64"),
                    "e3_semantic_count": Value("int64"),
                }
            )
            map_kwargs = {
                "num_proc": args.dataset_num_proc,
                "features": features,
                "desc": f"Preparing verified E3 relations ({dataset_name})",
            }
        with global_then_local_main_first():
            original_size = len(dataset) if isinstance(dataset, Dataset) else None
            dataset = dataset.map(prepare, **map_kwargs)
            if args.max_length is not None:
                filter_kwargs = {"num_proc": args.dataset_num_proc} if isinstance(dataset, Dataset) else {}
                dataset = dataset.filter(
                    lambda example: (
                        len(example["chosen_ids"]) <= args.max_length
                        and len(example["rejected_ids"]) <= args.max_length
                    ),
                    **filter_kwargs,
                )
            if isinstance(dataset, Dataset):
                logger.info(
                    "%s: kept %d/%d original pairs; relations supplied=%d, verified=%d, within length=%d. "
                    "Overlong variants are dropped without truncation; valid original pairs are retained.",
                    dataset_name,
                    len(dataset),
                    original_size,
                    sum(dataset["e3_relations_total"]),
                    sum(dataset["e3_relations_verified"]),
                    sum(dataset["e3_relations_kept"]),
                )
                if len(dataset) == 0:
                    raise ValueError(f"No {dataset_name} examples remain after max_length filtering.")
                if dataset_name == "train":
                    for name in ("invariance", "margin", "semantic"):
                        if vars(args)[f"e3_{name}_coefficient"] > 0 and sum(dataset[f"e3_{name}_count"]) == 0:
                            logger.warning(
                                "E3 %s coefficient is positive but no valid %s relations remain in the training "
                                "dataset; this objective will not contribute.",
                                name,
                                name,
                            )
        return dataset

    def get_batch_samples(self, epoch_iterator, num_batches, device):
        batches, num_items = super().get_batch_samples(epoch_iterator, num_batches, device)
        # 不同微批次的关系数量可能不同，必须按整个优化器更新窗口的有效分母计算，不能平均微批次均值。
        names = ("preference", "invariance", "margin", "semantic")
        counts = torch.tensor(
            [sum(batch[f"{name}_indices"].shape[0] for batch in batches) for name in names],
            dtype=torch.float32,
            device=device,
        )
        counts = self.accelerator.reduce(counts, reduction="sum")
        for batch in batches:
            batch["e3_normalization"] = counts
        return batches, num_items

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = ["chosen_ids", "rejected_ids", "margin", "e3_variant_ids", "e3_semantic_ids"]

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        if eval_dataset is not None and not isinstance(eval_dataset, str):
            self._check_source_overlap(self.train_dataset, eval_dataset)
        return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

    def evaluation_loop(
        self, dataloader, description, prediction_loss_only=None, ignore_keys=None, metric_key_prefix="eval"
    ):
        # 在评估循环返回前完成关系级聚合，使 evaluate、predict、日志和最佳 checkpoint 选择口径一致。
        self._e3_totals["eval"].clear()
        self._metrics["eval"].clear()
        output = super().evaluation_loop(dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix)
        output.metrics.update(self._flush_e3_metrics("eval", f"{metric_key_prefix}_"))
        return output

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        mode = "train" if self.model.training else "eval"
        total_steps = self.state.max_steps
        # 创新点四接入训练：根据已完成的优化器更新步数取动态权重，而不是按微批次推进阶段。
        # Trainer 从 checkpoint 恢复 global_step 后，同一计划总步数下会自动恢复课程位置与权重。
        # Standalone evaluation before train() has no planned update count; use final coefficients.
        weights = (
            get_e3_weights(self.args, self.state.global_step, total_steps)
            if total_steps > 0
            else {
                "invariance": self.args.e3_invariance_coefficient,
                "margin": self.args.e3_margin_coefficient,
                "semantic": self.args.e3_semantic_coefficient,
            }
        )
        active = torch.zeros(inputs["input_ids"].shape[0], dtype=torch.bool, device=inputs["input_ids"].device)
        active[inputs["preference_indices"].flatten()] = True
        # 阶段一只计算原始对；阶段二按非零权重加入等价变体；阶段三再加入语义变化对。
        # 评估计算全部有效关系，便于在任意训练阶段观察鲁棒性和语义判别力。
        for name in ("invariance", "margin", "semantic"):
            if weights[name] > 0 or mode == "eval":
                active[inputs[f"{name}_indices"].flatten()] = True
        # The model receives no relation labels, margins, source IDs, or other metadata.
        logits = (
            model(
                input_ids=inputs["input_ids"][active], attention_mask=inputs["attention_mask"][active], use_cache=False
            )
            .logits.squeeze(-1)
            .float()
        )
        rewards = logits.new_zeros(active.shape).masked_scatter(active, logits)
        losses = e3_loss_terms(rewards, inputs, self.args.e3_semantic_margin)
        chosen, rejected = rewards[inputs["preference_indices"]].unbind(-1)
        names = ("preference", "invariance", "margin", "semantic")
        counts = inputs.get("e3_normalization")
        if counts is None:
            counts = self.accelerator.reduce(
                logits.new_tensor([losses[name].numel() for name in names]), reduction="sum"
            )
        scale = self.accelerator.num_processes / counts.clamp_min(1)
        loss = losses["preference"].sum() * scale[0]
        if self.args.center_rewards_coefficient is not None:
            loss = loss + self.args.center_rewards_coefficient * (chosen + rejected).square().sum() * scale[0]
        # 联合目标：L = L_pref + λ_inv(t)L_inv + λ_margin(t)L_margin + λ_sem(t)L_sem。
        # 每项独立按有效关系数归一化；空关系张量的和为零，因此不会制造伪样本或产生 NaN。
        for index, name in enumerate(names[1:], start=1):
            # DDP averages gradients across ranks: scale local sums by the GLOBAL valid relation count.
            # All ranks participate, even if one rank has no relations of this kind.
            loss = loss + weights[name] * losses[name].sum() * scale[index]

        if mode == "train":
            tokens = self.accelerator.gather(inputs["attention_mask"][active].sum()).sum().item()
            self._total_train_tokens += tokens // self._tp_size
        self._metrics[mode]["num_tokens"] = [self._total_train_tokens]
        for name, weight in weights.items():
            self._metrics[mode][f"e3/weight_{name}"] = [weight]

        with torch.no_grad():
            self._record_e3_metrics(rewards, inputs, losses, weights, mode)
        # Keep prediction rows aligned with original examples, regardless of the number of variants.
        outputs = {"logits": torch.stack((chosen, rejected), dim=-1)}
        return (loss, outputs) if return_outputs else loss

    def _record_e3_metrics(self, rewards, inputs, losses, weights, mode):
        size = inputs["preference_indices"].shape[0]
        sums, counts = {}, {}

        def add(name, values, owners):
            sums[name] = rewards.new_zeros(size).index_add_(0, owners, values.float())
            counts[name] = rewards.new_zeros(size).index_add_(0, owners, torch.ones_like(values).float())

        owners = torch.arange(size, device=rewards.device)
        chosen, rejected = rewards[inputs["preference_indices"]].unbind(-1)
        add("accuracy", (chosen > rejected).float(), owners)
        add("margin", chosen - rejected, owners)
        add("mean_absolute_margin", (chosen - rejected).abs(), owners)
        add("mean_reward", (chosen + rejected) / 2, owners)
        add("e3/loss_preference", losses["preference"], owners)
        if self.args.center_rewards_coefficient is not None:
            add("e3/loss_center", (chosen + rejected).square(), owners)
        for name in ("invariance", "margin", "semantic"):
            if weights[name] == 0 and mode == "train":
                continue
            relation_owners = inputs[f"{name}_owners"]
            add(f"e3/loss_{name}", losses[name], relation_owners)
            if name == "invariance":
                a, b = rewards[inputs["invariance_indices"]].unbind(-1)
                add("reward_drift", (a - b).abs(), relation_owners)
            elif name == "margin":
                a, b, c, d = rewards[inputs["margin_indices"]].unbind(-1)
                # A transition to or from an exact tie counts as a flip.
                add("perturbation_flip_rate", (torch.sign(a - b) != torch.sign(c - d)).float(), relation_owners)
            else:
                a, b = rewards[inputs["semantic_indices"]].unbind(-1)
                correct = (inputs["semantic_sign"] * (a - b) > 0).float()
                add("semantic_sensitivity_accuracy", correct, relation_owners)
                for type_id, kind in enumerate(SEMANTIC_TYPES):
                    selected = inputs["semantic_type"] == type_id
                    add(f"semantic_sensitivity_accuracy/{kind}", correct[selected], relation_owners[selected])
        # One row per SOURCE example allows gather_for_metrics to remove sampler padding correctly.
        names = list(sums)
        # Every rank must gather the same columns, including ranks with empty relation tensors.
        stats = torch.stack([item for name in names for item in (sums[name], counts[name])], dim=-1)
        # 训练会预取整个累积窗口，end_of_dataloader 可能提前变为 True，不能据此裁剪每个训练微批次。
        # 评估逐批消费数据，可使用 gather_for_metrics 按原始样本去除分布式补齐项。
        gathered = (
            self.accelerator.gather(stats) if mode == "train" else self.accelerator.gather_for_metrics(stats)
        ).sum(0)
        for index, name in enumerate(names):
            aggregate = self._e3_totals[mode][name]
            aggregate[0] += gathered[2 * index].item()
            aggregate[1] += gathered[2 * index + 1].item()

    def _flush_e3_metrics(self, mode, prefix):
        logs = {}
        for name, (total, count) in self._e3_totals[mode].items():
            if count:
                logs[f"{prefix}{name}"] = total / count
            logs[f"{prefix}{name}_count"] = count
        for name, values in self._metrics[mode].items():
            logs[f"{prefix}{name}"] = sum(values) / len(values)
        if mode == "eval" and f"{prefix}e3/loss_preference" in logs:
            # Trainer 默认按原始样本数平均批次 loss；各批关系密度不同时，这不等于关系级联合目标。
            logs[f"{prefix}loss"] = (
                logs[f"{prefix}e3/loss_preference"]
                + sum(
                    logs[f"{prefix}e3/weight_{name}"] * logs.get(f"{prefix}e3/loss_{name}", 0.0)
                    for name in ("invariance", "margin", "semantic")
                )
                + (self.args.center_rewards_coefficient or 0.0) * logs.get(f"{prefix}e3/loss_center", 0.0)
            )
        self._e3_totals[mode].clear()
        self._metrics[mode].clear()
        return logs

    def log(self, logs, start_time=None):
        mode = "train" if self.model.training else "eval"
        logs.update(self._flush_e3_metrics(mode, "eval_" if mode == "eval" else ""))
        # E3 指标已按有效分母聚合并使用调用者的前缀，直接沿用公共训练器的日志写入机制。
        _BaseTrainer.log(self, logs, start_time)
