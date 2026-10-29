"""Batch explicit, verified relations without inferring semantic equivalence from text edits."""

from dataclasses import dataclass
from typing import Any

import torch
from transformers.data.data_collator import DataCollatorMixin

from ..trainer.utils import pad


SEMANTIC_TYPES = ("fact_reversal", "condition_omission", "instruction_violation", "other")


@dataclass
class DataCollatorForE3Preference(DataCollatorMixin):
    """
    Pad original pairs and a variable number of verified E3 relations.

    Args:
        pad_token_id (`int`):
            Token used for right padding.
        pad_to_multiple_of (`int`, *optional*):
            Optional sequence length padding multiple.
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            Tensor format. Only PyTorch is supported.

    Processed examples contain `chosen_ids`, `rejected_ids`, `e3_variant_ids`, and `e3_semantic_ids`.
    All relation tensors use explicit indices into the flattened model batch. Owner tensors identify the original
    example, allowing evaluation to discard distributed sampler padding before aggregating relation counts.
    """

    pad_token_id: int
    pad_to_multiple_of: int | None = None
    return_tensors: str = "pt"

    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        size = len(examples)
        sequences = [example["chosen_ids"] for example in examples]
        sequences += [example["rejected_ids"] for example in examples]
        invariance, margins, semantic = [], [], []
        inv_owners, margin_owners, sem_owners, signs, types = [], [], [], [], []
        for owner, example in enumerate(examples):
            for variant in example.get("e3_variant_ids") or []:
                pair = []
                for name, original in (("chosen_ids", owner), ("rejected_ids", owner + size)):
                    ids = variant.get(name)
                    if ids is None:
                        pair.append(None)
                        continue
                    transformed = len(sequences)
                    sequences.append(ids)
                    pair.append(transformed)
                    # 创新点一：为每个有效单侧等价变体建立“原回答—变体”的奖励一致性索引。
                    invariance.append([original, transformed])
                    inv_owners.append(owner)
                if all(index is not None for index in pair):
                    # 创新点二：两侧等价变体齐备时才建立差值一致性四元组，避免错配或补造另一侧。
                    margins.append([owner, owner + size, *pair])
                    margin_owners.append(owner)
            for relation in example.get("e3_semantic_ids") or []:
                index = len(sequences)
                sequences.extend([relation["a_ids"], relation["b_ids"]])
                # 创新点三：语义变化对单独建立索引，显式保留 sign，支持任一方向的正确排序。
                semantic.append([index, index + 1])
                sem_owners.append(owner)
                signs.append(relation["sign"])
                kind = relation["type"]
                types.append(SEMANTIC_TYPES.index(kind) if kind in SEMANTIC_TYPES else len(SEMANTIC_TYPES) - 1)

        sequences = [torch.tensor(ids, dtype=torch.long) for ids in sequences]
        output = {
            # Trainer infers the example count from the first tensor. Variants are not independent examples.
            "preference_indices": torch.tensor([[i, i + size] for i in range(size)], dtype=torch.long),
            "input_ids": pad(
                sequences,
                padding_value=self.pad_token_id,
                padding_side="right",
                pad_to_multiple_of=self.pad_to_multiple_of,
            ),
            "attention_mask": pad(
                [torch.ones_like(ids) for ids in sequences],
                padding_value=0,
                padding_side="right",
                pad_to_multiple_of=self.pad_to_multiple_of,
            ),
            "invariance_indices": torch.tensor(invariance, dtype=torch.long).reshape(-1, 2),
            "margin_indices": torch.tensor(margins, dtype=torch.long).reshape(-1, 4),
            "semantic_indices": torch.tensor(semantic, dtype=torch.long).reshape(-1, 2),
            "invariance_owners": torch.tensor(inv_owners, dtype=torch.long),
            "margin_owners": torch.tensor(margin_owners, dtype=torch.long),
            "semantic_owners": torch.tensor(sem_owners, dtype=torch.long),
            "semantic_sign": torch.tensor(signs, dtype=torch.float32),
            "semantic_type": torch.tensor(types, dtype=torch.long),
        }
        if any(example.get("margin") is not None for example in examples):
            output["margin"] = torch.tensor([example.get("margin") or 0.0 for example in examples])
        return output
