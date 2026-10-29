"""Exercise uneven E3 relations with two real CPU/Gloo processes."""

import json
import os
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datasets import Dataset

from e3_rm import E3RewardTrainer
from tests.test_e3_reward import make_args, make_model_and_tokenizer, make_rows


def _train_rank(rank, directory):
    os.environ.update(
        RANK=str(rank),
        WORLD_SIZE="2",
        LOCAL_RANK=str(rank),
        LOCAL_WORLD_SIZE="2",
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT="29594",
        OMP_NUM_THREADS="1",
    )
    torch.set_num_threads(1)
    directory = Path(directory)
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=2,
        init_method=(directory / "rendezvous").as_uri(),
        timeout=timedelta(seconds=90),
    )
    try:
        model, tokenizer = make_model_and_tokenizer()
        args = make_args(
            directory / f"rank-{rank}",
            max_steps=1,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=2,
            max_grad_norm=0,
            e3_stage2_start_ratio=0,
            e3_stage3_start_ratio=0,
            e3_ramp_ratio=0,
        )
        trainer = E3RewardTrainer(
            model=model,
            args=args,
            train_dataset=Dataset.from_list(make_rows()[:4]),
            processing_class=tokenizer,
            optimizers=(torch.optim.SGD(model.parameters(), lr=0.01), None),
        )
        assert trainer.accelerator.num_processes == 2
        trainer.train()
        metrics = trainer.evaluate(Dataset.from_list(make_rows("eval")))
        # Five original examples require a duplicated sampler item across two ranks; it must not enter metrics.
        assert metrics["eval_accuracy_count"] == 5
        assert metrics["eval_reward_drift_count"] == 9
        assert metrics["eval_semantic_sensitivity_accuracy_count"] == 4
        if rank == 0:
            torch.save(trainer.model.state_dict(), directory / "distributed-weights.pt")
            (directory / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="requires CPU Gloo")
def test_e3_two_process_matches_single_update_and_counts(tmp_path):
    mp.spawn(_train_rank, args=(str(tmp_path),), nprocs=2, join=True)
    model, tokenizer = make_model_and_tokenizer()
    args = make_args(
        tmp_path / "single",
        max_steps=1,
        per_device_train_batch_size=4,
        max_grad_norm=0,
        e3_stage2_start_ratio=0,
        e3_stage3_start_ratio=0,
        e3_ramp_ratio=0,
    )
    trainer = E3RewardTrainer(
        model=model,
        args=args,
        train_dataset=Dataset.from_list(make_rows()[:4]),
        processing_class=tokenizer,
        optimizers=(torch.optim.SGD(model.parameters(), lr=0.01), None),
    )
    trainer.train()
    distributed = torch.load(tmp_path / "distributed-weights.pt", weights_only=True)
    for name, value in trainer.model.state_dict().items():
        torch.testing.assert_close(value, distributed[name], atol=1e-6, rtol=1e-5)
    reference = trainer.evaluate(Dataset.from_list(make_rows("eval")))
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    for key in (
        "eval_loss",
        "eval_accuracy",
        "eval_reward_drift",
        "eval_perturbation_flip_rate",
        "eval_semantic_sensitivity_accuracy",
    ):
        assert metrics[key] == pytest.approx(reference[key], abs=1e-6)
