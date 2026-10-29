"""Configuration and optimizer-step curriculum for E3 reward modeling."""

import math
from dataclasses import dataclass, field

from ..trainer.reward_config import RewardConfig


@dataclass
class E3RewardConfig(RewardConfig):
    """
    Extend [`RewardConfig`] with the objectives described in the project README.

    Parameters:
        e3_invariance_coefficient (`float`, *optional*, defaults to `1.0`):
            Final weight of the reward invariance loss. Set to zero for ablations.
        e3_margin_coefficient (`float`, *optional*, defaults to `1.0`):
            Final weight of the preference margin consistency loss.
        e3_semantic_coefficient (`float`, *optional*, defaults to `1.0`):
            Final weight of the signed semantic ranking loss.
        e3_semantic_margin (`float`, *optional*, defaults to `1.0`):
            Positive minimum reward gap for labeled semantic pairs.
        e3_stage2_start_ratio (`float`, *optional*, defaults to `0.2`):
            Fraction of optimizer updates completed before introducing invariance losses.
        e3_stage3_start_ratio (`float`, *optional*, defaults to `0.6`):
            Fraction of optimizer updates completed before introducing semantic sensitivity.
        e3_ramp_ratio (`float`, *optional*, defaults to `0.1`):
            Fraction of total updates used for each linear ramp. Zero enables an immediate transition.
            Set both stage starts and this value to zero for simultaneous training.
    """

    # 创新点一至三的最终权重分别控制奖励不变性、偏好差值一致性和语义敏感性。
    # 将某项系数设为零即可关闭该目标，用于 README 中的逐项消融实验。
    e3_invariance_coefficient: float = field(default=1.0, metadata={"help": "Final reward invariance weight."})
    e3_margin_coefficient: float = field(default=1.0, metadata={"help": "Final margin consistency weight."})
    e3_semantic_coefficient: float = field(default=1.0, metadata={"help": "Final semantic sensitivity weight."})
    # 语义敏感性目标中的正排序间隔 m；不同于基础偏好数据的 margin 字段。
    e3_semantic_margin: float = field(default=1.0, metadata={"help": "Minimum signed semantic reward gap."})
    # 创新点四：默认 20% 进度进入阶段二，60% 进入阶段三，各用总步数的 10% 拉升权重。
    e3_stage2_start_ratio: float = field(default=0.2, metadata={"help": "Update fraction starting stage two."})
    e3_stage3_start_ratio: float = field(default=0.6, metadata={"help": "Update fraction starting stage three."})
    e3_ramp_ratio: float = field(default=0.1, metadata={"help": "Update fraction for each linear weight ramp."})

    def __post_init__(self):
        for name in ("e3_invariance_coefficient", "e3_margin_coefficient", "e3_semantic_coefficient"):
            value = vars(self)[name]
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"`{name}` must be finite and nonnegative.")
        if not math.isfinite(self.e3_semantic_margin) or self.e3_semantic_margin <= 0:
            raise ValueError("`e3_semantic_margin` must be finite and positive.")
        if not 0 <= self.e3_stage2_start_ratio <= self.e3_stage3_start_ratio < 1:
            raise ValueError("Stage start ratios must satisfy 0 <= stage2 <= stage3 < 1.")
        if not 0 <= self.e3_ramp_ratio <= 1 - self.e3_stage3_start_ratio:
            raise ValueError("`e3_ramp_ratio` must fit between stage three and the end of training.")
        super().__post_init__()


def get_e3_weights(args: E3RewardConfig, global_step: int, total_steps: int) -> dict[str, float]:
    """Return weights from completed optimizer updates, including after checkpoint restoration."""
    if total_steps <= 0:
        raise ValueError("`total_steps` must be positive.")
    # 创新点四：三阶段渐进调度。进度 t/T 由优化器更新次数决定，梯度累积不会提前推进课程。
    # 调度器不维护额外计数器，直接使用 TrainerState 中可随 checkpoint 恢复的 global_step。
    progress = min(max(global_step / total_steps, 0.0), 1.0)

    def ramp(start):
        # 线性拉升函数：clip((t/T - start) / ramp_ratio, 0, 1)。
        # 起点之前权重为零，拉升区间内平滑增加，之后保持最终权重，避免训练初期强正则导致坍缩。
        if args.e3_ramp_ratio == 0:
            return float(progress >= start)
        return min(max((progress - start) / args.e3_ramp_ratio, 0.0), 1.0)

    # 阶段一：三项附加权重均为零，仅训练 L_pref（及可选的基础奖励中心化项）。
    # 阶段二：共同拉升 L_inv 与 L_margin；阶段三：保留前两项并拉升 L_sem。
    invariant_scale = ramp(args.e3_stage2_start_ratio)
    return {
        "invariance": args.e3_invariance_coefficient * invariant_scale,
        "margin": args.e3_margin_coefficient * invariant_scale,
        "semantic": args.e3_semantic_coefficient * ramp(args.e3_stage3_start_ratio),
    }
