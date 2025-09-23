import logging
from typing import TYPE_CHECKING

from composer.optim.scheduler import (
    ComposerScheduler,
    _convert_time,
    _raise_if_warmup_and_max_incompatible,
    _raise_if_max_duration_exceeds_t_max,
    LinearScheduler,
    CosineAnnealingScheduler,
    _cosine_anneal,
)
from composer.core import State, Time

if TYPE_CHECKING:
    from typing import Protocol
else:
    # subclasses of Protocol cannot be instantiated in Python 3.8
    Protocol = object

log = logging.getLogger(__name__)


class ConstantWithLinWarmupAndCosCooldown(ComposerScheduler):
    def __init__(
        self,
        t_warmup: str | Time,
        t_cooldown: str | Time,
        t_max: str | Time = "1dur",
        alpha_f: float = 0.0,
        scale_warmup: bool = False,
        scale_cooldown: bool = False,
        t_demom: str | Time = "",
        t_offset: str | Time = "",
    ):
        self.t_warmup = t_warmup
        self.t_coolddown = t_cooldown
        self.t_max = t_max
        self.alpha_f = alpha_f
        self.scale_warmup = scale_warmup
        self.scale_cooldown = scale_cooldown
        self.t_denom = t_demom
        self.t_offset = t_offset
        self.warmup_scheduler = LinearScheduler(
            alpha_i=0.0, alpha_f=alpha_f, t_max=t_warmup
        )
        self.coold_down_scheduler = CosineAnnealingScheduler(
            alpha_f=alpha_f, t_max=t_cooldown
        )

    def __call__(self, state: State, ssr: float = 1.0):
        assert (
            state.max_duration is not None
        ), "max_duration should be set whenever schedulers are invoked"
        t_warmup = _convert_time(self.t_warmup, state)
        t_cooldown = _convert_time(self.t_coolddown, state)
        t_max = _convert_time(self.t_max, state, ssr=ssr)

        _raise_if_warmup_and_max_incompatible(t_warmup, t_max)
        _raise_if_warmup_and_max_incompatible(t_cooldown, t_max)
        _raise_if_max_duration_exceeds_t_max(t_max, state)
        t_cooldown = t_max - t_cooldown
        if self.t_denom != "":
            t_denom = _convert_time(self.t_denom, state)
            t_offset = _convert_time(self.t_offset, state)
            return _cosine_anneal(
                x=float(
                    (state.timestamp.get(t_cooldown.unit) - t_cooldown + t_offset)
                    / (t_denom)
                )
            )
        if state.timestamp < t_warmup:
            if self.scale_warmup:
                return self.warmup_scheduler(state, ssr)
            return self.warmup_scheduler(state)
        elif state.timestamp > t_cooldown:
            return _cosine_anneal(
                x=float(
                    (state.timestamp.get(t_cooldown.unit) - t_cooldown)
                    / (t_max - t_cooldown)
                )
            )
        else:
            return self.alpha_f

