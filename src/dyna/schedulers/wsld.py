from composer.core import State, Time
from composer.optim.scheduler import (
    ComposerScheduler,
    LinearScheduler,
    _convert_time,
    _raise_if_max_duration_exceeds_t_max,
    _raise_if_warmup_and_max_incompatible,
)


def _lin_cooldown(t: Time, t_max: Time, alpha_i=1.0, alpha_f=0.0):
    """Linear cooldown function that drives the factor from alpha_i to alpha_f.

    Args:
        t: Current time elapsed in the cooldown phase
        t_max: Total duration of the cooldown phase
        alpha_i: Initial factor at the start of cooldown (default: 1.0)
        alpha_f: Final factor at the end of cooldown (default: 0.0)

    Returns:
        Current factor linearly interpolated between alpha_i and alpha_f
    """
    frac_of_total = min(1.0, (t / t_max).value)

    current_factor = alpha_i + frac_of_total * (alpha_f - alpha_i)
    return current_factor


class WarmupStableLinearDecay(ComposerScheduler):
    def __init__(
        self,
        t_warmup: str | Time,
        t_cooldown: str | Time,
        t_max: str | Time = "1dur",
        alpha_f: float = 0.0,
        scale_warmup: bool = False,
        scale_cooldown: bool = False,
    ):
        """Linear warmup and cooldown scheduler.

        Args:
            t_warmup: Duration of warmup phase.
            t_cooldown: Duration of cooldown phase.
            t_max: Maximum duration of the scheduler.
            alpha_f: Final factor at the end of cooldown (default: 0.0).
            scale_warmup: Scale warmup phase by factor (default: False).
            scale_cooldown: Scale cooldown phase by factor (default: False).

        Returns:
            Current factor linearly interpolated between alpha_i and alpha_f
        """
        self.t_warmup = t_warmup - Time(1, "ba")
        self.t_cooldown = t_cooldown + Time(1, "ba")
        self.t_max = t_max
        self.alpha_f = alpha_f
        self.scale_warmup = scale_warmup
        self.scale_cooldown = scale_cooldown

        self.warmup_scheduler = LinearScheduler(
            alpha_i=0.0, alpha_f=1.0, t_max=self.t_warmup
        )

    def __call__(self, state: State, ssr: float = 1.0):
        assert (
            state.max_duration is not None
        ), "max_duration should be set whenever schedulers are invoked"
        t_warmup = _convert_time(self.t_warmup, state)
        t_cooldown = _convert_time(self.t_cooldown, state)
        t_max = _convert_time(self.t_max, state, ssr=ssr)
        _raise_if_warmup_and_max_incompatible(t_warmup, t_max)
        _raise_if_warmup_and_max_incompatible(t_cooldown, t_max)
        _raise_if_max_duration_exceeds_t_max(t_max, state)
        t_cooldown = t_max - t_cooldown

        if state.timestamp < t_warmup:
            if self.scale_warmup:
                return self.warmup_scheduler(state, ssr)
            return self.warmup_scheduler(state)
        elif state.timestamp > t_cooldown:
            if self.scale_cooldown:
                return _lin_cooldown(
                    state.timestamp.batch - t_cooldown, self.t_cooldown - Time(1, "ba")
                )
            return _lin_cooldown(
                state.timestamp.batch - t_cooldown, self.t_cooldown - Time(1, "ba")
            )
        else:
            return self.alpha_f
