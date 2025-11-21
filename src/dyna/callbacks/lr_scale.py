from composer.core import Callback, State, Time, TimeUnit
from composer.loggers import Logger


class LrScaleCallback(Callback):
    def __init__(
        self,
        log_interval: str | int = "1ba",
        eps: float = 1e-8,
        base_lr: float = 1e-8,
    ):
        """Initialize the AbbieNumberCallback.

        Args:
            log_interval (str | int): The interval at which to log the abbie number.
            number (int): The maximum number to generate.
            log_key (str): The key to use when logging the abbie number.
        """
        self.log_interval = (
            Time.from_timestring(log_interval)
            if isinstance(log_interval, str)
            else Time(log_interval, TimeUnit.BATCH)
        )
        self.eps = eps
        self.base_lr = base_lr

    def _should_log(self, state: State) -> bool:
        """Determine if it's time to log based on the log_interval."""
        if isinstance(self.log_interval, Time):
            return (
                state.timestamp.get(self.log_interval.unit) % self.log_interval.value
                == 0
            )
        return False

    def after_backward(self, state: State, logger: Logger) -> None:
        """Called at the end of each batch to compute and log entropy."""
        if not self._should_log(state):
            return
        # think about how this changes for geiping layers
        base_depth = state.model.model.transformer.base_depth
        base_width = state.model.model.transformer.base_width
        current_width = state.model.model.transformer.current_width
        base_depth = state.model.model.transformer.base_depth

        current_depth = (
            state.model.model.transformer.active_repeats
            * state.model.model.transformer.n_layers
        )
        depth_lr_scaling = current_depth / base_depth
        width_lr_scaling = (current_width / base_width) ** (-1)

        adam_eps = (
            self.eps
            * (current_width / base_width) ** (-1)
            * (current_depth / base_depth) ** (-1)
        )
        optim_groups = [
            {
                "lr": 1.0*self.base_lr,
                "eps": self.eps,
            },
            {
                "lr": depth_lr_scaling*self.base_lr,
                "eps": adam_eps,
            },
            {
                "lr": width_lr_scaling * depth_lr_scaling*self.base_lr,
                "eps": adam_eps,
            },
            {
                "lr": depth_lr_scaling*self.base_lr,
                "eps": adam_eps,
            },
            {
                "lr": 1.0*self.base_lr,
                "eps": adam_eps,
            },
            {
                "lr": 1.0*self.base_lr,
                "eps": self.eps,
            },
        ]
        for idx, ob in enumerate(state.optimizers[0].param_groups):
            for j in ob:
                if j in optim_groups[idx]:
                    ob[j] = optim_groups[idx][j]
