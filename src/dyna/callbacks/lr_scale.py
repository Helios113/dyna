from composer.core import Callback, State, Time, TimeUnit
from composer.loggers import Logger


class LrScaleCallback(Callback):
    def __init__(
        self,
        log_interval: str | int = "1ba",
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

        # base_depth = state.model.base_depth
        # current_depth = state.model.n_repeats * state.model.n_layers +
        #  state.model.head_size + state.model.tail_size
        # for i in state.optimizers[0].param_groups:
        #     i["lr_scale"] =
