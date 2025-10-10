import torch
from composer.core import Callback, State, Time, TimeUnit
from composer.loggers import Logger


class AbbieNumberCallback(Callback):
    def __init__(
        self,
        log_interval: str | int = "100ba",
        number: int = 6,
        log_key: str = "abbie_number",
    ):
        self.log_interval = (
            Time.from_timestring(log_interval)
            if isinstance(log_interval, str)
            else Time(log_interval, TimeUnit.BATCH)
        )

        self.number = number
        self.log_key = log_key  # Initialize log_key
        # Storage for per-layer entropy data
        # self.entropy_data = []  # Add this back for the plot

    def _should_log(self, state: State) -> bool:
        """Determine if it's time to log based on the log_interval."""
        if isinstance(self.log_interval, Time):
            return (
                state.timestamp.get(self.log_interval.unit) % self.log_interval.value
                == 0
            )
        return False

    def batch_start(self, state: State, logger: Logger) -> None:
        """Called at the end of each batch to compute and log entropy."""
        if not self._should_log(state):
            return

        number = torch.randint(1, self.number + 1, (1,)).item()
        print(f"Setting new abbie number: {number}", flush=True)
        state.model.model.transformer.repeats = number
        logger.log_metrics({self.log_key: number})
