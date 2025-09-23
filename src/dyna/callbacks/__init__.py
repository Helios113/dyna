from llmfoundry.registry import callbacks
from dyna.callbacks.activation_monitor import ActivationMonitor
from dyna.callbacks.layer_usage_monitor import LayerUsageMonitor
from dyna.callbacks.entropy_callback import ShannonEntropyCallback
from dyna.callbacks.exit_entropy_callback import ExitEntropyCallback
from dyna.callbacks.expert_selection_callback import ExpertSelectionCallback
from dyna.callbacks.clean_stats import CleanMetrics
from dyna.callbacks.residual_stream_mag_callback import ResidualMagnitudeCallback
from dyna.callbacks.loop_number_callback import LoopNumberCallback
from dyna.callbacks.abbie_number import AbbieNumberCallback

__all__ = [
    "ActivationMonitor",
    "LayerUsageMonitor", 
    "ShannonEntropyCallback",
    "ExitEntropyCallback",
    "ExpertSelectionCallback",
    "CleanMetrics",
    "ResidualMagnitudeCallback",
    "LoopNumberCallback",
    "AbbieNumberCallback",
]

callbacks.register(
    "activation_monitor_c",
    func=ActivationMonitor,
)

callbacks.register(
    "layer_usage_monitor",
    func=LayerUsageMonitor,
)
callbacks.register("entropy_callback", func=ShannonEntropyCallback)
callbacks.register("exit_callback", func=ExitEntropyCallback)
callbacks.register("expert_selection_callback", func=ExpertSelectionCallback)
callbacks.register("clean_metrics", func=CleanMetrics)
callbacks.register("residual_magnitude", func=ResidualMagnitudeCallback)
callbacks.register("loop_number_callback", func=LoopNumberCallback)
callbacks.register("abbie_number_callback", func=AbbieNumberCallback)







