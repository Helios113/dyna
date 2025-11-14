from llmfoundry.registry import callbacks

from dyna.callbacks.abbie_number import AbbieNumberCallback
from dyna.callbacks.clean_stats import CleanMetrics
from dyna.callbacks.entropy_callback import ShannonEntropyCallback
from dyna.callbacks.expert_selection_callback import ExpertSelectionCallback
from dyna.callbacks.gradient_noise import GradientNoiseScaleMonitor
from dyna.callbacks.layer_usage_monitor import LayerUsageMonitor
from dyna.callbacks.loop_number_callback import LoopNumberCallback
from dyna.callbacks.lr_scale import LrScaleCallback
from dyna.callbacks.residual_stream_mag_callback import ResidualMagnitudeCallback

__all__ = [
    "LayerUsageMonitor",
    "ShannonEntropyCallback",
    "ExpertSelectionCallback",
    "CleanMetrics",
    "ResidualMagnitudeCallback",
    "LoopNumberCallback",
    "AbbieNumberCallback",
    "LrScaleCallback",
    "GradientNoiseScaleMonitor",
]


callbacks.register(
    "layer_usage_monitor",
    func=LayerUsageMonitor,
)
callbacks.register("entropy_callback", func=ShannonEntropyCallback)
callbacks.register("expert_selection_callback", func=ExpertSelectionCallback)
callbacks.register("clean_metrics", func=CleanMetrics)
callbacks.register("residual_magnitude", func=ResidualMagnitudeCallback)
callbacks.register("loop_number_callback", func=LoopNumberCallback)
callbacks.register("abbie_number_callback", func=AbbieNumberCallback)
callbacks.register("lr_scale_callback", func=LrScaleCallback)
callbacks.register("gradient_noise_scale", func=GradientNoiseScaleMonitor)
