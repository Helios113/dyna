from llmfoundry.registry import callbacks
from dyna.callbacks.activation_monitor import ActivationMonitor
from dyna.callbacks.layer_usage_monitor import LayerUsageMonitor
from dyna.callbacks.entropy_callback import ShannonEntropyCallback
from dyna.callbacks.exit_entropy_callback import ExitEntropyCallback




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
