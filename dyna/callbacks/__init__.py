from llmfoundry.registry import callbacks
from dyna.callbacks.activation_monitor import ActivationMonitor
from dyna.callbacks.layer_usage_monitor import LayerUsageMonitor


callbacks.register(
    "activation_monitor_c",
    func=ActivationMonitor,
)

callbacks.register(
    "layer_usage_monitor",
    func=LayerUsageMonitor,
)