from llmfoundry.registry import callbacks
from dyna.callbacks.activation_monitor import ActivationMonitor


callbacks.register(
    "activation_monitor_c",
    func=ActivationMonitor,
)
