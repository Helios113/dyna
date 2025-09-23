from llmfoundry.registry import schedulers
from dyna.schedulers.wsld import WarmupStableLinearDecay


schedulers.register(
    "wsld",
    func=WarmupStableLinearDecay,
)
