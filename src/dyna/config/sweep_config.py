from dataclasses import dataclass


@dataclass
class SweepConfig:
    method: str
    metric: dict
    parameters: dict
