"""Version 1 of the context-conditioned, zero-shot SINDy prototype."""

from .foundation_encoder import (
    ChronosT5ContextEncoder,
    FoundationEncoderSpec,
    PandaPatchTSTContextEncoder,
)
from .model import AmortizedSINDy, AmortizedSINDyConfig, InferredDynamics
from .tabpfn_conditioner import (
    TabPFNCoefficientConditioner,
    TabPFNCoefficientSpec,
)

__all__ = [
    "AmortizedSINDy",
    "AmortizedSINDyConfig",
    "FoundationEncoderSpec",
    "ChronosT5ContextEncoder",
    "InferredDynamics",
    "PandaPatchTSTContextEncoder",
    "TabPFNCoefficientConditioner",
    "TabPFNCoefficientSpec",
]
