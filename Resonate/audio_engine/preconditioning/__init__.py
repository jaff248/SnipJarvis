"""
Pre-Conditioning Package - Prepare input audio BEFORE separation

Critical for phone recordings where:
- High noise floor degrades Demucs separation
- Clipping causes harsh artifacts
- AGC compression squashes dynamics

Modules:
- noise_reducer: Global noise reduction before separation
- declip: Detect and repair clipped peaks
- dynamics: Restore compressed dynamic range
- pipeline: Orchestrate pre-conditioning steps
"""

from .noise_reducer import NoiseReducer, reduce_noise, NoiseReductionConfig
from .declip import Declipper, declip_audio, DeclipConfig
from .dynamics import DynamicsRestorer, restore_dynamics, DynamicsConfig
from .pipeline import PreConditioningPipeline, PreConditioningConfig

__version__ = "1.0.0"
__all__ = [
    "NoiseReducer",
    "reduce_noise",
    "NoiseReductionConfig",
    "Declipper",
    "declip_audio",
    "DeclipConfig",
    "DynamicsRestorer",
    "restore_dynamics",
    "DynamicsConfig",
    "PreConditioningPipeline",
    "PreConditioningConfig",
]
