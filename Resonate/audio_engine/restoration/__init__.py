"""
Audio Restoration Package - Advanced DSP restoration techniques

Provides:
- FrequencyRestorer: Extend frequency response from phone-limited range
- Dereverberator: Remove venue acoustics from recordings
"""

from .frequency import FrequencyRestorer, restore_frequency, FrequencyRestorationConfig
from .dereverberation import Dereverberator, dereverberate, DereverberationConfig

__version__ = "1.0.0"
__all__ = [
    "FrequencyRestorer",
    "restore_frequency",
    "FrequencyRestorationConfig",
    "Dereverberator",
    "dereverberate",
    "DereverberationConfig"
]
