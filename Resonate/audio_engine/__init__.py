"""
Resonate Audio Engine - Live Music Reconstruction Pipeline

Core modules for processing phone-captured live music recordings
to studio-quality audio using ML source separation and DSP enhancement.
"""

from .pipeline import AudioPipeline
from .ingest import AudioIngest
from .separator import SeparatorEngine
from .mixing import StemMixer
from .mastering import AudioMaster

__version__ = "1.0.0"
__all__ = ["AudioPipeline", "AudioIngest", "SeparatorEngine", "StemMixer", "AudioMaster"]
