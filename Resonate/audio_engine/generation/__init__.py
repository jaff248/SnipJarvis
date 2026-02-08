"""
Generation module for Resonate - JASCO integration for AI-powered stem regeneration.

This module provides classes for generating and regenerating audio stems using
AudioCraft's JASCO/MusicGen model, with support for selective regeneration
and smooth blending of damaged regions.

Key Classes:
    - JASCOGenerator: Main generator class wrapping AudioCraft's MusicGen/JASCO
    - StemRegenerator: Orchestrates selective regeneration of damaged regions
    - Blender: Handles crossfade blending of regenerated audio
    - GenerationConfig: Configuration dataclass for generation parameters
    - GenerationResult: Result dataclass containing generated audio and metadata
    - RegenerationRegion: Dataclass for damaged region identification
    - RegenerationPlan: Complete plan for stem regeneration
    - MelodyConditioning: Melody contour conditioning parameters
    - DrumConditioning: Drum pattern conditioning parameters
    - GenerationCallbacks: Progress reporting callbacks
    - ModelLoadError: Exception raised when model loading fails
    - GenerationError: Exception raised when generation fails
    - StemType: Enum for supported stem types (vocals, drums, bass, other, all)
    - ChordConditioning: Chord progression conditioning parameters
    - TempoConditioning: Tempo conditioning parameters
"""

from .jasco_generator import (
    JASCOGenerator,
    GenerationConfig,
    GenerationResult,
    MelodyConditioning,
    DrumConditioning,
    ChordConditioning,
    TempoConditioning,
    GenerationCallbacks,
    ModelLoadError,
    GenerationError,
    StemType,
    ConditioningType,
    create_generator,
)

from .stem_regenerator import (
    StemRegenerator,
    RegenerationRegion,
    RegenerationPlan,
    RegenerationSummary,
    create_regenerator,
    regenerate_stem,
)

from .blender import (
    Blender,
    create_crossfade,
    match_and_blend,
)

__all__ = [
    # Main generator
    "JASCOGenerator",
    "create_generator",
    
    # Stem regenerator
    "StemRegenerator",
    "create_regenerator",
    "regenerate_stem",
    
    # Regeneration data classes
    "RegenerationRegion",
    "RegenerationPlan",
    "RegenerationSummary",
    
    # Blender
    "Blender",
    "create_crossfade",
    "match_and_blend",
    
    # Configuration and results
    "GenerationConfig",
    "GenerationResult",
    
    # Conditioning dataclasses
    "MelodyConditioning",
    "DrumConditioning",
    "ChordConditioning",
    "TempoConditioning",
    
    # Callbacks
    "GenerationCallbacks",
    
    # Enums
    "StemType",
    "ConditioningType",
    
    # Exceptions
    "ModelLoadError",
    "GenerationError",
]
