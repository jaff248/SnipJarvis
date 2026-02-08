"""
Enhancement Pipeline - Orchestrates all stem enhancers

Manages the per-stem enhancement process, applying appropriate
processing to vocals, drums, bass, and other instruments.
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .base import (
    BaseStemEnhancer, EnhancementConfig, EnhancementResult,
    EnhancementPipeline, create_enhancer
)
from .vocal import VocalEnhancer, VocalConfig
from .drums import DrumEnhancer, DrumConfig
from .bass import BassEnhancer, BassConfig
from .instruments import InstrumentEnhancer, InstrumentConfig

logger = logging.getLogger(__name__)


class ProcessingMode(Enum):
    """Processing quality modes."""
    PREVIEW = "preview"  # Fast, lower quality
    RENDER = "render"    # Slow, high quality


@dataclass
class GlobalEnhancementConfig:
    """Global configuration for all stem enhancements."""
    # Processing mode
    mode: ProcessingMode = ProcessingMode.RENDER
    
    # Overall intensity (affects all stems)
    intensity: float = 0.5
    
    # Individual stem intensities (override global)
    vocal_intensity: float = None
    drum_intensity: float = None
    bass_intensity: float = None
    other_intensity: float = None
    
    # Enable/disable specific stems
    enable_vocals: bool = True
    enable_drums: bool = True
    enable_bass: bool = True
    enable_other: bool = True
    
    # Caching
    use_cache: bool = True
    
    def __post_init__(self):
        """Set defaults for per-stem intensities."""
        if self.vocal_intensity is None:
            self.vocal_intensity = self.intensity
        if self.drum_intensity is None:
            self.drum_intensity = self.intensity
        if self.bass_intensity is None:
            self.bass_intensity = self.intensity
        if self.other_intensity is None:
            self.other_intensity = self.intensity


@dataclass
class StemEnhancementResult:
    """Result of stem enhancement processing."""
    stems: Dict[str, np.ndarray]
    sample_rate: int
    enhancements_applied: Dict[str, List[str]] = field(default_factory=dict)
    original_stems: Dict[str, np.ndarray] = field(default_factory=dict)
    processing_time: float = 0.0
    
    def get_stem(self, name: str) -> Optional[np.ndarray]:
        """Get enhanced stem by name."""
        return self.stems.get(name)
    
    def get_original(self, name: str) -> Optional[np.ndarray]:
        """Get original stem by name."""
        return self.original_stems.get(name)


class StemEnhancementPipeline:
    """
    Orchestrates per-stem audio enhancement.
    
    Manages:
    - Individual stem enhancers (vocals, drums, bass, other)
    - Configuration per stem type
    - Parallel or sequential processing
    - Result aggregation
    
    The pipeline applies different enhancement strategies to each
    stem based on its characteristics for optimal results.
    """
    
    STEM_NAMES = ['vocals', 'drums', 'bass', 'other']
    
    def __init__(self, config: GlobalEnhancementConfig = None):
        """
        Initialize enhancement pipeline.
        
        Args:
            config: Global configuration (uses defaults if None)
        """
        self.config = config or GlobalEnhancementConfig()
        
        # Initialize stem enhancers
        self._enhancers: Dict[str, BaseStemEnhancer] = {}
        self._setup_enhancers()
        
        logger.info(f"Initialized StemEnhancementPipeline: {self.config.mode.value}")
    
    def _setup_enhancers(self):
        """Initialize enhancers with appropriate configurations."""
        # Vocal enhancer
        if self.config.enable_vocals:
            vocal_config = VocalConfig(
                intensity=self.config.vocal_intensity,
                noise_reduction=True,
                eq=True,
                compression=True
            )
            self._enhancers['vocals'] = VocalEnhancer(vocal_config)
        
        # Drum enhancer
        if self.config.enable_drums:
            drum_config = DrumConfig(
                intensity=self.config.drum_intensity,
                transient_shaping=0.3,
                compression=True
            )
            self._enhancers['drums'] = DrumEnhancer(drum_config)
        
        # Bass enhancer
        if self.config.enable_bass:
            bass_config = BassConfig(
                intensity=self.config.bass_intensity,
                harmonic_excitation=True,
                compression=True
            )
            self._enhancers['bass'] = BassEnhancer(bass_config)
        
        # Instrument/other enhancer
        if self.config.enable_other:
            instrument_config = InstrumentConfig(
                intensity=self.config.other_intensity,
                harmonic_excitation=True,
                compression=True
            )
            self._enhancers['other'] = InstrumentEnhancer(instrument_config)
    
    def process(self, stems: Dict[str, np.ndarray], 
                sample_rate: int) -> StemEnhancementResult:
        """
        Process all stems through their respective enhancers.
        
        Args:
            stems: Dictionary of stem name -> audio array
            sample_rate: Sample rate
            
        Returns:
            StemEnhancementResult with processed stems
        """
        import time
        start_time = time.time()
        
        enhanced_stems = {}
        original_stems = {}
        all_enhancements: Dict[str, List[str]] = {}
        
        logger.info(f"Processing {len(stems)} stems...")
        
        # Process each stem
        for stem_name, audio in stems.items():
            if stem_name not in self._enhancers:
                logger.warning(f"No enhancer for stem: {stem_name}, skipping")
                continue
            
            enhancer = self._enhancers[stem_name]
            
            logger.debug(f"Processing {stem_name}...")
            
            # Process stem
            result = enhancer.enhance(audio, sample_rate)
            
            enhanced_stems[stem_name] = result.audio
            original_stems[stem_name] = result.original_audio
            all_enhancements[stem_name] = result.enhancements_applied
            
            logger.debug(f"  {stem_name}: {result.enhancements_applied}")
        
        processing_time = time.time() - start_time
        logger.info(f"✅ Enhancement complete in {processing_time:.1f}s")
        
        return StemEnhancementResult(
            stems=enhanced_stems,
            sample_rate=sample_rate,
            enhancements_applied=all_enhancements,
            original_stems=original_stems,
            processing_time=processing_time
        )
    
    def process_single(self, stem_name: str, audio: np.ndarray,
                      sample_rate: int) -> EnhancementResult:
        """
        Process a single stem.
        
        Args:
            stem_name: Name of stem to process
            audio: Audio array for the stem
            sample_rate: Sample rate
            
        Returns:
            EnhancementResult for the stem
        """
        if stem_name not in self._enhancers:
            raise ValueError(f"No enhancer for stem: {stem_name}")
        
        enhancer = self._enhancers[stem_name]
        return enhancer.enhance(audio, sample_rate)
    
    def get_enhancer_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all configured enhancers."""
        return {name: enhancer.get_info() 
                for name, enhancer in self._enhancers.items()}
    
    def update_intensity(self, stem_name: str, intensity: float):
        """
        Update intensity for a specific stem.
        
        Args:
            stem_name: Name of stem to update
            intensity: New intensity value (0-1)
        """
        if stem_name not in self._enhancers:
            raise ValueError(f"No enhancer for stem: {stem_name}")
        
        enhancer = self._enhancers[stem_name]
        enhancer.config.intensity = max(0, min(1, intensity))
        logger.info(f"Updated {stem_name} intensity to {intensity:.2f}")
    
    def set_mode(self, mode: ProcessingMode):
        """
        Set processing mode and update all enhancers.
        
        Args:
            mode: Processing mode (preview/render)
        """
        self.config.mode = mode
        
        # Adjust intensities based on mode
        intensity_mult = 0.7 if mode == ProcessingMode.PREVIEW else 1.0
        
        for name, enhancer in self._enhancers.items():
            original_intensity = getattr(self.config, f'{name}_intensity', None)
            if original_intensity is not None:
                enhancer.config.intensity = original_intensity * intensity_mult
        
        logger.info(f"Set mode to {mode.value}")
    
    def __repr__(self) -> str:
        enabled_stems = [name for name in self.STEM_NAMES 
                        if name in self._enhancers]
        return (f"StemEnhancementPipeline(stems={enabled_stems}, "
                f"mode={self.config.mode.value})")


# Convenience function
def create_enhancement_pipeline(mode: str = "render",
                                intensity: float = 0.5) -> StemEnhancementPipeline:
    """
    Create enhancement pipeline with specified settings.
    
    Args:
        mode: "preview" or "render"
        intensity: Overall enhancement intensity (0-1)
        
    Returns:
        Configured StemEnhancementPipeline
    """
    mode_enum = ProcessingMode.RENDER if mode == "render" else ProcessingMode.PREVIEW
    
    config = GlobalEnhancementConfig(
        mode=mode_enum,
        intensity=intensity
    )
    
    return StemEnhancementPipeline(config)


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing StemEnhancementPipeline...")
    
    # Create test stems
    sample_rate = 44100
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Simulated stems (simple tones)
    stems = {
        'vocals': 0.5 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.randn(len(t)),
        'drums': 0.6 * np.sin(2 * np.pi * 50 * t) * np.exp(-t) + 0.1 * np.random.randn(len(t)),
        'bass': 0.5 * np.sin(2 * np.pi * 80 * t) + 0.05 * np.random.randn(len(t)),
        'other': 0.4 * np.sin(2 * np.pi * 330 * t) + 0.05 * np.random.randn(len(t))
    }
    
    print(f"Created {len(stems)} test stems")
    
    # Create pipeline
    pipeline = create_enhancement_pipeline(mode="preview", intensity=0.5)
    print(f"Created: {pipeline}")
    
    # Get enhancer info
    info = pipeline.get_enhancer_info()
    print("Enhancer configurations:")
    for name, data in info.items():
        print(f"  {name}: intensity={data['config']['intensity']}")
    
    # Process all stems
    result = pipeline.process(stems, sample_rate)
    print(f"✅ Processing complete in {result.processing_time:.2f}s")
    print(f"  Enhanced stems: {list(result.stems.keys())}")
    print(f"  Total enhancements: {sum(len(v) for v in result.enhancements_applied.values())}")
    
    # Test single stem processing
    vocal_result = pipeline.process_single('vocals', stems['vocals'], sample_rate)
    print(f"Single vocal processing: {vocal_result.enhancements_applied}")
    
    # Update intensity and re-test
    pipeline.update_intensity('vocals', 0.8)
    print("Updated vocal intensity to 0.8")
