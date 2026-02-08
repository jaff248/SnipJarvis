"""
Base Enhancer Class - Foundation for audio stem enhancement

Defines the interface and common functionality for all stem-specific
enhancers in the audio processing pipeline.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import dataclasses
from typing import Dict, Any, Optional, List
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class EnhancementType(Enum):
    """Types of enhancement operations."""
    NOISE_REDUCTION = "noise_reduction"
    EQ = "eq"
    COMPRESSION = "compression"
    HARMONIC_EXCITATION = "harmonic_excitation"
    CLARITY = "clarity"
    TRANSIENT = "transient"
    DE_ESSING = "de_essing"
    DE_REVERB = "dereverb"
    DISTORTION = "distortion"


@dataclass
class EnhancementConfig:
    """Configuration for audio enhancement."""
    # Overall intensity (0.0 to 1.0)
    intensity: float = 0.5
    
    # Enable/disable specific enhancements
    noise_reduction: bool = True
    eq: bool = True
    compression: bool = True
    
    # Advanced options
    bypass: bool = False
    wet_dry_mix: float = 1.0  # 1.0 = fully processed, 0.0 = original
    
    def __post_init__(self):
        """Validate configuration values after initialization."""
        self.validate()
    
    def validate(self):
        """Validate configuration values."""
        if not 0.0 <= self.intensity <= 1.0:
            raise ValueError(f"Intensity must be 0-1, got {self.intensity}")
        if not 0.0 <= self.wet_dry_mix <= 1.0:
            raise ValueError(f"Wet/dry mix must be 0-1, got {self.wet_dry_mix}")


@dataclass
class EnhancementResult:
    """Result of enhancement processing."""
    audio: np.ndarray
    sample_rate: int
    enhancements_applied: List[str] = field(default_factory=list)
    original_audio: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Store original if not provided."""
        if self.original_audio is None:
            self.original_audio = self.audio.copy()


class BaseStemEnhancer(ABC):
    """
    Base class for stem-specific audio enhancement.
    
    Defines the interface that all stem enhancers must implement.
    Each stem (vocals, drums, bass, other) has specialized processing.
    """
    
    # Stem name (override in subclasses)
    STEM_NAME = "base"
    
    # Default enhancement configuration
    DEFAULT_CONFIG = EnhancementConfig(
        intensity=0.5,
        noise_reduction=True,
        eq=True,
        compression=True
    )
    
    def __init__(self, config: EnhancementConfig = None):
        """
        Initialize enhancer.
        
        Args:
            config: Enhancement configuration (uses defaults if None)
        """
        self.config = config or dataclasses.replace(self.DEFAULT_CONFIG)
        self.config.validate()
        
        logger.info(f"Initialized {self.STEM_NAME} enhancer with intensity={self.config.intensity}")
    
    @property
    def stem_name(self) -> str:
        """Get stem name."""
        return self.STEM_NAME
    
    @abstractmethod
    def enhance(self, audio: np.ndarray, sample_rate: int) -> EnhancementResult:
        """
        Apply enhancement to audio stem.
        
        Args:
            audio: Input audio (float32, mono)
            sample_rate: Sample rate
            
        Returns:
            EnhancementResult with processed audio
        """
        pass
    
    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Process audio and return just the enhanced audio.
        
        Args:
            audio: Input audio
            sample_rate: Sample rate
            
        Returns:
            Enhanced audio array
        """
        result = self.enhance(audio, sample_rate)
        return result.audio
    
    def _apply_wet_dry(self, original: np.ndarray, processed: np.ndarray) -> np.ndarray:
        """Apply wet/dry mix to blend original and processed audio."""
        if self.config.wet_dry_mix == 1.0:
            return processed
        elif self.config.wet_dry_mix == 0.0:
            return original
        else:
            return (self.config.wet_dry_mix * processed + 
                   (1.0 - self.config.wet_dry_mix) * original)
    
    def _check_bypass(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """Check if processing should be bypassed."""
        if self.config.bypass:
            logger.debug(f"Bypassing {self.stem_name} enhancement")
            return audio
        return None
    
    def __repr__(self) -> str:
        return f"{self.STEM_NAME.title()}Enhancer(intensity={self.config.intensity})"
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about the enhancer."""
        return {
            "stem_name": self.stem_name,
            "config": {
                "intensity": self.config.intensity,
                "noise_reduction": self.config.noise_reduction,
                "eq": self.config.eq,
                "compression": self.config.compression,
                "bypass": self.config.bypass,
                "wet_dry_mix": self.config.wet_dry_mix
            }
        }


class EnhancementPipeline:
    """
    Chain multiple enhancement processors together.
    
    Allows applying multiple effects in sequence with
    proper gain staging and validation.
    """
    
    def __init__(self):
        """Initialize empty enhancement pipeline."""
        self.processors: List[BaseStemEnhancer] = []
    
    def add(self, processor: BaseStemEnhancer) -> 'EnhancementPipeline':
        """
        Add processor to pipeline.
        
        Args:
            processor: Enhancer to add
            
        Returns:
            Self for chaining
        """
        self.processors.append(processor)
        logger.debug(f"Added {processor.stem_name} to pipeline")
        return self
    
    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Process audio through all processors.
        
        Args:
            audio: Input audio
            sample_rate: Sample rate
            
        Returns:
            Processed audio
        """
        processed = audio.copy()
        
        for processor in self.processors:
            logger.debug(f"Applying {processor.stem_name} enhancement...")
            processed = processor.process(processed, sample_rate)
            
            # Validate audio after each processor
            if np.max(np.abs(processed)) > 2.0:
                logger.warning(f"Clipping detected after {processor.stem_name}")
                processed = np.clip(processed, -1.0, 1.0)
        
        return processed
    
    def __len__(self) -> int:
        """Get number of processors in pipeline."""
        return len(self.processors)
    
    def __repr__(self) -> str:
        return f"EnhancementPipeline({len(self.processors)} processors)"


# Factory function for creating enhancers
def create_enhancer(stem_type: str, intensity: float = 0.5) -> BaseStemEnhancer:
    """
    Create enhancer for specified stem type.
    
    Args:
        stem_type: Type of stem ("vocals", "drums", "bass", "other")
        intensity: Enhancement intensity (0.0-1.0)
        
    Returns:
        Configured stem enhancer
        
    Raises:
        ValueError: If stem_type is unknown
    """
    from .vocal import VocalEnhancer
    from .drums import DrumEnhancer
    from .bass import BassEnhancer
    from .instruments import InstrumentEnhancer
    
    stem_map = {
        "vocals": VocalEnhancer,
        "drums": DrumEnhancer,
        "bass": BassEnhancer,
        "other": InstrumentEnhancer,
        "instruments": InstrumentEnhancer
    }
    
    if stem_type not in stem_map:
        raise ValueError(f"Unknown stem type: {stem_type}. "
                        f"Available: {list(stem_map.keys())}")
    
    config = EnhancementConfig(intensity=intensity)
    return stem_map[stem_type](config)


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test base enhancer
    print("Testing BaseEnhancer infrastructure...")
    
    # Create test audio
    sample_rate = 44100
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    test_audio = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.randn(len(t))
    
    # Create pipeline
    pipeline = EnhancementPipeline()
    print(f"Created: {pipeline}")
    
    # Check pipeline length
    print(f"Processors in pipeline: {len(pipeline)}")
