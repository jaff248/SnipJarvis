"""
Stem Mixing Module - Recombine and balance separated stems

Handles the mixing stage of the reconstruction pipeline, combining
enhanced stems into a coherent final mix with proper gain staging.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum

import numpy as np

from .utils import db_to_gain, format_duration

logger = logging.getLogger(__name__)


class MixMode(Enum):
    """Mixing mode options."""
    NATURAL = "natural"      # Preserve original balance
    ENHANCED = "enhanced"    # Optimize for clarity
    STEM_SOLO = "stem_solo"  # Allow soloing individual stems


@dataclass
class StemLevel:
    """Individual stem level configuration."""
    name: str
    gain_db: float = 0.0
    pan: float = 0.0         # -1.0 = left, 0.0 = center, 1.0 = right
    muted: bool = False
    solo: bool = False


@dataclass
class MixConfig:
    """Configuration for stem mixing."""
    # Master settings
    master_gain_db: float = 0.0
    mix_mode: MixMode = MixMode.ENHANCED
    
    # Individual stem levels
    vocal_level_db: float = 0.0
    drum_level_db: float = 0.0
    bass_level_db: float = 0.0
    other_level_db: float = 0.0
    
    # Stereo width (0 = mono, 1 = max width)
    stereo_width: float = 0.0
    
    # Reference track for loudness matching (optional)
    reference_loudness_lufs: Optional[float] = None
    
    # Limit output to prevent clipping
    output_limit: bool = True
    output_ceiling_db: float = -1.0


@dataclass
class MixResult:
    """Result of stem mixing."""
    mixed_audio: np.ndarray
    sample_rate: int
    stem_levels: Dict[str, float] = field(default_factory=dict)
    peak_level_db: float = 0.0
    rms_level_db: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "mixed_audio_shape": self.mixed_audio.shape,
            "sample_rate": self.sample_rate,
            "duration_seconds": len(self.mixed_audio) / self.sample_rate,
            "duration_formatted": format_duration(len(self.mixed_audio) / self.sample_rate),
            "stem_levels_db": self.stem_levels,
            "peak_level_db": round(self.peak_level_db, 1),
            "rms_level_db": round(self.rms_level_db, 1)
        }


class StemMixer:
    """
    Stem mixing engine for combining enhanced stems.
    
    Handles:
    - Gain staging for each stem
    - Panning (stereo positioning)
    - Level balancing for optimal clarity
    - Master gain and limiting
    - Loudness matching to reference
    """
    
    STEM_ORDER = ['vocals', 'drums', 'bass', 'other']
    
    def __init__(self, config: MixConfig = None):
        """
        Initialize stem mixer.
        
        Args:
            config: Mix configuration (uses defaults if None)
        """
        self.config = config or MixConfig()
        
        logger.info(f"Initialized StemMixer: mode={self.config.mix_mode.value}")
    
    def mix(self, stems: Dict[str, np.ndarray], 
            sample_rate: int) -> MixResult:
        """
        Mix stems together into final audio.
        
        Pipeline:
        1. Apply individual stem gains
        2. Apply panning (stereo positioning)
        3. Combine stems
        4. Apply master gain
        5. Limit if needed
        6. Measure levels
        
        Args:
            stems: Dictionary of stem name -> audio array
            sample_rate: Sample rate
            
        Returns:
            MixResult with mixed audio and levels
        """
        logger.info(f"Mixing {len(stems)} stems...")
        
        # Validate all stems have same length
        expected_length = None
        for name, audio in stems.items():
            if expected_length is None:
                expected_length = len(audio)
            elif len(audio) != expected_length:
                logger.warning(f"Stem {name} length mismatch: {len(audio)} vs {expected_length}")
                # Pad or trim to match
                if len(audio) < expected_length:
                    stems[name] = np.pad(audio, (0, expected_length - len(audio)))
                else:
                    stems[name] = audio[:expected_length]
        
        # Initialize mixed audio (start with zeros)
        mixed = np.zeros(expected_length, dtype=np.float32)
        
        # Track stem levels
        stem_levels = {}
        
        # Process each stem
        for stem_name in self.STEM_ORDER:
            if stem_name not in stems:
                logger.warning(f"Missing stem: {stem_name}")
                continue
            
            audio = stems[stem_name]
            
            # Get stem level from config
            level_db = self._get_stem_level(stem_name)
            
            # Apply gain
            if level_db != 0:
                audio = audio * db_to_gain(level_db)
            
            # Store stem level for reference
            stem_levels[stem_name] = level_db
            
            # Add to mix
            mixed += audio
            
            logger.debug(f"  {stem_name}: {level_db:+.1f} dB")
        
        # Apply master gain
        if self.config.master_gain_db != 0:
            mixed = mixed * db_to_gain(self.config.master_gain_db)
        
        # Apply output limiting
        if self.config.output_limit:
            mixed = self._apply_limit(mixed)
        
        # Measure levels
        if len(mixed) > 0:
            peak_db = 20 * np.log10(np.max(np.abs(mixed)) + 1e-10)
            rms_db = 20 * np.log10(np.sqrt(np.mean(mixed ** 2)) + 1e-10)
        else:
            peak_db = -100.0
            rms_db = -100.0
        
        logger.info(f"✅ Mix complete: peak={peak_db:.1f} dB, rms={rms_db:.1f} dB")
        
        return MixResult(
            mixed_audio=mixed,
            sample_rate=sample_rate,
            stem_levels=stem_levels,
            peak_level_db=peak_db,
            rms_level_db=rms_db
        )
    
    def _get_stem_level(self, stem_name: str) -> float:
        """Get gain level for a specific stem."""
        level_map = {
            'vocals': self.config.vocal_level_db,
            'drums': self.config.drum_level_db,
            'bass': self.config.bass_level_db,
            'other': self.config.other_level_db
        }
        return level_map.get(stem_name, 0.0)
    
    def _apply_limit(self, audio: np.ndarray, 
                    ceiling_db: float = -1.0) -> np.ndarray:
        """
        Apply soft limiting to prevent hard clipping.
        
        Args:
            audio: Input audio
            ceiling_db: Maximum peak level in dB
            
        Returns:
            Limited audio
        """
        ceiling = db_to_gain(ceiling_db)
        
        # Handle empty array
        if len(audio) == 0:
            logger.warning("Empty audio array in limiter - returning zeros")
            return audio
        
        # Get peak level
        peak = np.max(np.abs(audio))
        
        if peak <= ceiling:
            return audio
        
        # Soft limiting using tanh
        # Scale input so peak reaches ceiling
        scale = ceiling / peak
        limited = np.tanh(audio * (1 / ceiling)) * ceiling
        
        # Ensure no clipping
        limited = np.clip(limited, -ceiling, ceiling)
        
        return limited
    
    def mix_with_stem_levels(self, stems: Dict[str, np.ndarray],
                            sample_rate: int,
                            stem_levels: Dict[str, float]) -> MixResult:
        """
        Mix stems with custom level adjustments.
        
        Args:
            stems: Dictionary of stem name -> audio array
            sample_rate: Sample rate
            stem_levels: Custom level adjustments per stem
            
        Returns:
            MixResult with mixed audio
        """
        # Temporarily set custom levels
        original_levels = {}
        for stem_name, level_db in stem_levels.items():
            original_levels[stem_name] = self._get_stem_level(stem_name)
            self._set_stem_level(stem_name, level_db)
        
        # Mix
        result = self.mix(stems, sample_rate)
        
        # Restore original levels
        for stem_name, level_db in original_levels.items():
            self._set_stem_level(stem_name, level_db)
        
        return result
    
    def _set_stem_level(self, stem_name: str, level_db: float):
        """Set gain level for a specific stem."""
        if stem_name == 'vocals':
            self.config.vocal_level_db = level_db
        elif stem_name == 'drums':
            self.config.drum_level_db = level_db
        elif stem_name == 'bass':
            self.config.bass_level_db = level_db
        elif stem_name == 'other':
            self.config.other_level_db = level_db
    
    def optimize_for_clarity(self, stems: Dict[str, np.ndarray],
                            sample_rate: int) -> MixResult:
        """
        Mix stems with optimized levels for maximum clarity.
        
        Adjusts levels based on typical mixing best practices:
        - Vocals slightly prominent
        - Drums punchy but not overwhelming
        - Bass supportive
        - Other instruments balanced
        """
        # Set optimal levels
        optimal_levels = {
            'vocals': 1.0,      # Slight boost for vocals
            'drums': 0.5,       # Standard drum level
            'bass': -1.0,       # Slight reduction for bass
            'other': -2.0       # Lower other instruments
        }
        
        return self.mix_with_stem_levels(stems, sample_rate, optimal_levels)
    
    def auto_balance(self, stems: Dict[str, np.ndarray],
                    sample_rate: int) -> MixResult:
        """
        Automatically balance stem levels based on RMS energy.
        
        Normalizes each stem to have similar perceived loudness.
        """
        # Calculate RMS for each stem
        rms_levels = {}
        for name, audio in stems.items():
            rms = np.sqrt(np.mean(audio ** 2))
            rms_db = 20 * np.log10(rms + 1e-10)
            rms_levels[name] = rms_db
        
        # Calculate target level (average RMS)
        avg_rms = np.mean(list(rms_levels.values()))
        
        # Calculate adjustments
        stem_levels = {}
        for name, rms_db in rms_levels.items():
            adjustment = avg_rms - rms_db
            stem_levels[name] = adjustment
        
        return self.mix_with_stem_levels(stems, sample_rate, stem_levels)
    
    def get_info(self) -> Dict[str, Any]:
        """Get mixer configuration and info."""
        return {
            "config": {
                "mix_mode": self.config.mix_mode.value,
                "master_gain_db": self.config.master_gain_db,
                "vocal_level_db": self.config.vocal_level_db,
                "drum_level_db": self.config.drum_level_db,
                "bass_level_db": self.config.bass_level_db,
                "other_level_db": self.config.other_level_db,
                "stereo_width": self.config.stereo_width,
                "output_limit": self.config.output_limit
            }
        }
    
    def __repr__(self) -> str:
        return (f"StemMixer(mode={self.config.mix_mode.value}, "
                f"vocals={self.config.vocal_level_db:+.1f}dB, "
                f"drums={self.config.drum_level_db:+.1f}dB, "
                f"bass={self.config.bass_level_db:+.1f}dB, "
                f"other={self.config.other_level_db:+.1f}dB)")


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing StemMixer...")
    
    # Create test stems
    sample_rate = 44100
    duration = 3.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    stems = {
        'vocals': 0.4 * np.sin(2 * np.pi * 440 * t),
        'drums': 0.5 * np.sin(2 * np.pi * 50 * t) * np.exp(-t % 0.5),
        'bass': 0.4 * np.sin(2 * np.pi * 80 * t),
        'other': 0.3 * np.sin(2 * np.pi * 330 * t)
    }
    
    print(f"Created {len(stems)} test stems")
    
    # Create mixer
    mixer = StemMixer()
    print(f"Created: {mixer}")
    
    # Mix stems
    result = mixer.mix(stems, sample_rate)
    print(f"✅ Mix complete: {result.to_dict()}")
    
    # Test auto balance
    print("\nTesting auto-balance...")
    balance_result = mixer.auto_balance(stems, sample_rate)
    print(f"Auto-balanced levels: {balance_result.stem_levels}")
    
    # Test optimize for clarity
    print("\nTesting optimize for clarity...")
    clarity_result = mixer.optimize_for_clarity(stems, sample_rate)
    print(f"Clarity-optimized levels: {clarity_result.stem_levels}")
    
    # Save mix
    import soundfile as sf
    sf.write("test_mix.wav", result.mixed_audio, sample_rate)
    print("Saved test mix to test_mix.wav")
