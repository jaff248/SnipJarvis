"""
Instrument Enhancer - General instrument audio enhancement for "other" stem

Provides harmonic excitation, clarity enhancement, and spectral shaping
optimized for guitars, keyboards, and other instruments in live recordings.
"""

import logging
import dataclasses
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

import numpy as np
from scipy import signal

from .base import BaseStemEnhancer, EnhancementConfig, EnhancementResult
from ..utils import db_to_gain

logger = logging.getLogger(__name__)


@dataclass
class InstrumentConfig(EnhancementConfig):
    """Instrument-specific enhancement configuration."""
    # Harmonic excitation
    harmonic_excitation: bool = True
    harmonic_amount: float = 0.25      # Intensity of harmonics (0-1)
    harmonic_mix: float = 0.3          # Mix of harmonics with original
    
    # Clarity and presence
    clarity_boost_db: float = 2.0      # Boost at 2-5 kHz
    clarity_freq: float = 4000.0
    
    # De-mudding (remove 200-300 Hz boominess)
    demud_amount: float = -1.5
    
    # High-frequency air
    air_boost_db: float = 1.5          # Boost above 10 kHz
    air_freq: float = 12000.0
    
    # EQ adjustments
    low_shelf_db: float = 0.0          # Low frequency adjustment
    low_shelf_freq: float = 200.0
    
    # Stereo width (for stereo files)
    stereo_width: float = 0.0          # 0 = mono, 1 = max width
    
    # Compression
    compression: bool = True
    compressor_threshold_db: float = -20.0
    compressor_ratio: float = 3.0
    compressor_attack_ms: float = 5.0
    compressor_release_ms: float = 100.0
    
    def __post_init__(self):
        """Validate instrument-specific settings."""
        super().__post_init__()
        if not 0 <= self.harmonic_amount <= 1:
            raise ValueError("Harmonic amount must be 0-1")


class InstrumentEnhancer(BaseStemEnhancer):
    """
    Instrument stem enhancer for "other" category.
    
    Handles guitars, keyboards, synths, and other melodic instruments.
    This is a general-purpose enhancer that adds warmth and clarity.
    
    Optimization goals:
    - Add harmonic content for warmth
    - Enhance clarity and presence
    - Remove muddiness in low-mids
    - Add "air" to high frequencies
    - Control dynamics gently
    - Preserve instrument character
    """
    
    STEM_NAME = "other"
    DEFAULT_CONFIG = InstrumentConfig(
        intensity=0.5,
        harmonic_excitation=True,
        harmonic_amount=0.25,
        harmonic_mix=0.3,
        clarity_boost_db=2.0,
        clarity_freq=4000.0,
        demud_amount=-1.5,
        air_boost_db=1.5,
        air_freq=12000.0,
        low_shelf_db=0.0,
        low_shelf_freq=200.0,
        stereo_width=0.0,
        compression=True,
        compressor_threshold_db=-20.0,
        compressor_ratio=3.0,
        compressor_attack_ms=5.0,
        compressor_release_ms=100.0
    )
    
    def __init__(self, config: InstrumentConfig = None):
        """Initialize instrument enhancer."""
        super().__init__(config)
    
    def enhance(self, audio: np.ndarray, sample_rate: int) -> EnhancementResult:
        """
        Apply instrument-specific enhancement.
        
        Pipeline:
        1. De-mudding (cut 200-300 Hz if muddy)
        2. Low shelf adjustment
        3. Harmonic excitation (adds warmth)
        4. Clarity boost (2-5 kHz)
        5. Air boost (10+ kHz)
        6. Gentle compression
        7. Optional stereo widening
        
        Args:
            audio: Input instrument audio
            sample_rate: Sample rate
            
        Returns:
            EnhancementResult with processed instruments
        """
        # Check bypass
        if bypass_result := self._check_bypass(audio):
            return EnhancementResult(
                audio=bypass_result,
                sample_rate=sample_rate,
                enhancements_applied=["bypass"]
            )
        
        processed = audio.copy()
        enhancements: List[str] = []
        
        # Apply intensity scaling
        intensity = self.config.intensity
        
        # 1. De-mudding
        if self.config.demud_amount < 0:
            cut_db = self.config.demud_amount * intensity
            processed = self._demud(processed, sample_rate, cut_db)
            enhancements.append("demudding")
        
        # 2. Low shelf adjustment
        if self.config.low_shelf_db != 0:
            shelf_db = self.config.low_shelf_db * intensity
            processed = self._low_shelf(processed, sample_rate,
                                       self.config.low_shelf_freq, shelf_db)
            enhancements.append("low_shelf")
        
        # 3. Harmonic excitation
        if self.config.harmonic_excitation and self.config.harmonic_amount > 0:
            processed = self._harmonic_excitation(processed, sample_rate,
                                                 self.config.harmonic_amount * intensity,
                                                 self.config.harmonic_mix)
            enhancements.append("harmonic_excitation")
        
        # 4. Clarity boost
        if self.config.clarity_boost_db != 0:
            clarity_db = self.config.clarity_boost_db * intensity
            processed = self._clarity_boost(processed, sample_rate,
                                           self.config.clarity_freq, clarity_db)
            enhancements.append("clarity")
        
        # 5. Air boost
        if self.config.air_boost_db != 0:
            air_db = self.config.air_boost_db * intensity
            processed = self._air_boost(processed, sample_rate,
                                       self.config.air_freq, air_db)
            enhancements.append("air")
        
        # 6. Compression
        if self.config.compression:
            processed = self._compression(processed, sample_rate)
            enhancements.append("compression")
        
        # Apply wet/dry mix
        processed = self._apply_wet_dry(audio, processed)
        if self.config.wet_dry_mix < 1.0:
            enhancements.append("wet_dry_mix")
        
        # Ensure proper range
        processed = np.clip(processed, -1.0, 1.0)
        
        return EnhancementResult(
            audio=processed,
            sample_rate=sample_rate,
            enhancements_applied=enhancements,
            original_audio=audio
        )
    
    def _demud(self, audio: np.ndarray, sample_rate: int,
              cut_db: float) -> np.ndarray:
        """Cut muddy frequencies (200-300 Hz range)."""
        if cut_db >= 0:
            return audio
        
        nyquist = sample_rate / 2
        
        # Center frequency for mud cut
        mud_freq = 250
        Q = 1.5
        
        normalized_freq = mud_freq / nyquist
        normalized_freq = min(max(normalized_freq, 1e-3), 0.99)
        gain = db_to_gain(cut_db)
        
        b, a = signal.iirpeak(normalized_freq, Q)
        filtered = signal.filtfilt(b, a, audio)
        
        # Blend
        return audio + (filtered * (gain - 1.0)) * 0.5
    
    def _low_shelf(self, audio: np.ndarray, sample_rate: int,
                  freq: float, shelf_db: float) -> np.ndarray:
        """Apply low shelf EQ."""
        nyquist = sample_rate / 2
        normalized_freq = min(freq / nyquist, 0.99)
        normalized_freq = max(normalized_freq, 1e-3)
        
        # Simple shelf using Butterworth
        if shelf_db > 0:
            b, a = signal.butter(2, normalized_freq, btype='low')
        else:
            b, a = signal.butter(2, normalized_freq, btype='high')
        
        filtered = signal.filtfilt(b, a, audio)
        
        # Mix based on shelf direction
        if shelf_db > 0:
            # Boost low frequencies
            boosted = audio + (filtered - audio) * (shelf_db / 6.0)
        else:
            # Cut low frequencies
            boosted = audio + (filtered - audio) * (abs(shelf_db) / 6.0)
        
        return boosted
    
    def _harmonic_excitation(self, audio: np.ndarray, sample_rate: int,
                            amount: float, mix: float) -> np.ndarray:
        """
        Add harmonic content for warmth and presence.
        
        Generates even harmonics (2nd, 3rd, 4th) for musical enhancement.
        """
        # Generate harmonics
        harmonics = np.zeros_like(audio)
        
        # Clip input for harmonic generation
        clipped = np.clip(audio * 2, -1, 1)
        
        # Even harmonics (warmth)
        harmonics += 0.5 * (np.sign(clipped) * np.abs(clipped) ** 1.5)
        
        # Third harmonic (presence)
        harmonics += 0.2 * (np.sign(clipped) * np.abs(clipped) ** 1.8)
        
        # Fourth harmonic (brilliance)
        harmonics += 0.1 * (np.sign(clipped) * np.abs(clipped) ** 2.0)
        
        # Mix harmonics with original
        excited = audio + harmonics * amount * mix
        
        return excited
    
    def _clarity_boost(self, audio: np.ndarray, sample_rate: int,
                      freq: float, boost_db: float) -> np.ndarray:
        """Boost presence frequencies for clarity."""
        if boost_db <= 0:
            return audio
            
        nyquist = sample_rate / 2
        normalized_freq = min(freq / nyquist, 0.99)
        
        # Peaking filter
        Q = 3.0
        gain = db_to_gain(boost_db)
        
        b, a = signal.iirpeak(normalized_freq, Q)
        filtered = signal.filtfilt(b, a, audio)
        
        # Add presence
        boosted = audio + (filtered * (gain - 1.0)) * 0.4
        
        return boosted
    
    def _air_boost(self, audio: np.ndarray, sample_rate: int,
                  freq: float, boost_db: float) -> np.ndarray:
        """Boost high frequencies for air and brilliance."""
        if boost_db <= 0:
            return audio
            
        nyquist = sample_rate / 2
        normalized_freq = min(freq / nyquist, 0.99)
        
        # High shelf using high-pass
        b, a = signal.butter(2, normalized_freq, btype='high')
        high_content = signal.filtfilt(b, a, audio)
        
        # Add high frequencies with boost
        boost_factor = db_to_gain(boost_db)
        boosted = audio + high_content * (boost_factor - 1) * 0.3
        
        return boosted
    
    def _compression(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply gentle compression for consistency."""
        threshold = db_to_gain(self.config.compressor_threshold_db)
        ratio = self.config.compressor_ratio
        
        attack_samples = int(self.config.compressor_attack_ms * sample_rate / 1000)
        release_samples = int(self.config.compressor_release_ms * sample_rate / 1000)
        
        # Calculate envelope
        envelope = np.abs(audio)
        
        # Smoothing
        alpha = 0.3
        envelope_smoothed = np.zeros_like(envelope)
        for i in range(len(envelope)):
            if i == 0:
                envelope_smoothed[i] = envelope[i]
            else:
                envelope_smoothed[i] = alpha * envelope[i] + (1 - alpha) * envelope_smoothed[i-1]
        
        # Calculate gain reduction
        gain_reduction = np.ones_like(audio)
        above_threshold = envelope_smoothed > threshold
        
        if np.any(above_threshold):
            overs = envelope_smoothed[above_threshold] - threshold
            reduction = 1 - (overs / envelope_smoothed[above_threshold]) * (1 - 1/ratio)
            reduction = np.clip(reduction, 0.6, 1.0)
            gain_reduction[above_threshold] = reduction
        
        # Apply compression
        compressed = audio * gain_reduction
        
        # Makeup gain
        makeup_db = 1.5
        compressed *= db_to_gain(makeup_db)
        
        return compressed
    
    def get_info(self) -> Dict[str, Any]:
        """Get instrument enhancer information."""
        info = super().get_info()
        info["config"].update({
            "harmonic_excitation": self.config.harmonic_excitation,
            "harmonic_amount": self.config.harmonic_amount,
            "clarity_boost_db": self.config.clarity_boost_db,
            "demud_amount": self.config.demud_amount,
            "air_boost_db": self.config.air_boost_db,
            "low_shelf_db": self.config.low_shelf_db,
            "compression": self.config.compression
        })
        return info


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing InstrumentEnhancer...")
    
    # Create test audio (simulated guitar)
    sample_rate = 44100
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Guitar-like sound with harmonics
    guitar = (
        0.4 * np.sin(2 * np.pi * 330 * t) +  # E4
        0.3 * np.sin(2 * np.pi * 440 * t) +  # A4
        0.2 * np.sin(2 * np.pi * 554 * t) +  # C#5
        0.15 * np.sin(2 * np.pi * 659 * t)   # E5
    )
    
    # Add some strumming envelope
    strum = np.sin(np.pi * t / duration)
    guitar *= strum
    
    # Add some "mud"
    guitar += 0.1 * np.sin(2 * np.pi * 250 * t)
    
    # Add noise
    guitar += 0.01 * np.random.randn(len(guitar))
    
    print(f"Input audio: {guitar.shape}, range=[{guitar.min():.3f}, {guitar.max():.3f}]")
    
    # Create enhancer
    enhancer = InstrumentEnhancer()
    print(f"Created: {enhancer}")
    
    # Process
    result = enhancer.enhance(guitar, sample_rate)
    print(f"Output audio: {result.audio.shape}")
    print(f"Enhancements: {result.enhancements_applied}")
    print(f"Output range: [{result.audio.min():.3f}, {result.audio.max():.3f}]")
    
    # Save test audio
    import soundfile as sf
    sf.write("test_instruments_before.wav", guitar, sample_rate)
    sf.write("test_instruments_after.wav", result.audio, sample_rate)
    print("Saved test audio files")
