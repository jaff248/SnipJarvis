"""
Drum Enhancer - Percussion-specific audio enhancement for drums

Provides transient shaping, clarity enhancement, and punch improvement
optimized for kick and snare in live music recordings.
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
class DrumConfig(EnhancementConfig):
    """Drum-specific enhancement configuration."""
    # Transient shaping
    transient_shaping: float = 0.3    # -1.0 = soften, +1.0 = sharpen
    
    # EQ settings
    kick_bass_boost_db: float = 4.0   # Boost below 80 Hz
    kick_bass_freq: float = 60.0
    snare_bodied_db: float = 2.0      # Boost 200-400 Hz for snare body
    snare_bodied_freq: float = 300.0
    high_freq_clarity_db: float = 2.0 # 5-10 kHz for transients
    
    # Compression
    punch_compression: bool = True
    punch_threshold_db: float = -20.0
    punch_ratio: float = 6.0
    punch_attack_ms: float = 1.0      # Fast attack for punch
    punch_release_ms: float = 50.0
    
    # Clarity
    clarity_boost: bool = True
    clarity_freq: float = 5000.0
    
    def __post_init__(self):
        """Validate drum-specific settings."""
        super().__post_init__()
        if not -1.0 <= self.transient_shaping <= 1.0:
            raise ValueError("Transient shaping must be -1 to 1")


class DrumEnhancer(BaseStemEnhancer):
    """
    Drum stem enhancer with specialized processing.
    
    Optimization goals:
    - Enhance transient attack for punch and clarity
    - Control low-end (especially kick)
    - Add body to snare and toms
    - Preserve natural drum sound character
    - Avoid over-processing that sounds artificial
    """
    
    STEM_NAME = "drums"
    DEFAULT_CONFIG = DrumConfig(
        intensity=0.5,
        transient_shaping=0.3,
        kick_bass_boost_db=4.0,
        kick_bass_freq=60.0,
        snare_bodied_db=2.0,
        snare_bodied_freq=300.0,
        high_freq_clarity_db=2.0,
        punch_compression=True,
        punch_threshold_db=-20.0,
        punch_ratio=6.0,
        punch_attack_ms=1.0,
        punch_release_ms=50.0,
        clarity_boost=True,
        clarity_freq=5000.0
    )
    
    def __init__(self, config: DrumConfig = None):
        """Initialize drum enhancer."""
        super().__init__(config)
    
    def enhance(self, audio: np.ndarray, sample_rate: int) -> EnhancementResult:
        """
        Apply drum-specific enhancement.
        
        Pipeline:
        1. High-pass filter (remove sub-bass rumble below 40 Hz)
        2. Transient shaping (enhance or soften attacks)
        3. EQ for kick (low-end boost) and snare (body + clarity)
        4. Punch compression (fast attack, medium release)
        5. High-frequency clarity boost
        
        Args:
            audio: Input drum audio
            sample_rate: Sample rate
            
        Returns:
            EnhancementResult with processed drums
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
        
        # 1. High-pass filter (remove sub-sonic)
        processed = self._highpass_filter(processed, sample_rate, 40)
        enhancements.append("subsonic_filter")
        
        # 2. Transient shaping
        if abs(self.config.transient_shaping) > 0.1:
            processed = self._transient_shaping(processed, sample_rate,
                                               self.config.transient_shaping * intensity)
            enhancements.append("transient_shaping")
        
        # 3. Kick enhancement (low-end boost)
        boost_db = self.config.kick_bass_boost_db * intensity
        processed = self._low_boost(processed, sample_rate,
                                   self.config.kick_bass_freq, boost_db)
        enhancements.append("kick_boost")
        
        # 4. Snare body enhancement
        body_db = self.config.snare_bodied_db * intensity
        processed = self._mid_boost(processed, sample_rate,
                                   self.config.snare_bodied_freq, body_db)
        enhancements.append("snare_body")
        
        # 5. Punch compression
        if self.config.punch_compression:
            processed = self._punch_compression(processed, sample_rate)
            enhancements.append("punch_compression")
        
        # 6. High-frequency clarity
        if self.config.clarity_boost:
            clarity_db = self.config.high_freq_clarity_db * intensity
            processed = self._high_boost(processed, sample_rate,
                                        self.config.clarity_freq, clarity_db)
            enhancements.append("clarity_boost")
        
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
    
    def _highpass_filter(self, audio: np.ndarray, sample_rate: int,
                        cutoff_freq: float) -> np.ndarray:
        """Remove sub-sonic frequencies."""
        nyquist = sample_rate / 2
        normalized_cutoff = cutoff_freq / nyquist
        
        # Fourth-order Butterworth for steeper roll-off
        b, a = signal.butter(4, normalized_cutoff, btype='high')
        return signal.filtfilt(b, a, audio)
    
    def _transient_shaping(self, audio: np.ndarray, sample_rate: int,
                          amount: float) -> np.ndarray:
        """
        Apply transient shaping.
        
        Positive = enhance attacks (sharpen)
        Negative = soften attacks (smooth)
        
        Implementation: Different envelope followers for attack/sustain.
        """
        # Get envelope
        envelope = np.abs(signal.hilbert(audio))
        
        # Attack and release coefficients
        attack_coeff = 0.1 if amount > 0 else 0.01
        release_coeff = 0.01 if amount > 0 else 0.1
        
        # Simple envelope follower
        follower = np.zeros_like(envelope)
        for i in range(1, len(envelope)):
            if envelope[i] > follower[i-1]:
                follower[i] = follower[i-1] + attack_coeff * (envelope[i] - follower[i-1])
            else:
                follower[i] = follower[i-1] + release_coeff * (envelope[i] - follower[i-1])
        
        # Normalize
        follower = follower / (np.max(follower) + 1e-8)
        
        # Apply transient shaping
        # Positive amount: increase transients (attack)
        # Negative amount: decrease transients (sustain)
        shaping = 1.0 + amount * 0.5
        
        # Modify audio based on envelope
        shaped = audio * (1 + (shaping - 1) * follower)
        
        return shaped
    
    def _low_boost(self, audio: np.ndarray, sample_rate: int,
                  freq: float, boost_db: float) -> np.ndarray:
        """Apply low-frequency boost (for kick)."""
        if boost_db <= 0:
            return audio
            
        nyquist = sample_rate / 2
        normalized_freq = freq / nyquist
        
        # Low-shelving filter
        Q = 0.5
        gain = boost_db
        
        b, a = signal.butter(2, normalized_freq, btype='low')
        
        # Apply filter and add to original for boost effect
        filtered = signal.filtfilt(b, a, audio)
        boosted = audio + filtered * db_to_gain(gain) * 0.5
        
        return boosted
    
    def _mid_boost(self, audio: np.ndarray, sample_rate: int,
                  freq: float, boost_db: float) -> np.ndarray:
        """Apply mid-frequency boost (for snare body)."""
        if boost_db <= 0:
            return audio
            
        nyquist = sample_rate / 2
        normalized_freq = freq / nyquist
        
        # Peaking filter
        Q = 2.0
        gain = boost_db
        
        b, a = signal.iirpeak(normalized_freq, Q, gain)
        boosted = signal.filtfilt(b, a, audio)
        
        return audio + (boosted - audio) * 0.5
    
    def _high_boost(self, audio: np.ndarray, sample_rate: int,
                   freq: float, boost_db: float) -> np.ndarray:
        """Apply high-frequency boost (for clarity/transients)."""
        if boost_db <= 0:
            return audio
            
        nyquist = sample_rate / 2
        normalized_freq = freq / nyquist
        
        # High-shelving or peak for clarity
        Q = 1.0
        gain = boost_db
        
        b, a = signal.butter(2, normalized_freq, btype='high')
        filtered = signal.filtfilt(b, a, audio)
        
        # Blend filtered high frequencies
        boosted = audio + filtered * db_to_gain(gain) * 0.3
        
        return boosted
    
    def _punch_compression(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply punchy compression for drums."""
        threshold = db_to_gain(self.config.punch_threshold_db)
        ratio = self.config.punch_ratio
        
        attack_samples = int(self.config.punch_attack_ms * sample_rate / 1000)
        release_samples = int(self.config.punch_release_ms * sample_rate / 1000)
        
        # Calculate envelope
        envelope = np.abs(audio)
        
        # Peak detection for fast attack
        envelope_smoothed = np.zeros_like(envelope)
        for i in range(len(envelope)):
            if i == 0:
                envelope_smoothed[i] = envelope[i]
            else:
                # Fast attack, medium release
                if envelope[i] > envelope_smoothed[i-1]:
                    coeff = 0.8  # Fast attack
                else:
                    coeff = 0.3  # Medium release
                envelope_smoothed[i] = coeff * envelope[i] + (1 - coeff) * envelope_smoothed[i-1]
        
        # Calculate gain reduction
        gain_reduction = np.ones_like(audio)
        above_threshold = envelope_smoothed > threshold
        
        if np.any(above_threshold):
            # Simplified compression
            overs = envelope_smoothed[above_threshold] - threshold
            reduction = 1 - (overs / envelope_smoothed[above_threshold]) * (1 - 1/ratio)
            reduction = np.clip(reduction, 0.5, 1.0)  # Limit max reduction
            gain_reduction[above_threshold] = reduction
        
        # Apply compression
        compressed = audio * gain_reduction
        
        # Makeup gain
        makeup_db = 2.0
        compressed *= db_to_gain(makeup_db)
        
        return compressed
    
    def get_info(self) -> Dict[str, Any]:
        """Get drum enhancer information."""
        info = super().get_info()
        info["config"].update({
            "transient_shaping": self.config.transient_shaping,
            "kick_bass_boost_db": self.config.kick_bass_boost_db,
            "snare_body_db": self.config.snare_bodied_db,
            "high_freq_clarity_db": self.config.high_freq_clarity_db,
            "punch_compression": self.config.punch_compression,
            "punch_threshold_db": self.config.punch_threshold_db
        })
        return info


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing DrumEnhancer...")
    
    # Create test audio (simulated drums)
    sample_rate = 44100
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Kick drum (low frequency pulse)
    kick = np.sin(2 * np.pi * 50 * t) * np.exp(-3 * t)
    kick = kick * np.exp(-10 * (t % 0.5))  # Repeat every 0.5s
    
    # Snare (noise + tone)
    snare_tone = np.sin(2 * np.pi * 200 * t)
    snare_noise = np.random.randn(len(t)) * 0.3
    snare = (snare_tone + snare_noise) * np.exp(-5 * (t % 0.25))
    
    # Combine
    drums = kick + snare * 0.5
    
    # Add some noise
    drums += 0.02 * np.random.randn(len(drums))
    
    print(f"Input audio: {drums.shape}, range=[{drums.min():.3f}, {drums.max():.3f}]")
    
    # Create enhancer
    enhancer = DrumEnhancer()
    print(f"Created: {enhancer}")
    
    # Process
    result = enhancer.enhance(drums, sample_rate)
    print(f"Output audio: {result.audio.shape}")
    print(f"Enhancements: {result.enhancements_applied}")
    print(f"Output range: [{result.audio.min():.3f}, {result.audio.max():.3f}]")
    
    # Save test audio
    import soundfile as sf
    sf.write("test_drums_before.wav", drums, sample_rate)
    sf.write("test_drums_after.wav", result.audio, sample_rate)
    print("Saved test audio files")
