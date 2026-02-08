"""
Bass Enhancer - Low-frequency audio enhancement for bass

Provides clarity enhancement, low-end control, and harmonic excitation
optimized for bass guitar and sub-bass in live music recordings.
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
class BassConfig(EnhancementConfig):
    """Bass-specific enhancement configuration."""
    # Low-end clarity
    low_clarity_db: float = 3.0      # Boost at 200-400 Hz for clarity
    low_clarity_freq: float = 300.0
    sub_bass_boost_db: float = 2.0   # Boost below 60 Hz for weight
    
    # Harmonic excitation
    harmonic_excitation: bool = True
    harmonic_amount: float = 0.3     # Intensity of harmonics (0-1)
    harmonic_type: str = "tube"      # "tube", "solidstate", "tape"
    
    # De-mudding (remove 200-300 Hz muddiness)
    demud_amount: float = -2.0       # Cut in mud frequency range
    
    # Compression
    compression: bool = True
    compressor_threshold_db: float = -24.0
    compressor_ratio: float = 4.0
    compressor_attack_ms: float = 20.0
    compressor_release_ms: float = 150.0
    
    # High-pass for input (remove sub-sonic)
    input_highpass_hz: float = 20.0
    
    def __post_init__(self):
        """Validate bass-specific settings."""
        super().__post_init__()
        if not 0 <= self.harmonic_amount <= 1:
            raise ValueError("Harmonic amount must be 0-1")


class BassEnhancer(BaseStemEnhancer):
    """
    Bass stem enhancer with specialized processing.
    
    Optimization goals:
    - Add clarity and definition to bass notes
    - Enhance sub-bass for weight and impact
    - Add harmonic content (excitation) for warmth
    - Remove "muddy" low-mid frequencies
    - Control dynamics for consistent bass presence
    - Preserve natural bass character
    """
    
    STEM_NAME = "bass"
    DEFAULT_CONFIG = BassConfig(
        intensity=0.5,
        low_clarity_db=3.0,
        low_clarity_freq=300.0,
        sub_bass_boost_db=2.0,
        harmonic_excitation=True,
        harmonic_amount=0.3,
        harmonic_type="tube",
        demud_amount=-2.0,
        compression=True,
        compressor_threshold_db=-24.0,
        compressor_ratio=4.0,
        compressor_attack_ms=20.0,
        compressor_release_ms=150.0,
        input_highpass_hz=20.0
    )
    
    def __init__(self, config: BassConfig = None):
        """Initialize bass enhancer."""
        super().__init__(config)
    
    def enhance(self, audio: np.ndarray, sample_rate: int) -> EnhancementResult:
        """
        Apply bass-specific enhancement.
        
        Pipeline:
        1. High-pass filter (remove sub-sonic below 20 Hz)
        2. De-mudding (cut 200-300 Hz muddiness)
        3. Low-frequency clarity boost
        4. Sub-bass enhancement
        5. Harmonic excitation (adds warmth)
        6. Compression for consistency
        7. Final limiting to prevent clipping
        
        Args:
            audio: Input bass audio
            sample_rate: Sample rate
            
        Returns:
            EnhancementResult with processed bass
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
        if self.config.input_highpass_hz > 0:
            processed = self._highpass_filter(processed, sample_rate,
                                             self.config.input_highpass_hz)
            enhancements.append("subsonic_filter")
        
        # 2. De-mudding
        if self.config.demud_amount < 0:
            cut_db = self.config.demud_amount * intensity
            processed = self._demud(processed, sample_rate, cut_db)
            enhancements.append("demudding")
        
        # 3. Low clarity boost
        boost_db = self.config.low_clarity_db * intensity
        processed = self._low_clarity(processed, sample_rate,
                                     self.config.low_clarity_freq, boost_db)
        enhancements.append("low_clarity")
        
        # 4. Sub-bass enhancement
        sub_db = self.config.sub_bass_boost_db * intensity
        processed = self._sub_bass_boost(processed, sample_rate, sub_db)
        enhancements.append("sub_bass")
        
        # 5. Harmonic excitation
        if self.config.harmonic_excitation and self.config.harmonic_amount > 0:
            processed = self._harmonic_excitation(processed, sample_rate,
                                                 self.config.harmonic_amount * intensity,
                                                 self.config.harmonic_type)
            enhancements.append("harmonic_excitation")
        
        # 6. Compression
        if self.config.compression:
            processed = self._compression(processed, sample_rate)
            enhancements.append("compression")
        
        # 7. Soft limiting to prevent clipping
        processed = self._soft_limit(processed)
        enhancements.append("soft_limiting")
        
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
        
        # Third-order Butterworth
        b, a = signal.butter(3, normalized_cutoff, btype='high')
        return signal.filtfilt(b, a, audio)
    
    def _demud(self, audio: np.ndarray, sample_rate: int,
              cut_db: float) -> np.ndarray:
        """
        Cut "muddy" frequencies (200-300 Hz range).
        
        This removes the boominess that masks bass clarity.
        """
        if cut_db >= 0:
            return audio
        
        nyquist = sample_rate / 2
        
        # Center frequency for mud cut
        mud_freq = 250  # Hz
        Q = 1.5
        
        normalized_freq = mud_freq / nyquist
        normalized_freq = min(max(normalized_freq, 1e-3), 0.99)
        gain = db_to_gain(cut_db)
        
        b, a = signal.iirpeak(normalized_freq, Q)
        filtered = signal.filtfilt(b, a, audio)
        
        # Blend: apply cut to the peak band
        return audio + (filtered * (gain - 1.0)) * 0.5
    
    def _low_clarity(self, audio: np.ndarray, sample_rate: int,
                    freq: float, boost_db: float) -> np.ndarray:
        """
        Boost low-mid frequencies for clarity.
        
        This adds definition and helps bass stand out in the mix.
        """
        if boost_db <= 0:
            return audio
            
        nyquist = sample_rate / 2
        normalized_freq = freq / nyquist
        normalized_freq = min(max(normalized_freq, 1e-3), 0.99)
        
        # Peaking filter for clarity
        Q = 2.0
        gain = db_to_gain(boost_db)
        
        b, a = signal.iirpeak(normalized_freq, Q)
        filtered = signal.filtfilt(b, a, audio)
        
        # Add filtered band with gain
        boosted = audio + (filtered * (gain - 1.0)) * 0.4
        
        return boosted
    
    def _sub_bass_boost(self, audio: np.ndarray, sample_rate: int,
                       boost_db: float) -> np.ndarray:
        """Boost sub-bass frequencies for weight and impact."""
        if boost_db <= 0:
            return audio
            
        nyquist = sample_rate / 2
        
        # Sub-bass shelf
        sub_freq = 50  # Hz
        normalized_freq = sub_freq / nyquist
        
        # Low-shelving filter
        b, a = signal.butter(2, normalized_freq, btype='low')
        low_content = signal.filtfilt(b, a, audio)
        
        # Add low content with boost
        boost_factor = db_to_gain(boost_db)
        boosted = audio + (low_content * (boost_factor - 1) * 0.5)
        
        return boosted
    
    def _harmonic_excitation(self, audio: np.ndarray, sample_rate: int,
                            amount: float, excitation_type: str) -> np.ndarray:
        """
        Add harmonic content for warmth and character.
        
        Types:
        - "tube": Soft, warm harmonics (even harmonics)
        - "solidstate": Clean, punchy (more odd harmonics)
        - "tape": Vintage warmth (soft clipping)
        """
        # Generate harmonics
        harmonics = np.zeros_like(audio)
        
        # Even harmonics (tube-like)
        harmonics += 0.3 * np.sign(audio) * np.abs(audio) ** 1.2  # 2nd harmonic
        harmonics += 0.1 * np.sign(audio) * np.abs(audio) ** 1.4  # 3rd harmonic
        
        if excitation_type == "solidstate":
            # More odd harmonics for punch
            harmonics = 0.5 * np.sign(audio) * np.abs(audio) ** 1.1
        
        elif excitation_type == "tape":
            # Soft clipping characteristic
            harmonics = 0.4 * np.tanh(audio * 2)
        
        # Blend harmonics with original based on amount
        excited = audio + harmonics * amount * 0.3
        
        return excited
    
    def _compression(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply gentle compression for consistent bass."""
        threshold = db_to_gain(self.config.compressor_threshold_db)
        ratio = self.config.compressor_ratio
        
        attack_samples = int(self.config.compressor_attack_ms * sample_rate / 1000)
        release_samples = int(self.config.compressor_release_ms * sample_rate / 1000)
        
        # Calculate envelope
        envelope = np.abs(audio)
        
        # Smoothing
        alpha_attack = 1 - np.exp(-1/attack_samples)
        alpha_release = 1 - np.exp(-1/release_samples)
        
        envelope_smoothed = np.zeros_like(envelope)
        for i in range(len(envelope)):
            if i == 0:
                envelope_smoothed[i] = envelope[i]
            else:
                if envelope[i] > envelope_smoothed[i-1]:
                    envelope_smoothed[i] = (alpha_attack * envelope[i] + 
                                           (1 - alpha_attack) * envelope_smoothed[i-1])
                else:
                    envelope_smoothed[i] = (alpha_release * envelope[i] + 
                                           (1 - alpha_release) * envelope_smoothed[i-1])
        
        # Calculate gain reduction
        gain_reduction = np.ones_like(audio)
        above_threshold = envelope_smoothed > threshold
        
        if np.any(above_threshold):
            overs = envelope_smoothed[above_threshold] - threshold
            reduction = 1 - (overs / envelope_smoothed[above_threshold]) * (1 - 1/ratio)
            reduction = np.clip(reduction, 0.5, 1.0)
            gain_reduction[above_threshold] = reduction
        
        # Apply compression
        compressed = audio * gain_reduction
        
        # Makeup gain
        makeup_db = 2.0
        compressed *= db_to_gain(makeup_db)
        
        return compressed
    
    def _soft_limit(self, audio: np.ndarray, ceiling: float = 0.95) -> np.ndarray:
        """Soft limiting to prevent hard clipping."""
        # Use tanh for soft clipping
        return np.tanh(audio * (1 / ceiling)) * ceiling
    
    def get_info(self) -> Dict[str, Any]:
        """Get bass enhancer information."""
        info = super().get_info()
        info["config"].update({
            "low_clarity_db": self.config.low_clarity_db,
            "sub_bass_boost_db": self.config.sub_bass_boost_db,
            "harmonic_excitation": self.config.harmonic_excitation,
            "harmonic_amount": self.config.harmonic_amount,
            "harmonic_type": self.config.harmonic_type,
            "demud_amount": self.config.demud_amount,
            "compression": self.config.compression,
            "compressor_threshold_db": self.config.compressor_threshold_db
        })
        return info


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing BassEnhancer...")
    
    # Create test audio (simulated bass)
    sample_rate = 44100
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Bass notes (low frequency)
    bass_notes = []
    for i in range(4):
        freq = 55 + (i % 2) * 27.5  # A1, A2, E1, E2 alternation
        start = i * 0.5
        end = start + 0.45
        note = np.sin(2 * np.pi * freq * t[start*sample_rate:end*sample_rate])
        envelope = np.exp(-3 * np.linspace(0, 0.45, int(0.45 * sample_rate)))
        bass_notes.append(note * envelope * 0.8)
    
    bass = np.zeros_like(t)
    for i, note in enumerate(bass_notes):
        start = int(i * 0.5 * sample_rate)
        end = start + len(note)
        bass[start:end] = note
    
    # Add some "muddiness" in 200-300 Hz
    mud = 0.1 * np.sin(2 * np.pi * 250 * t)
    bass += mud
    
    # Add noise
    bass += 0.02 * np.random.randn(len(bass))
    
    print(f"Input audio: {bass.shape}, range=[{bass.min():.3f}, {bass.max():.3f}]")
    
    # Create enhancer
    enhancer = BassEnhancer()
    print(f"Created: {enhancer}")
    
    # Process
    result = enhancer.enhance(bass, sample_rate)
    print(f"Output audio: {result.audio.shape}")
    print(f"Enhancements: {result.enhancements_applied}")
    print(f"Output range: [{result.audio.min():.3f}, {result.audio.max():.3f}]")
    
    # Save test audio
    import soundfile as sf
    sf.write("test_bass_before.wav", bass, sample_rate)
    sf.write("test_bass_after.wav", result.audio, sample_rate)
    print("Saved test audio files")
