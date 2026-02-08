"""
Vocal Enhancer - Speech-specific audio enhancement for vocals

Provides noise reduction, presence EQ, de-essing, and compression
optimized for vocal clarity in live music recordings.
"""

import logging
import dataclasses
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from enum import Enum

import numpy as np
from scipy import signal

from .base import BaseStemEnhancer, EnhancementConfig, EnhancementResult
from ..utils import db_to_gain

logger = logging.getLogger(__name__)


@dataclass
class VocalConfig(EnhancementConfig):
    """Vocal-specific enhancement configuration."""
    # EQ settings (dB)
    low_cut_freq: float = 80.0       # High-pass filter frequency
    presence_boost_db: float = 3.0   # Boost at 2-4 kHz for clarity
    presence_freq: float = 3000.0    # Frequency for presence boost
    sibilance_reduction_db: float = -4.0  # De-essing at 5-8 kHz
    
    # Compression
    compressor_threshold_db: float = -24.0
    compressor_ratio: float = 4.0
    compressor_attack_ms: float = 10.0
    compressor_release_ms: float = 100.0
    
    # Noise reduction
    noise_reduction_strength: float = 0.5
    
    # De-reverb
    dereverb_amount: float = 0.0     # 0.0 = disabled, 0.3 = moderate
    
    def __post_init__(self):
        """Validate vocal-specific settings."""
        super().__post_init__()
        if not 0 <= self.noise_reduction_strength <= 1:
            raise ValueError("Noise reduction strength must be 0-1")
        if not 0 <= self.dereverb_amount <= 1:
            raise ValueError("Dereverb amount must be 0-1")


class VocalEnhancer(BaseStemEnhancer):
    """
    Vocal stem enhancer with specialized processing.
    
    Optimization goals:
    - Remove background noise and crowd sounds
    - Increase vocal presence and intelligibility
    - Reduce sibilance and harshness
    - Control dynamics without squashing
    - Remove room reverb for dry, close-mic sound
    """
    
    STEM_NAME = "vocals"
    DEFAULT_CONFIG = VocalConfig(
        intensity=0.5,
        low_cut_freq=80.0,
        presence_boost_db=3.0,
        presence_freq=3000.0,
        sibilance_reduction_db=-4.0,
        compressor_threshold_db=-24.0,
        compressor_ratio=4.0,
        noise_reduction_strength=0.5,
        dereverb_amount=0.0
    )
    
    # Fix for dataclass copy issue - create new config with modified intensity
    DEFAULT_CONFIG_INTENSITY_0_5 = dataclasses.replace(DEFAULT_CONFIG, intensity=0.5)
    
    def __init__(self, config: VocalConfig = None):
        """Initialize vocal enhancer."""
        super().__init__(config)
        if config is None:
            self.config = VocalConfig(
                intensity=0.5,
                low_cut_freq=80.0,
                presence_boost_db=3.0,
                presence_freq=3000.0,
                sibilance_reduction_db=-4.0,
                compressor_threshold_db=-24.0,
                compressor_ratio=4.0,
                noise_reduction_strength=0.5,
                dereverb_amount=0.0
            )
    
    def enhance(self, audio: np.ndarray, sample_rate: int) -> EnhancementResult:
        """
        Apply vocal-specific enhancement.
        
        Pipeline:
        1. High-pass filter (remove rumble)
        2. Noise reduction (spectral gating)
        3. Presence EQ (2-4 kHz boost)
        4. De-essing (5-8 kHz reduction)
        5. Compression
        6. Optional de-reverb
        
        Args:
            audio: Input vocal audio
            sample_rate: Sample rate
            
        Returns:
            EnhancementResult with processed vocals
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
        
        # Apply intensity scaling to effects
        intensity = self.config.intensity
        
        # 1. High-pass filter
        if self.config.low_cut_freq > 0:
            processed = self._highpass_filter(processed, sample_rate, 
                                             self.config.low_cut_freq)
            enhancements.append("highpass")
        
        # 2. Noise reduction
        if self.config.noise_reduction and self.config.noise_reduction_strength > 0:
            processed = self._noise_reduction(processed, sample_rate,
                                             self.config.noise_reduction_strength * intensity)
            enhancements.append("noise_reduction")
        
        # 3. Presence EQ
        if self.config.eq:
            boost_db = self.config.presence_boost_db * intensity
            processed = self._presence_boost(processed, sample_rate,
                                           self.config.presence_freq, boost_db)
            enhancements.append("presence_eq")
        
        # 4. De-essing
        if self.config.compression:
            reduction_db = self.config.sibilance_reduction_db * intensity
            processed = self._de_essing(processed, sample_rate, reduction_db)
            enhancements.append("de_essing")
        
        # 5. Compression
        if self.config.compression:
            processed = self._compression(processed, sample_rate)
            enhancements.append("compression")
        
        # 6. De-reverb (optional)
        if hasattr(self.config, 'dereverb_amount') and self.config.dereverb_amount > 0:
            processed = self._simple_dereverb(processed, sample_rate,
                                            self.config.dereverb_amount * intensity)
            enhancements.append("dereverb")
        
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
        """Apply high-pass filter to remove low frequencies."""
        nyquist = sample_rate / 2
        normalized_cutoff = cutoff_freq / nyquist
        
        # Second-order Butterworth
        b, a = signal.butter(2, normalized_cutoff, btype='high')
        return signal.filtfilt(b, a, audio)
    
    def _noise_reduction(self, audio: np.ndarray, sample_rate: int,
                        strength: float) -> np.ndarray:
        """
        Apply spectral gating noise reduction.
        
        Uses noise profile estimation from quiet sections.
        """
        try:
            import noisereduce as nr
            
            # Estimate noise from beginning (assumed quiet)
            noise_duration = min(0.5, len(audio) / sample_rate / 4)
            noise_samples = int(noise_duration * sample_rate)
            noise_profile = audio[:noise_samples]
            
            # Apply noise reduction
            reduced = nr.reduce_noise(
                y=audio,
                sr=sample_rate,
                y_noise=noise_profile,
                prop_decrease=strength,
                stationary=True,
                n_fft=2048,
                win_length=2048,
                hop_length=512
            )
            
            return reduced
            
        except ImportError:
            # Fallback to simple gate if noisereduce not available
            logger.warning("noisereduce not available, using simple gate")
            return self._simple_noise_gate(audio, strength)
    
    def _simple_noise_gate(self, audio: np.ndarray, threshold: float) -> np.ndarray:
        """Simple noise gate as fallback."""
        threshold_abs = threshold * 0.1  # Scale to reasonable range
        return np.where(np.abs(audio) > threshold_abs, audio, 0)
    
    def _presence_boost(self, audio: np.ndarray, sample_rate: int,
                       freq: float, boost_db: float) -> np.ndarray:
        """Apply presence boost at specified frequency."""
        nyquist = sample_rate / 2
        normalized_freq = freq / nyquist
        
        # Peaking filter for presence
        Q = 2.0  # Quality factor
        gain = boost_db
        
        b, a = signal.iirpeak(normalized_freq, Q, gain)
        return signal.filtfilt(b, a, audio)
    
    def _de_essing(self, audio: np.ndarray, sample_rate: int,
                  reduction_db: float) -> np.ndarray:
        """
        Apply de-essing to reduce sibilance.
        
        Uses multi-band compression on high frequencies.
        """
        if reduction_db >= 0:
            return audio
        
        # Split into bands
        # Use crossover for high frequencies (5-8 kHz region)
        low_freq = 4000  # Hz
        high_freq = 8000  # Hz
        
        nyquist = sample_rate / 2
        
        # Low-pass for sibilance band
        low_cutoff = min(low_freq / nyquist, 0.99)
        b_low, a_low = signal.butter(4, low_cutoff, btype='low')
        low_band = signal.filtfilt(b_low, a_low, audio)
        
        # High-pass for sibilance band
        high_cutoff = min(high_freq / nyquist, 0.99)
        b_high, a_high = signal.butter(4, high_cutoff, btype='high')
        high_band = signal.filtfilt(b_high, a_high, audio)
        
        # Apply reduction to high band
        reduction = db_to_gain(reduction_db)
        high_band_reduced = high_band * (1.0 + reduction)
        
        # Recombine
        # Note: This is approximate - proper implementation would use crossover
        return audio + (high_band_reduced - high_band) * 0.5
    
    def _compression(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply compression to control dynamics."""
        threshold = db_to_gain(self.config.compressor_threshold_db)
        ratio = self.config.compressor_ratio
        
        # Calculate envelope
        envelope = np.abs(signal.hilbert(audio))
        
        # Simple feed-forward compression
        compressed = np.copy(audio)
        above_threshold = envelope > threshold
        
        if np.any(above_threshold):
            # Calculate gain reduction
            overs = envelope[above_threshold] - threshold
            # Apply ratio
            reduction = 1 - (overs / envelope[above_threshold]) * (1 - 1/ratio)
            reduction = np.clip(reduction, 0, 1)
            
            # Apply to signal
            compressed[above_threshold] *= reduction
        
        # Makeup gain (approximate)
        makeup_db = 3.0  # Standard makeup gain
        compressed *= db_to_gain(makeup_db)
        
        return compressed
    
    def _simple_dereverb(self, audio: np.ndarray, sample_rate: int,
                        amount: float) -> np.ndarray:
        """
        Simple de-reverberation using spectral subtraction.
        
        Note: Full de-reverberation requires more sophisticated algorithms.
        This is a basic spectral subtraction approach.
        """
        # Simple approach: subtract delayed version (comb filter subtraction)
        # This is very basic - real dereverberation needs WPE or DNN
        
        delay_ms = 20  # 20ms delay
        delay_samples = int(delay_ms * sample_rate / 1000)
        
        # Create reverb estimate
        reverb_estimate = np.zeros_like(audio)
        if len(audio) > delay_samples:
            reverb_estimate[delay_samples:] = audio[:-delay_samples] * 0.3 * amount
        
        # Subtract reverb estimate
        dereverbbed = audio - reverb_estimate
        
        # Normalize back
        dereverbbed = dereverbbed / (1 + amount * 0.3) if amount > 0 else dereverbbed
        
        return dereverbbed
    
    def get_info(self) -> Dict[str, Any]:
        """Get vocal enhancer information."""
        info = super().get_info()
        info["config"].update({
            "low_cut_freq": self.config.low_cut_freq,
            "presence_boost_db": self.config.presence_boost_db,
            "presence_freq": self.config.presence_freq,
            "sibilance_reduction_db": self.config.sibilance_reduction_db,
            "compressor_threshold_db": self.config.compressor_threshold_db,
            "compressor_ratio": self.config.compressor_ratio,
            "noise_reduction_strength": self.config.noise_reduction_strength,
            "dereverb_amount": self.config.dereverb_amount
        })
        return info


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing VocalEnhancer...")
    
    # Create test audio (simulated vocals)
    sample_rate = 44100
    duration = 3.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Simulated vocal with harmonics
    vocal = (0.5 * np.sin(2 * np.pi * 220 * t) + 
            0.3 * np.sin(2 * np.pi * 440 * t) +
            0.2 * np.sin(2 * np.pi * 660 * t) +
            0.05 * np.random.randn(len(t)))
    
    # Add some "sibilance"
    sibilance = 0.1 * np.sin(2 * np.pi * 6000 * t)
    vocal += sibilance
    
    print(f"Input audio: {vocal.shape}, range=[{vocal.min():.3f}, {vocal.max():.3f}]")
    
    # Create enhancer
    enhancer = VocalEnhancer()
    print(f"Created: {enhancer}")
    
    # Process
    result = enhancer.enhance(vocal, sample_rate)
    print(f"Output audio: {result.audio.shape}")
    print(f"Enhancements: {result.enhancements_applied}")
    print(f"Output range: [{result.audio.min():.3f}, {result.audio.max():.3f}]")
    
    # Save test audio
    import soundfile as sf
    sf.write("test_vocals_before.wav", vocal, sample_rate)
    sf.write("test_vocals_after.wav", result.audio, sample_rate)
    print("Saved test audio files")
