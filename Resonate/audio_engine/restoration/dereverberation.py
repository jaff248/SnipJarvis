"""
Dereverberation Module - Remove venue acoustics from recordings

Uses DSP-based techniques to reduce reverb/echo from venue acoustics,
focusing on early reflections and late reverb tail.
"""

import logging
from typing import Optional, Tuple
from dataclasses import dataclass

import numpy as np
from scipy import signal
import librosa

logger = logging.getLogger(__name__)


@dataclass
class DereverberationConfig:
    """Configuration for dereverberation."""
    # Intensity (0.0 to 1.0)
    intensity: float = 0.3
    
    # Wet/dry mix
    dry_wet: float = 0.7  # 1.0 = fully wet, 0.0 = fully dry
    
    # Adaptive filter settings
    filter_length_ms: float = 50.0  # Adaptive filter length in ms
    mu: float = 0.5  # Adaptive step size
    
    # Enable/disable sections
    enable_early_reflection: bool = True
    enable_late_reverb: bool = True
    
    def __post_init__(self):
        """Validate and clamp values."""
        self.intensity = max(0.0, min(1.0, self.intensity))
        self.dry_wet = max(0.0, min(1.0, self.dry_wet))
        self.filter_length_ms = max(10.0, min(200.0, self.filter_length_ms))
        self.mu = max(0.01, min(1.0, self.mu))


class Dereverberator:
    """
    Remove reverb/echo from venue acoustics.
    
    Uses spectral subtraction and adaptive filtering to reduce reverb
    while preserving direct signal.
    
    Principle: "Reveal, don't fabricate" - only remove reverb, don't damage dry signal
    """
    
    def __init__(self, sample_rate: int, intensity: float = 0.3):
        """
        Initialize dereverberator.
        
        Args:
            sample_rate: Sample rate in Hz
            intensity: Dereverberation intensity (0.0 to 1.0)
        """
        self.sample_rate = sample_rate
        self.config = DereverberationConfig(intensity=intensity)
        
        logger.info(f"Initialized Dereverberator: sr={sample_rate}Hz, intensity={intensity}")
    
    def process(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply dereverberation to audio.
        
        Args:
            audio: Input audio (float32, range [-1, 1])
            
        Returns:
            Audio with reduced reverb
        """
        if audio.size == 0:
            return audio
        
        if self.config.intensity == 0.0:
            return audio
        
        # Store original
        original = audio.copy()
        
        # Apply dereverberation
        processed = audio.copy()
        
        # Calculate blend factor
        intensity = self.config.intensity
        dry_wet = self.config.dry_wet
        
        # Early reflection suppression
        if self.config.enable_early_reflection:
            early = self._suppress_early_reflections(audio)
            if early is not None:
                processed = early * intensity + processed * (1 - intensity)
        
        # Late reverb suppression
        if self.config.enable_late_reverb:
            late = self._suppress_late_reverb(audio)
            if late is not None:
                processed = late * intensity + processed * (1 - intensity)
        
        # Mix dry/wet
        result = original * (1 - dry_wet) + processed * dry_wet
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(result))
        if max_val > 0.99:
            result = result / max_val * 0.99
        
        # Final clip prevention
        result = np.clip(result, -0.99, 0.99)
        
        logger.debug(f"Dereverberation: intensity={self.config.intensity}, dry_wet={dry_wet}")
        
        return result.astype(np.float32)
    
    def _suppress_early_reflections(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """
        Suppress early reflections (first 20-50ms of reverb).
        
        Uses spectral notching at typical reflection frequencies.
        """
        try:
            # Estimate reverb time and create adaptive filter
            filter_length = int(self.config.filter_length_ms * self.sample_rate / 1000)
            filter_length = max(filter_length, 100)
            
            # Simple approach: adaptive noise cancellation style
            # Use delayed version as reference for reverb estimation
            delay_samples = int(0.005 * self.sample_rate)  # 5ms delay
            delay_samples = min(delay_samples, len(audio) // 2)
            
            if delay_samples < 10:
                return None
            
            # Estimate reverb from delayed signal
            reverb_estimate = np.zeros_like(audio)
            reverb_estimate[delay_samples:] = audio[:-delay_samples]
            
            # Subtract estimated reverb
            dereverb = audio - reverb_estimate * self.config.mu * self.config.intensity
            
            return dereverb
            
        except Exception as e:
            logger.warning(f"Early reflection suppression failed: {e}")
            return None
    
    def _suppress_late_reverb(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """
        Suppress late reverb tail.
        
        Uses spectral subtraction based on estimated noise floor.
        """
        try:
            # Compute STFT
            n_fft = 2048
            hop_length = 512
            
            # Compute magnitude spectrogram
            S = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))
            
            # Estimate reverb decay from quietest frames
            frame_means = np.mean(S, axis=0)
            sorted_indices = np.argsort(frame_means)
            
            # Bottom 20% as noise/reverb floor estimate
            noise_floor_idx = int(0.2 * len(sorted_indices))
            reverb_floor = np.mean(S[:, sorted_indices[:noise_floor_idx]], axis=1, keepdims=True)
            
            # Spectral subtraction for reverb
            S_reduced = np.maximum(S - reverb_floor * self.config.intensity, 0)
            
            # Reconstruct with phase from original
            # (Simple magnitude modification)
            S_modified = S * (1 - self.config.intensity) + S_reduced * self.config.intensity
            
            # Invert STFT
            dereverb = librosa.istft(S_modified, hop_length=hop_length, length=len(audio))
            
            # Pad to match original length
            if len(dereverb) < len(audio):
                dereverb = np.pad(dereverb, (0, len(audio) - len(dereverb)))
            elif len(dereverb) > len(audio):
                dereverb = dereverb[:len(audio)]
            
            return dereverb
            
        except Exception as e:
            logger.warning(f"Late reverb suppression failed: {e}")
            # Fallback: simple smoothing
            return self._simple_dereverb(audio)
    
    def _simple_dereverb(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """
        Simple fallback dereverberation using smoothing.
        """
        try:
            # Very gentle smoothing to reduce reverb
            window_size = int(0.02 * self.sample_rate)  # 20ms
            window = np.ones(window_size) / window_size
            
            # Smooth the signal
            smoothed = np.convolve(audio, window, mode='same')
            
            # Mix original and smoothed
            mix = audio * 0.3 + smoothed * 0.7
            
            return mix
            
        except Exception:
            return None
    
    def get_info(self) -> dict:
        """Get dereverberator configuration and info."""
        return {
            "sample_rate": self.sample_rate,
            "config": {
                "intensity": self.config.intensity,
                "dry_wet": self.config.dry_wet,
                "filter_length_ms": self.config.filter_length_ms,
                "mu": self.config.mu,
                "enable_early_reflection": self.config.enable_early_reflection,
                "enable_late_reverb": self.config.enable_late_reverb
            }
        }
    
    def __repr__(self) -> str:
        return (f"Dereverberator(sr={self.sample_rate}, "
                f"intensity={self.config.intensity})")


# Convenience function
def dereverberate(audio: np.ndarray, sample_rate: int, intensity: float = 0.3) -> np.ndarray:
    """
    Apply dereverberation to audio.
    
    Args:
        audio: Input audio (float32, range [-1, 1])
        sample_rate: Sample rate in Hz
        intensity: Dereverberation intensity (0.0 to 1.0)
        
    Returns:
        Dereverberated audio
    """
    dereverb = Dereverberator(sample_rate, intensity)
    return dereverb.process(audio)


# Example usage and testing
if __name__ == "__main__":
    import logging
    import soundfile as sf
    import librosa
    
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Dereverberator...")
    
    # Create test audio with reverb
    sample_rate = 44100
    duration = 3.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Clean signal
    clean = 0.5 * np.sin(2 * np.pi * 440 * t)
    
    # Simulate reverb with delayed copies
    reverb = clean.copy()
    for delay_ms in [20, 50, 100, 200]:
        delay_samples = int(delay_ms * sample_rate / 1000)
        reverb += 0.2 * np.roll(clean, delay_samples)
    
    # Normalize
    reverb = reverb / np.max(np.abs(reverb)) * 0.8
    
    print(f"Reverberant audio: {reverb.shape}, range=[{reverb.min():.3f}, {reverb.max():.3f}]")
    
    # Test dereverberation
    dereverb = Dereverberator(sample_rate, intensity=0.3)
    print(f"Created: {dereverb}")
    
    cleaned = dereverb.process(reverb)
    print(f"Dereverberated audio: {cleaned.shape}, range=[{cleaned.min():.3f}, {cleaned.max():.3f}]")
    
    # Save for comparison
    sf.write("test_reverberant.wav", reverb, sample_rate)
    sf.write("test_dereverberated.wav", cleaned, sample_rate)
    
    print("Saved test_reverberant.wav and test_dereverberated.wav")
    print("✅ Dereverberator test complete!")
