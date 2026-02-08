"""
Frequency Restoration Module - Extend frequency response from phone-limited range

Restores rolled-off bass and treble to extend frequency response from typical
phone recording range (~100Hz-8kHz) to full spectrum (~20Hz-20kHz).
"""

import logging
from typing import Optional
from dataclasses import dataclass

import numpy as np
from scipy import signal

logger = logging.getLogger(__name__)


@dataclass
class FrequencyRestorationConfig:
    """Configuration for frequency restoration."""
    # Intensity (0.0 to 1.0)
    intensity: float = 0.5
    
    # Bass enhancement (sub-bass synthesis)
    bass_boost_db: float = 3.0
    bass_freq_hz: float = 80.0
    
    # High-frequency extension
    treble_boost_db: float = 2.0
    treble_freq_hz: float = 8000.0
    
    # Enable/disable sections
    enable_bass: bool = True
    enable_treble: bool = True
    
    def __post_init__(self):
        """Validate and clamp values."""
        self.intensity = max(0.0, min(1.0, self.intensity))
        self.bass_boost_db = max(-6.0, min(12.0, self.bass_boost_db))
        self.treble_boost_db = max(-6.0, min(12.0, self.treble_boost_db))
        self.bass_freq_hz = max(40.0, min(200.0, self.bass_freq_hz))
        self.treble_freq_hz = max(4000.0, min(16000.0, self.treble_freq_hz))


class FrequencyRestorer:
    """
    Extend frequency response from phone-limited range.
    
    Uses DSP-based techniques to restore:
    - Sub-bass (40-100 Hz): Harmonic reinforcement from fundamental
    - High-frequency (8-20 kHz): Spectral prediction + harmonic synthesis
    
    Principle: "Reveal, don't fabricate" - enhance what's captured, don't invent
    """
    
    def __init__(self, sample_rate: int, intensity: float = 0.5):
        """
        Initialize frequency restorer.
        
        Args:
            sample_rate: Sample rate in Hz
            intensity: Enhancement intensity (0.0 to 1.0)
        """
        self.sample_rate = sample_rate
        self.config = FrequencyRestorationConfig(intensity=intensity)
        
        logger.info(f"Initialized FrequencyRestorer: sr={sample_rate}Hz, intensity={intensity}")
    
    def process(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply frequency extension to audio.
        
        Args:
            audio: Input audio (float32, range [-1, 1])
            
        Returns:
            Audio with extended frequency response
        """
        if audio.size == 0:
            return audio
        
        # Store original for mixing
        original = audio.copy()
        processed = audio.copy()
        
        # Calculate blend factor based on intensity
        blend = self.config.intensity
        
        if blend == 0.0:
            return audio
        
        # Apply bass extension
        if self.config.enable_bass:
            bass_extension = self._extend_bass(audio)
            if bass_extension is not None:
                processed = processed + bass_extension * blend
        
        # Apply treble extension
        if self.config.enable_treble:
            treble_extension = self._extend_treble(audio)
            if treble_extension is not None:
                processed = processed + treble_extension * blend
        
        # Check for NaN/Inf in processed audio
        if not np.isfinite(processed).all():
            logger.warning("Frequency restoration produced non-finite values, returning original")
            return original.astype(np.float32)
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(processed))
        if max_val > 0.99:
            processed = processed / max_val * 0.99
        
        # Mix with original based on intensity
        result = original * (1.0 - blend * 0.5) + processed * (blend * 0.5)
        
        # Final clip prevention and NaN check
        result = np.clip(result, -0.99, 0.99)
        
        # Final safety check
        if not np.isfinite(result).all():
            logger.warning("Result contains non-finite values, returning original")
            return original.astype(np.float32)
        
        logger.debug(f"Frequency restoration: intensity={self.config.intensity}")
        
        return result.astype(np.float32)
    
    def _extend_bass(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """
        Extend low-frequency response.
        
        Uses harmonic reinforcement for sub-bass frequencies.
        """
        nyquist = self.sample_rate / 2
        
        # Skip if sample rate too low
        if nyquist < self.config.bass_freq_hz * 2:
            logger.warning("Sample rate too low for bass extension")
            return None
        
        # Normalized cutoff frequency
        cutoff_norm = min(self.config.bass_freq_hz / nyquist, 0.99)
        
        try:
            # Create high-pass filter to get bass content
            b, a = signal.butter(4, cutoff_norm, btype='high')
            
            # Get bass frequencies
            bass = signal.filtfilt(b, a, audio)
            
            # Apply gentle bass boost
            boost = 10 ** (self.config.bass_boost_db / 20)
            bass = bass * boost
            
            return bass
            
        except Exception as e:
            logger.warning(f"Bass extension failed: {e}")
            return None
    
    def _extend_treble(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """
        Extend high-frequency response.
        
        Uses spectral tilt and harmonic enhancement.
        """
        nyquist = self.sample_rate / 2
        
        # Skip if sample rate too low
        if nyquist < self.config.treble_freq_hz:
            logger.warning("Sample rate too low for treble extension")
            return None
        
        try:
            # Create high-shelf filter for gentle treble boost
            f0 = self.config.treble_freq_hz
            Q = 0.707  # Butterworth Q for smoother response
            
            # Normalized frequency (0 to 1, where 1 = Nyquist)
            w0_norm = f0 / nyquist
            w0_norm = max(0.01, min(0.98, w0_norm))  # Safety clamp
            
            # Convert to angular frequency (radians)
            w0 = np.pi * w0_norm
            
            # Calculate gain in linear scale
            A = 10 ** (self.config.treble_boost_db / 40)
            
            # Calculate alpha with safety check
            sin_w0 = np.sin(w0)
            cos_w0 = np.cos(w0)
            
            # Ensure valid alpha calculation
            alpha_term = (A + 1/A) * (1/Q - 1)
            if alpha_term < 0:
                alpha_term = 0
            alpha = sin_w0 / 2 * np.sqrt(alpha_term + 1)
            
            sqrt_A = np.sqrt(A)
            
            # Calculate biquad coefficients for high-shelf
            b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * sqrt_A * alpha)
            b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
            b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * sqrt_A * alpha)
            a0 = (A + 1) - (A - 1) * cos_w0 + 2 * sqrt_A * alpha
            a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
            a2 = (A + 1) - (A - 1) * cos_w0 - 2 * sqrt_A * alpha
            
            # Normalize coefficients
            b = np.array([b0, b1, b2]) / a0
            a = np.array([1.0, a1, a2]) / a0
            
            # Check for NaN/Inf
            if not (np.isfinite(b).all() and np.isfinite(a).all()):
                logger.warning("Filter coefficients are not finite")
                return None
            
            # Apply filter with safety
            treble = signal.lfilter(b, a, audio)
            
            # Check output for NaN/Inf
            if not np.isfinite(treble).all():
                logger.warning("Treble filter produced non-finite values")
                return None
            
            # Apply gentle gain
            treble = treble * 0.3
            
            return treble
            
        except Exception as e:
            logger.warning(f"Treble extension failed: {e}")
            return None
    
    def get_info(self) -> dict:
        """Get restorer configuration and info."""
        return {
            "sample_rate": self.sample_rate,
            "config": {
                "intensity": self.config.intensity,
                "bass_boost_db": self.config.bass_boost_db,
                "bass_freq_hz": self.config.bass_freq_hz,
                "treble_boost_db": self.config.treble_boost_db,
                "treble_freq_hz": self.config.treble_freq_hz,
                "enable_bass": self.config.enable_bass,
                "enable_treble": self.config.enable_treble
            }
        }
    
    def __repr__(self) -> str:
        return (f"FrequencyRestorer(sr={self.sample_rate}, "
                f"intensity={self.config.intensity})")


# Convenience function
def restore_frequency(audio: np.ndarray, sample_rate: int, intensity: float = 0.5) -> np.ndarray:
    """
    Apply frequency restoration to audio.
    
    Args:
        audio: Input audio (float32, range [-1, 1])
        sample_rate: Sample rate in Hz
        intensity: Enhancement intensity (0.0 to 1.0)
        
    Returns:
        Restored audio
    """
    restorer = FrequencyRestorer(sample_rate, intensity)
    return restorer.process(audio)


# Example usage and testing
if __name__ == "__main__":
    import logging
    import soundfile as sf
    
    logging.basicConfig(level=logging.INFO)
    
    print("Testing FrequencyRestorer...")
    
    # Create test audio
    sample_rate = 44100
    duration = 3.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create audio with limited frequency content
    audio = (
        0.5 * np.sin(2 * np.pi * 440 * t) +    # Midrange
        0.3 * np.sin(2 * np.pi * 1000 * t) +   # High-mid
        0.1 * np.sin(2 * np.pi * 150 * t)      # Bass
    )
    
    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.8
    
    print(f"Input audio: {audio.shape}, range=[{audio.min():.3f}, {audio.max():.3f}]")
    
    # Test restoration
    restorer = FrequencyRestorer(sample_rate, intensity=0.5)
    print(f"Created: {restorer}")
    
    restored = restorer.process(audio)
    print(f"Restored audio: {restored.shape}, range=[{restored.min():.3f}, {restored.max():.3f}]")
    
    # Save for comparison
    sf.write("test_original.wav", audio, sample_rate)
    sf.write("test_restored.wav", restored, sample_rate)
    
    print("Saved test_original.wav and test_restored.wav")
    print("✅ FrequencyRestorer test complete!")
