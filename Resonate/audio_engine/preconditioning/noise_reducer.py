"""
Noise Reducer - Global noise reduction for phone recordings

Handles:
- Crowd noise / ambient sound
- Background hiss
- Wind noise
- HVAC / rumble
- Handling noise

Uses noisereduce library with adaptive noise profile estimation.
Should run BEFORE Demucs separation for best results.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class NoiseReductionConfig:
    """Configuration for noise reduction."""
    # Strength of noise reduction (0.0 = none, 1.0 = aggressive)
    strength: float = 0.5
    
    # Use stationary noise assumption (True = constant noise like hiss)
    stationary: bool = False
    
    # FFT parameters
    n_fft: int = 2048
    hop_length: int = 512
    
    # Noise profile estimation
    noise_profile_duration: float = 0.5  # seconds to use for noise profile
    use_auto_detection: bool = True  # Auto-detect quiet sections
    
    # Frequency-specific settings
    low_freq_reduction: float = 1.0  # Extra reduction below 200Hz (rumble)
    high_freq_reduction: float = 0.8  # Less reduction above 8kHz (preserve sparkle)
    
    def __post_init__(self):
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("Strength must be between 0.0 and 1.0")


class NoiseReducer:
    """
    Global noise reduction for phone recordings.
    
    Key features:
    - Adaptive noise profile estimation
    - Frequency-dependent reduction (more on low-end rumble)
    - Preserves music transients
    - Designed to run BEFORE source separation
    """
    
    def __init__(self, sample_rate: int = 44100, config: NoiseReductionConfig = None):
        """
        Initialize noise reducer.
        
        Args:
            sample_rate: Audio sample rate
            config: Noise reduction configuration
        """
        self.sample_rate = sample_rate
        self.config = config or NoiseReductionConfig()
        
    def process(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply noise reduction to audio.
        
        Args:
            audio: Input audio (mono or stereo)
            
        Returns:
            Noise-reduced audio
        """
        if self.config.strength <= 0:
            return audio
            
        try:
            import noisereduce as nr
        except ImportError:
            logger.warning("noisereduce not installed, skipping noise reduction")
            return audio
        
        # Handle stereo
        if audio.ndim > 1:
            # Process each channel
            channels = []
            for ch in range(audio.shape[1] if audio.ndim == 2 else 1):
                ch_audio = audio[:, ch] if audio.ndim == 2 else audio
                channels.append(self._process_mono(ch_audio, nr))
            return np.column_stack(channels) if audio.ndim == 2 else channels[0]
        else:
            return self._process_mono(audio, nr)
    
    def _process_mono(self, audio: np.ndarray, nr) -> np.ndarray:
        """Process mono audio."""
        # Estimate noise profile
        noise_profile = self._estimate_noise_profile(audio)
        
        # Apply main noise reduction
        reduced = nr.reduce_noise(
            y=audio,
            sr=self.sample_rate,
            y_noise=noise_profile,
            prop_decrease=self.config.strength,
            stationary=self.config.stationary,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
        )
        
        # Apply frequency-specific reduction
        if self.config.low_freq_reduction > 1.0:
            reduced = self._reduce_low_frequency_noise(reduced)
        
        return reduced.astype(np.float32)
    
    def _estimate_noise_profile(self, audio: np.ndarray) -> np.ndarray:
        """
        Estimate noise profile from quiet sections.
        
        Uses RMS analysis to find quietest portions.
        """
        if not self.config.use_auto_detection:
            # Use first N seconds as noise profile
            samples = int(self.config.noise_profile_duration * self.sample_rate)
            return audio[:min(samples, len(audio))]
        
        # Auto-detect quiet sections
        frame_length = int(0.05 * self.sample_rate)  # 50ms frames
        hop = frame_length // 2
        
        # Calculate RMS for each frame
        n_frames = (len(audio) - frame_length) // hop + 1
        rms_values = []
        
        for i in range(n_frames):
            start = i * hop
            end = start + frame_length
            frame = audio[start:end]
            rms = np.sqrt(np.mean(frame ** 2))
            rms_values.append((rms, start, end))
        
        if not rms_values:
            return audio[:int(self.config.noise_profile_duration * self.sample_rate)]
        
        # Find quietest 10% of frames
        rms_values.sort(key=lambda x: x[0])
        n_quiet = max(1, len(rms_values) // 10)
        quiet_frames = rms_values[:n_quiet]
        
        # Extract and concatenate quiet audio
        noise_segments = []
        for _, start, end in quiet_frames:
            noise_segments.append(audio[start:end])
        
        noise_profile = np.concatenate(noise_segments)
        
        # Ensure minimum length
        min_samples = int(self.config.noise_profile_duration * self.sample_rate)
        if len(noise_profile) < min_samples:
            # Repeat to get minimum length
            repeats = int(np.ceil(min_samples / len(noise_profile)))
            noise_profile = np.tile(noise_profile, repeats)[:min_samples]
        
        return noise_profile
    
    def _reduce_low_frequency_noise(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply extra reduction to low frequencies (rumble).
        
        Uses multi-band approach to reduce sub-200Hz noise more aggressively.
        """
        from scipy import signal
        
        # Design high-pass filter at 80Hz
        nyquist = self.sample_rate / 2
        cutoff = 80.0 / nyquist
        
        # Separate low frequencies
        b_low, a_low = signal.butter(4, cutoff, btype='low')
        b_high, a_high = signal.butter(4, cutoff, btype='high')
        
        low_band = signal.filtfilt(b_low, a_low, audio)
        high_band = signal.filtfilt(b_high, a_high, audio)
        
        # Reduce low band more aggressively
        low_reduction = self.config.low_freq_reduction * self.config.strength
        low_band_reduced = low_band * (1.0 - low_reduction * 0.5)
        
        # Recombine
        return high_band + low_band_reduced


def reduce_noise(audio: np.ndarray, sample_rate: int, 
                strength: float = 0.5,
                stationary: bool = False) -> np.ndarray:
    """
    Convenience function for noise reduction.
    
    Args:
        audio: Input audio
        sample_rate: Sample rate
        strength: Reduction strength (0.0-1.0)
        stationary: Use stationary noise assumption
        
    Returns:
        Noise-reduced audio
    """
    config = NoiseReductionConfig(strength=strength, stationary=stationary)
    reducer = NoiseReducer(sample_rate, config)
    return reducer.process(audio)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing NoiseReducer...")
    
    # Create test audio with noise
    sample_rate = 44100
    duration = 3.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Clean signal (music)
    clean = 0.5 * np.sin(2 * np.pi * 440 * t)
    
    # Add noise
    noise = 0.1 * np.random.randn(len(t))  # White noise
    rumble = 0.05 * np.sin(2 * np.pi * 30 * t)  # Low frequency rumble
    
    noisy = clean + noise + rumble
    
    print(f"Input: {noisy.shape}, SNR estimate: {10*np.log10(np.var(clean)/np.var(noise)):.1f} dB")
    
    # Process
    reducer = NoiseReducer(sample_rate, NoiseReductionConfig(strength=0.7))
    reduced = reducer.process(noisy)
    
    # Estimate improvement
    residual_noise = reduced - clean
    output_snr = 10 * np.log10(np.var(clean) / (np.var(residual_noise) + 1e-10))
    print(f"Output SNR estimate: {output_snr:.1f} dB")
    
    print("Noise reduction test complete")
