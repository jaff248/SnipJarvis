from dataclasses import dataclass
import numpy as np
from scipy.signal import hilbert

@dataclass
class DynamicsConfig:
    expansion_threshold_db: float = -40.0  # Expand signal below this level
    expansion_ratio: float = 1.5  # Upward expansion ratio (>1.0)
    attack_ms: float = 10.0  # Envelope follower attack time
    release_ms: float = 100.0  # Envelope follower release time
    makeup_gain_db: float = 0.0  # Optional gain after expansion

@dataclass
class DynamicsProfile:
    peak_db: float  # Peak level in dB
    rms_db: float  # RMS level in dB
    dynamic_range_db: float  # Peak - RMS (measure of DR)
    crest_factor_db: float  # Peak / RMS ratio in dB

class DynamicsRestorer:
    def __init__(self, sample_rate: int, config: DynamicsConfig = None):
        """
        Initialize with sample rate and optional config.

        Args:
            sample_rate: The audio sample rate in Hz.
            config: Optional DynamicsConfig. If None, default config is used.
        """
        self.sample_rate = sample_rate
        self.config = config if config is not None else DynamicsConfig()
        
        # Pre-calculate attack and release coefficients from time constants
        # Attack alpha
        if self.config.attack_ms > 0:
            self.attack_alpha = 1.0 - np.exp(-2 * np.pi / (self.sample_rate * self.config.attack_ms / 1000.0))
        else:
            self.attack_alpha = 1.0
            
        # Release alpha
        if self.config.release_ms > 0:
            self.release_alpha = 1.0 - np.exp(-2 * np.pi / (self.sample_rate * self.config.release_ms / 1000.0))
        else:
            self.release_alpha = 1.0
    
    def analyze_dynamics(self, audio: np.ndarray) -> DynamicsProfile:
        """
        Analyze the dynamic characteristics of input audio.

        Args:
            audio: Input audio array.

        Returns:
            DynamicsProfile containing peak, rms, dynamic range, and crest factor in dB.
        """
        # 1. peak_db
        peak_amp = np.max(np.abs(audio)) if audio.size > 0 else 0.0
        peak_db = 20 * np.log10(peak_amp + 1e-10)
        
        # 2. rms
        if audio.size > 0:
            rms = np.sqrt(np.mean(audio**2)) + 1e-10
        else:
            rms = 1e-10
        rms_db = 20 * np.log10(rms)
        
        # 3. dynamic_range_db
        dynamic_range_db = peak_db - rms_db
        
        # 4. crest_factor_db
        crest_factor_db = peak_db - rms_db
        
        return DynamicsProfile(
            peak_db=float(peak_db),
            rms_db=float(rms_db),
            dynamic_range_db=float(dynamic_range_db),
            crest_factor_db=float(crest_factor_db)
        )
    
    def apply_upward_expansion(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply upward expansion to restore dynamics.

        Args:
            audio: Input audio array.

        Returns:
            Processed audio array with expanded dynamics.
        """
        if audio.size == 0:
            return audio.copy()
            
        # 1. Calculate envelope using Hilbert transform
        envelope = np.abs(hilbert(audio))
        
        # 2. Convert to dB
        envelope_db = 20 * np.log10(envelope + 1e-10)
        
        # 3. Create gain reduction array initialized to zeros
        gain_db = np.zeros_like(envelope_db)
        
        # 4. Find samples below threshold
        below_threshold = envelope_db < self.config.expansion_threshold_db
        
        # 5. For those samples, calculate expansion
        if np.any(below_threshold):
            distance_below = self.config.expansion_threshold_db - envelope_db[below_threshold]
            # gain_db[below_threshold] = -distance_below * (expansion_ratio - 1)
            # Note: negative gain makes quiet parts quieter, increasing DR
            gain_db[below_threshold] = -distance_below * (self.config.expansion_ratio - 1.0)
            
        # 6. Apply envelope attack/release smoothing if needed (optional)
        # Smoothing skipped to strictly follow dependency constraints (no lfilter import in spec)
        # and performance considerations for pure Python loops.
        # The Hilbert envelope already provides a baseband signal.
        
        # 7. Convert gain_db to linear
        gain = 10**(gain_db / 20.0)
        
        # 8. Apply makeup gain
        if self.config.makeup_gain_db != 0.0:
            gain *= 10**(self.config.makeup_gain_db / 20.0)
            
        # 9. Return audio * gain
        return audio * gain
    
    def process(self, audio: np.ndarray) -> np.ndarray:
        """
        Main entry point - the public interface.
        
        Args:
            audio: Input audio array.
            
        Returns:
            Processed audio array.
        """
        return self.apply_upward_expansion(audio)

def restore_dynamics(audio: np.ndarray, sample_rate: int, expansion_ratio: float = 1.5) -> np.ndarray:
    """
    Create a DynamicsRestorer with default config but custom expansion_ratio.
    Process the audio and return the result.
    
    Args:
        audio: Input audio array.
        sample_rate: Audio sample rate.
        expansion_ratio: Ratio for upward expansion.
        
    Returns:
        Processed audio array.
    """
    config = DynamicsConfig(expansion_ratio=expansion_ratio)
    restorer = DynamicsRestorer(sample_rate, config)
    return restorer.process(audio)
