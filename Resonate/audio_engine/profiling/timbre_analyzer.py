from dataclasses import dataclass
import numpy as np
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

@dataclass
class TimbreProfile:
    """Spectral characteristics of an audio stem."""
    brightness: float  # Spectral centroid (0-1)
    warmth: float  # Low-frequency energy (0-1)
    harshness: float  # High-frequency distortion (0-1)
    breathiness: float  # Noise content (0-1, for vocals)
    texture_description: str  # "bright", "warm", "harsh", "breathy", etc.

class TimbreAnalyzer:
    """Analyzes audio timbre to generate descriptive prompts."""
    
    def __init__(self):
        pass
        
    def analyze(self, audio: np.ndarray, sr: int, stem_type: str) -> TimbreProfile:
        """
        Analyze the timbre of the audio stem.
        
        Args:
            audio: Audio samples
            sr: Sample rate
            stem_type: Type of stem ('vocals', 'drums', 'bass', 'other')
            
        Returns:
            TimbreProfile object
        """
        try:
            import librosa
            
            # Ensure audio is mono
            if audio.ndim > 1:
                y = np.mean(audio, axis=1)
            else:
                y = audio
                
            # Normalize
            if np.max(np.abs(y)) > 0:
                y = y / np.max(np.abs(y))
                
            # Spectral Centroid (Brightness)
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            avg_centroid = np.mean(spectral_centroid)
            # Normalize centroid (approximate range 0-5000Hz mapped to 0-1)
            brightness = min(1.0, avg_centroid / 5000.0)
            
            # Low Frequency Energy (Warmth)
            S = np.abs(librosa.stft(y))
            freqs = librosa.fft_frequencies(sr=sr)
            low_freq_mask = freqs < 500
            low_energy = np.sum(S[low_freq_mask, :])
            total_energy = np.sum(S)
            warmth = min(1.0, (low_energy / (total_energy + 1e-10)) * 1.5) # Boost slightly
            
            # Zero Crossing Rate (Harshness/Noise)
            zcr = librosa.feature.zero_crossing_rate(y)
            avg_zcr = np.mean(zcr)
            harshness = min(1.0, avg_zcr * 5.0) # Scale up
            
            # Breathiness (approximate using spectral flatness/noise-likeness)
            flatness = librosa.feature.spectral_flatness(y=y)
            avg_flatness = np.mean(flatness)
            breathiness = min(1.0, avg_flatness * 2.0)
            
            # Generate description based on metrics and stem type
            description = self._generate_description(brightness, warmth, harshness, breathiness, stem_type)
            
            return TimbreProfile(
                brightness=float(brightness),
                warmth=float(warmth),
                harshness=float(harshness),
                breathiness=float(breathiness),
                texture_description=description
            )
            
        except Exception as e:
            logger.warning(f"Timbre analysis failed: {e}")
            return TimbreProfile(0.5, 0.5, 0.0, 0.0, "standard")

    def _generate_description(self, brightness: float, warmth: float, harshness: float, breathiness: float, stem_type: str) -> str:
        """Generate a text description from timbre metrics."""
        descriptors = []
        
        if stem_type == 'vocals':
            if brightness > 0.7:
                descriptors.append("bright")
            elif brightness < 0.3:
                descriptors.append("dark")
                
            if breathiness > 0.3:
                descriptors.append("breathy")
            elif breathiness < 0.1:
                descriptors.append("clear")
                
            if warmth > 0.6:
                descriptors.append("warm")
                
            descriptors.append("vocals")
            
        elif stem_type == 'drums':
            if harshness > 0.3:
                descriptors.append("crisp")
            if warmth > 0.6:
                descriptors.append("punchy")
            else:
                descriptors.append("tight")
            descriptors.append("drums")
            
        elif stem_type == 'bass':
            if warmth > 0.7:
                descriptors.append("deep sub")
            elif warmth > 0.5:
                descriptors.append("warm")
            
            if harshness > 0.2:
                descriptors.append("distorted")
            else:
                descriptors.append("clean")
            descriptors.append("bass")
            
        else: # Other
            if brightness > 0.6:
                descriptors.append("bright")
            if warmth > 0.6:
                descriptors.append("warm")
            descriptors.append("instrumental")
            
        return " ".join(descriptors)

    def to_description(self, profile: TimbreProfile, stem_type: str) -> str:
        """Return the texture description."""
        return profile.texture_description
