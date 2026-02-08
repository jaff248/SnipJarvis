from dataclasses import dataclass, field
import numpy as np
import logging
from typing import List, Tuple, Optional, Any

logger = logging.getLogger(__name__)

@dataclass
class ArticulationProfile:
    """Expressive musical details."""
    vibrato_detected: bool
    vibrato_rate_hz: float
    pitch_bends: List[Tuple[float, float]] = field(default_factory=list) # (time, bend_cents)
    dynamic_range_db: float = 0.0
    attack_time_ms: float = 0.0
    articulation_description: str = "normal"

class ArticulationDetector:
    """Detects musical articulation and expression."""
    
    def __init__(self):
        pass
        
    def detect(self, audio: np.ndarray, sr: int, stem_type: str) -> ArticulationProfile:
        """
        Detect articulation features.
        
        Args:
            audio: Audio samples
            sr: Sample rate
            stem_type: Type of stem
            
        Returns:
            ArticulationProfile object
        """
        try:
            import librosa
            
            # Ensure audio is mono
            if audio.ndim > 1:
                y = np.mean(audio, axis=1)
            else:
                y = audio
                
            # Dynamic Range
            rms = librosa.feature.rms(y=y)[0]
            if len(rms) > 0:
                rms_db = librosa.amplitude_to_db(rms, ref=np.max)
                dynamic_range_db = np.max(rms_db) - np.min(rms_db)
            else:
                dynamic_range_db = 0.0
                
            # Attack Time (Onset strength slope)
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            if len(onset_env) > 0:
                # Average time between onset peak and start (rough approximation)
                attack_time_ms = 10.0 # Placeholder/Default
                if np.max(onset_env) > 0:
                    attack_time_ms = 1000.0 * (1.0 / (np.mean(onset_env[onset_env > np.mean(onset_env)]) + 1e-6))
                    attack_time_ms = min(500.0, max(5.0, attack_time_ms))
            else:
                attack_time_ms = 50.0
                
            # Vibrato Detection (Pitch modulation)
            vibrato_detected = False
            vibrato_rate = 0.0
            
            if stem_type in ['vocals', 'other']:
                try:
                    f0, voiced_flag, voiced_probs = librosa.pyin(
                        y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr
                    )
                    
                    if np.any(voiced_flag):
                        # Simple vibrato detection on F0
                        # Look for 4-8Hz modulation in F0 contour
                        f0_voiced = f0[voiced_flag]
                        if len(f0_voiced) > 20:
                            # Normalize F0
                            f0_norm = f0_voiced - np.mean(f0_voiced)
                            # FFT of F0 contour
                            f0_fft = np.abs(np.fft.rfft(f0_norm))
                            freqs = np.fft.rfftfreq(len(f0_norm), d=librosa.get_duration(y=y, sr=sr)/len(f0_norm))
                            
                            # Check energy in 4-8Hz band
                            vib_mask = (freqs >= 4) & (freqs <= 8)
                            if np.any(vib_mask):
                                vib_energy = np.sum(f0_fft[vib_mask])
                                total_energy = np.sum(f0_fft)
                                if (vib_energy / (total_energy + 1e-10)) > 0.1: # Threshold
                                    vibrato_detected = True
                                    vibrato_rate = freqs[vib_mask][np.argmax(f0_fft[vib_mask])]
                except Exception:
                    pass
            
            # Generate description
            description = self._generate_description(
                vibrato_detected, dynamic_range_db, attack_time_ms, stem_type
            )
            
            return ArticulationProfile(
                vibrato_detected=vibrato_detected,
                vibrato_rate_hz=float(vibrato_rate),
                pitch_bends=[], # Todo: implement pitch bend detection
                dynamic_range_db=float(dynamic_range_db),
                attack_time_ms=float(attack_time_ms),
                articulation_description=description
            )
            
        except Exception as e:
            logger.warning(f"Articulation detection failed: {e}")
            return ArticulationProfile(False, 0.0, [], 0.0, 0.0, "normal")

    def _generate_description(self, vibrato: bool, dyn_range: float, attack_ms: float, stem_type: str) -> str:
        descriptors = []
        
        if vibrato:
            descriptors.append("expressive vibrato")
            
        if dyn_range > 40:
            descriptors.append("wide dynamic range")
        elif dyn_range < 10:
            descriptors.append("compressed dynamics")
            
        if attack_ms < 20:
            descriptors.append("sharp attack")
        elif attack_ms > 100:
            descriptors.append("slow attack")
            
        if not descriptors:
            return "natural articulation"
            
        return ", ".join(descriptors)

    def to_description(self, profile: ArticulationProfile) -> str:
        return profile.articulation_description
