"""
Tempo and Key Analyzer - Detect tempo (BPM) and musical key from audio.

Uses librosa for onset detection and chroma analysis, with Krumhansl-Schmuckler
key profiles for key detection.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


# Krumhansl-Schmuckler key profiles (major and minor)
KRUMSHANSL_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KRUMSHANSL_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# Standard note names (pitch classes)
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Common time signatures
TIME_SIGNATURES = [
    (4, 4),  # Common time
    (3, 4),  # Waltz time
    (6, 8),  # Compound duple
    (2, 4),  # Cut time
    (5, 4),  # 5/4
    (7, 8),  # 7/8
]


@dataclass
class TempoKeyInfo:
    """
    Structured output for tempo and key detection.
    
    Attributes:
        tempo: Detected tempo in beats per minute (BPM)
        tempo_confidence: Confidence in tempo detection (0-1)
        key: Detected key (e.g., "C", "F#", "Am")
        key_mode: "major" or "minor"
        key_confidence: Confidence in key detection (0-1)
        time_signature: Tuple of (beats per measure, note value)
    """
    tempo: float
    tempo_confidence: float
    key: str
    key_mode: str
    key_confidence: float
    time_signature: Tuple[int, int]
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'tempo': self.tempo,
            'tempo_confidence': self.tempo_confidence,
            'key': self.key,
            'key_mode': self.key_mode,
            'key_confidence': self.key_confidence,
            'time_signature': self.time_signature,
        }


class TempoKeyAnalyzer:
    """
    Analyze audio to detect tempo and musical key.
    
    Uses multiple techniques:
    - Tempo: Onset detection with autocorrelation for beat tracking
    - Key: Chroma analysis with Krumhansl-Schmuckler profiles
    
    Attributes:
        hop_length: FFT hop length for analysis
        n_fft: FFT window size
        frame_rate: Computed frames per second
    """
    
    def __init__(self, hop_length: int = 512, n_fft: int = 2048):
        """
        Initialize tempo/key analyzer.
        
        Args:
            hop_length: FFT hop length (samples between frames)
            n_fft: FFT window size
        """
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.frame_rate = None
        
    def analyze(self, audio: np.ndarray, sample_rate: int) -> TempoKeyInfo:
        """
        Analyze audio for tempo and key.
        
        Args:
            audio: Audio samples (float32, range [-1, 1])
            sample_rate: Sample rate in Hz
            
        Returns:
            TempoKeyInfo with detected tempo and key
        """
        try:
            import librosa
            
            self.frame_rate = sample_rate / self.hop_length
            
            # Detect tempo using onset envelope
            tempo, tempo_confidence = self._detect_tempo(audio, sample_rate)
            
            # Detect key using chroma analysis
            key, key_mode, key_confidence = self._detect_key(audio, sample_rate)
            
            # Detect time signature
            time_signature = self._detect_time_signature(audio, sample_rate, tempo)
            
            return TempoKeyInfo(
                tempo=tempo,
                tempo_confidence=tempo_confidence,
                key=key,
                key_mode=key_mode,
                key_confidence=key_confidence,
                time_signature=time_signature,
            )
            
        except ImportError:
            logger.warning("librosa not available, using fallback detection")
            return self._fallback_analysis()
        except Exception as e:
            logger.error(f"Tempo/key analysis failed: {e}")
            return self._fallback_analysis()
    
    def _detect_tempo(self, audio: np.ndarray, sample_rate: int) -> Tuple[float, float]:
        """
        Detect tempo using onset detection and autocorrelation.
        
        Returns:
            Tuple of (tempo_bpm, confidence)
        """
        try:
            import librosa
            
            # Compute onset strength envelope
            onset_env = librosa.onset.onset_strength(
                y=audio, sr=sample_rate,
                hop_length=self.hop_length
            )
            
            # Track beats
            tempo = librosa.beat.tempo(
                y=audio, sr=sample_rate,
                hop_length=self.hop_length,
                onset_envelope=onset_env
            )[0]
            
            # Compute confidence based on onset clarity
            onset_peaks = librosa.onset.onset_detect(
                y=audio, sr=sample_rate,
                hop_length=self.hop_length,
                onset_envelope=onset_env
            )
            
            if len(onset_peaks) > 5:
                # More onsets = higher confidence
                confidence = min(len(onset_peaks) / 50.0, 1.0)
            else:
                confidence = 0.3  # Low confidence for sparse onsets
            
            # Clamp tempo to reasonable range
            tempo = min(max(tempo, 60.0), 200.0)
            
            return float(tempo), float(confidence)
            
        except Exception as e:
            logger.warning(f"Tempo detection failed: {e}")
            return 120.0, 0.5
    
    def _detect_key(self, audio: np.ndarray, sample_rate: int) -> Tuple[str, str, float]:
        """
        Detect musical key using chroma analysis with Krumhansl profiles.
        
        Returns:
            Tuple of (key, mode, confidence)
        """
        try:
            import librosa
            
            # Compute chromagram
            chroma = librosa.feature.chroma_cqt(
                y=audio, sr=sample_rate,
                n_chroma=12, octaves=1,
                hop_length=self.hop_length
            )
            
            # Average chroma over time
            avg_chroma = np.mean(chroma, axis=1)
            avg_chroma = avg_chroma / (np.sum(avg_chroma) + 1e-10)
            
            best_key = "C"
            best_mode = "major"
            best_score = -1
            
            # Check all 12 possible roots
            for root in range(12):
                # Rotate chroma to this root perspective
                rotated_chroma = np.roll(avg_chroma, -root)
                
                # Compare to major profile
                major_score = np.dot(rotated_chroma, KRUMSHANSL_MAJOR)
                if major_score > best_score:
                    best_score = major_score
                    best_key = NOTE_NAMES[root]
                    best_mode = "major"
                
                # Compare to minor profile
                minor_score = np.dot(rotated_chroma, KRUMSHANSL_MINOR)
                if minor_score > best_score:
                    best_score = minor_score
                    best_key = NOTE_NAMES[root]
                    best_mode = "minor"
            
            # Normalize confidence to 0-1
            max_possible = max(np.sum(KRUMSHANSL_MAJOR), np.sum(KRUMSHANSL_MINOR))
            confidence = min(best_score / max_possible, 1.0)
            
            return best_key, best_mode, float(confidence)
            
        except Exception as e:
            logger.warning(f"Key detection failed: {e}")
            return "C", "major", 0.5
    
    def _detect_time_signature(self, audio: np.ndarray, 
                                sample_rate: int, 
                                tempo: float) -> Tuple[int, int]:
        """
        Detect time signature from beat pattern.
        
        Returns:
            Tuple of (beats per measure, note value)
        """
        try:
            import librosa
            
            # Get beat frames
            beat_frames = librosa.beat.beat_track(
                y=audio, sr=sample_rate,
                hop_length=self.hop_length
            )[1]
            
            if len(beat_frames) < 4:
                return (4, 4)  # Default to common time
            
            # Compute inter-beat intervals
            beat_times = librosa.frames_to_time(beat_frames, sr=sample_rate, hop_length=self.hop_length)
            intervals = np.diff(beat_times)
            
            # Estimate measure length in beats
            # Most common patterns: 4/4 (4 beats), 3/4 (3 beats), 6/8 (6 beats)
            avg_interval = np.mean(intervals)
            measure_duration = 2.0  # Assume ~2 second measure
            
            beats_per_measure = round(measure_duration / avg_interval)
            beats_per_measure = max(min(beats_per_measure, 8), 2)  # Clamp 2-8
            
            # Common time signatures
            if beats_per_measure == 4:
                return (4, 4)
            elif beats_per_measure == 3:
                return (3, 4)
            elif beats_per_measure == 6:
                return (6, 8)
            elif beats_per_measure == 2:
                return (2, 4)
            else:
                return (beats_per_measure, 4)
                
        except Exception as e:
            logger.warning(f"Time signature detection failed: {e}")
            return (4, 4)
    
    def _fallback_analysis(self) -> TempoKeyInfo:
        """Return default values when analysis fails."""
        return TempoKeyInfo(
            tempo=120.0,
            tempo_confidence=0.3,
            key="C",
            key_mode="major",
            key_confidence=0.3,
            time_signature=(4, 4),
        )
    
    def analyze_conditioning(self, audio: np.ndarray, 
                             sample_rate: int) -> dict:
        """
        Extract structured conditioning for JASCO generation.
        
        Returns:
            Dictionary with tempo and key info for conditioning
        """
        info = self.analyze(audio, sample_rate)
        
        return {
            'tempo': info.tempo,
            'tempo_confidence': info.tempo_confidence,
            'key': info.key,
            'key_mode': info.key_mode,
            'key_confidence': info.key_confidence,
            'time_signature': info.time_signature,
            'key_string': f"{info.key}{'m' if info.key_mode == 'minor' else ''}",
        }


# Convenience function
def detect_tempo_key(audio: np.ndarray, sample_rate: int) -> TempoKeyInfo:
    """
    Quick function to detect tempo and key.
    
    Args:
        audio: Audio samples
        sample_rate: Sample rate
        
    Returns:
        TempoKeyInfo with detected tempo and key
    """
    analyzer = TempoKeyAnalyzer()
    return analyzer.analyze(audio, sample_rate)
