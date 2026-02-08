"""
Melody Extractor - Extract melody contour and salience for JASCO conditioning.

Uses pitch detection (YIN or CREPE) to extract dominant melody as time-pitch
salience matrix for conditioning stem regeneration.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
import logging
import librosa

logger = logging.getLogger(__name__)


# Frequency to MIDI number conversion
def freq_to_midi(freq: float) -> float:
    """Convert frequency to MIDI note number."""
    if freq <= 0:
        return 0
    return 69 + 12 * np.log2(freq / 440.0)


def midi_to_freq(midi: float) -> float:
    """Convert MIDI note number to frequency."""
    return 440.0 * 2 ** ((midi - 69) / 12.0)


# Typical vocal range in MIDI (can be extended for instruments)
VOCAL_MIDI_MIN = 48  # C3
VOCAL_MIDI_MAX = 84  # C6


@dataclass
class MelodyContour:
    """
    Structured output for melody extraction.
    
    Attributes:
        timestamps: Time stamps for each frame (seconds)
        salience: Salience/amplitude for each pitch (0-1)
        pitch_contour: Pitch values in MIDI numbers
        confidence: Overall confidence in melody detection (0-1)
    """
    timestamps: np.ndarray
    salience: np.ndarray
    pitch_contour: np.ndarray
    confidence: float
    
    def to_pitch_sequence(self) -> list:
        """Convert to list of (time, pitch) tuples."""
        return [(t, p) for t, p, s in zip(self.timestamps, self.pitch_contour, self.salience) if s > 0.1]
    
    def to_midi_notes(self) -> np.ndarray:
        """Get pitch contour quantized to MIDI notes."""
        return np.round(self.pitch_contour).astype(int)
    
    def get_note_durations(self, threshold: float = 0.2) -> list:
        """
        Extract note events with durations.
        
        Args:
            threshold: Salience threshold for note detection
            
        Returns:
            List of (start_time, duration, midi_note) tuples
        """
        notes = []
        in_note = False
        note_start = 0
        current_note = 0
        
        for i, (t, p, s) in enumerate(zip(self.timestamps, self.pitch_contour, self.salience)):
            if s > threshold and not in_note:
                in_note = True
                note_start = t
                current_note = int(round(p))
            elif s > threshold and in_note:
                # Continuing note
                pass
            elif s <= threshold and in_note:
                # Note ended
                duration = t - note_start
                if duration > 0.1:  # Minimum note duration
                    notes.append((note_start, duration, current_note))
                in_note = False
        
        # Handle note extending to end
        if in_note and len(self.timestamps) > 0:
            duration = self.timestamps[-1] - note_start
            if duration > 0.1:
                notes.append((note_start, duration, current_note))
        
        return notes
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'timestamps': self.timestamps.tolist(),
            'salience': self.salience.tolist(),
            'pitch_contour': self.pitch_contour.tolist(),
            'confidence': self.confidence,
        }


class MelodyExtractor:
    """
    Extract melody contour and salience from audio.
    
    Uses multiple pitch detection strategies:
    1. librosa.yin - Fast YIN algorithm for pitch detection
    2. salience_estimation - Fallback using spectral prominence
    
    The melody is extracted as a time-pitch salience matrix that can be
    used to condition JASCO stem regeneration.
    """
    
    def __init__(self, frame_length: int = 2048, hop_length: int = 512):
        """
        Initialize melody extractor.
        
        Args:
            frame_length: FFT frame length
            hop_length: Hop length between frames
        """
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.sample_rate = None
        self.frequencies = None  # MIDI frequency bins
        
        # Create frequency bins for salience matrix
        self._setup_frequency_bins()
    
    def _setup_frequency_bins(self, 
                               f_min: float = 130.81,  # C3
                               f_max: float = 1046.50,  # C6
                               bins: int = 48):
        """Setup frequency bins for pitch salience matrix."""
        self.f_min = f_min
        self.f_max = f_max
        self.pitch_bins = bins
        
        # Log-spaced frequency bins
        self.frequencies = np.logspace(
            np.log2(f_min), np.log2(f_max), 
            num=bins, base=2
        )
    
    def extract(self, audio: np.ndarray, 
                sample_rate: int,
                vocal_range: bool = True) -> MelodyContour:
        """
        Extract melody contour from audio.
        
        Args:
            audio: Audio samples (float32, range [-1, 1])
            sample_rate: Sample rate in Hz
            vocal_range: If True, restrict to typical vocal range
            
        Returns:
            MelodyContour with extracted melody data
        """
        self.sample_rate = sample_rate
        
        try:
            import librosa
            
            # Try YIN pitch detection
            try:
                return self._extract_with_yin(audio, sample_rate, vocal_range)
            except Exception as e:
                logger.warning(f"YIN pitch detection failed: {e}, using salience fallback")
                return self._extract_with_salience(audio, sample_rate)
                
        except ImportError:
            logger.warning("librosa not available, using numpy fallback")
            return self._extract_fallback(audio, sample_rate)
        except Exception as e:
            logger.error(f"Melody extraction failed: {e}")
            return self._extract_fallback(audio, sample_rate)
    
    def _extract_with_yin(self, audio: np.ndarray, 
                          sample_rate: int,
                          vocal_range: bool) -> MelodyContour:
        """
        Extract melody using YIN pitch detection.
        
        Returns:
            MelodyContour with pitch and salience
        """
        import librosa
        
        # Get pitch using YIN
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y=audio,
            fmin=130.81 if vocal_range else 50,
            fmax=1046.50 if vocal_range else 2000,
            sr=sample_rate,
            frame_length=self.frame_length,
            hop_length=self.hop_length,
            center=True
        )
        
        # Convert to salience matrix
        timestamps = librosa.frames_to_time(
            np.arange(len(f0)), 
            sr=sample_rate, 
            hop_length=self.hop_length
        )
        
        # Compute salience from voiced probabilities
        salience = np.where(voiced_flag, voiced_probs, 0.0)
        
        # Convert frequencies to MIDI
        pitch_contour = np.zeros_like(f0, dtype=float)
        valid_mask = voiced_flag & (f0 > 0)
        pitch_contour[valid_mask] = freq_to_midi(f0[valid_mask])
        
        # Fill gaps using interpolation
        pitch_contour = self._interpolate_pitch(pitch_contour, voiced_flag)
        
        # Compute overall confidence
        confidence = float(np.mean(salience)) if len(salience) > 0 else 0.0
        
        return MelodyContour(
            timestamps=timestamps,
            salience=salience,
            pitch_contour=pitch_contour,
            confidence=confidence,
        )
    
    def _extract_with_salience(self, audio: np.ndarray, 
                                sample_rate: int) -> MelodyContour:
        """
        Extract melody using spectral salience estimation (fallback).
        
        Returns:
            MelodyContour with salience matrix
        """
        import librosa
        
        # Compute magnitude spectrogram
        S = np.abs(librosa.stft(
            audio, 
            n_fft=self.frame_length,
            hop_length=self.hop_length
        ))
        
        # Compute times
        timestamps = librosa.frames_to_time(
            np.arange(S.shape[1]),
            sr=sample_rate,
            hop_length=self.hop_length
        )
        
        # Create pitch salience matrix
        salience_matrix = self._compute_salience_matrix(S, sample_rate)
        
        # Extract dominant pitch per frame
        dominant_indices = np.argmax(salience_matrix, axis=0)
        pitch_contour = self._indices_to_midi(dominant_indices)
        
        # Get salience at dominant pitch
        salience = np.array([salience_matrix[idx, t] for t, idx in enumerate(dominant_indices)])
        
        # Compute confidence
        max_salience = np.max(salience_matrix, axis=0)
        confidence = float(np.mean(max_salience))
        
        return MelodyContour(
            timestamps=timestamps,
            salience=salience,
            pitch_contour=pitch_contour,
            confidence=confidence,
        )
    
    def _compute_salience_matrix(self, S: np.ndarray, 
                                   sample_rate: int) -> np.ndarray:
        """
        Compute pitch salience matrix from spectrogram.
        
        Args:
            S: Magnitude spectrogram (freq x time)
            sample_rate: Sample rate
            
        Returns:
            Salience matrix (pitch_bins x time)
        """
        freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=self.frame_length)
        
        # Map spectrogram to pitch bins
        salience_matrix = np.zeros((self.pitch_bins, S.shape[1]))
        
        for pitch_idx, pitch_freq in enumerate(self.frequencies):
            # Harmonic summation for this pitch
            for harmonic in range(1, 5):
                target_freq = pitch_freq * harmonic
                
                # Find nearest frequency bin
                freq_idx = np.argmin(np.abs(freqs - target_freq))
                
                if freq_idx < len(freqs):
                    salience_matrix[pitch_idx] += S[freq_idx] / harmonic
        
        # Normalize each time frame
        salience_matrix = salience_matrix / (np.max(salience_matrix, axis=0) + 1e-10)
        
        return salience_matrix
    
    def _indices_to_midi(self, indices: np.ndarray) -> np.ndarray:
        """Convert pitch bin indices to MIDI notes."""
        midi_range = VOCAL_MIDI_MAX - VOCAL_MIDI_MIN
        midi_notes = VOCAL_MIDI_MIN + (indices / self.pitch_bins) * midi_range
        return midi_notes
    
    def _interpolate_pitch(self, pitch_contour: np.ndarray, 
                           voiced_mask: np.ndarray,
                           max_gap: int = 10) -> np.ndarray:
        """Interpolate pitch values over unvoiced gaps."""
        result = pitch_contour.copy()
        
        n = len(pitch_contour)
        i = 0
        
        while i < n:
            if not voiced_mask[i]:
                # Find start of gap
                gap_start = i
                while i < n and not voiced_mask[i]:
                    i += 1
                gap_end = i
                
                # Interpolate if gap is not too long
                if gap_end < n and gap_end - gap_start < max_gap:
                    result[gap_start:gap_end] = np.linspace(
                        pitch_contour[gap_start - 1] if gap_start > 0 else pitch_contour[gap_end],
                        pitch_contour[gap_end] if gap_end < n else pitch_contour[gap_start - 1],
                        gap_end - gap_start
                    )
            else:
                i += 1
        
        return result
    
    def _extract_fallback(self, audio: np.ndarray, 
                          sample_rate: int) -> MelodyContour:
        """Fallback melody extraction using basic signal analysis."""
        duration = len(audio) / sample_rate
        num_frames = int(duration * sample_rate / self.hop_length)
        
        timestamps = np.linspace(0, duration, num_frames)
        
        # Very basic pitch estimation from zero crossings
        pitch_contour = np.zeros(num_frames)
        salience = np.zeros(num_frames)
        
        for i in range(num_frames):
            start = i * self.hop_length
            end = min(start + self.frame_length, len(audio))
            frame = audio[start:end]
            
            if len(frame) > 0:
                # Zero crossing rate as rough pitch proxy
                zcr = np.sum(np.diff(np.sign(frame)) != 0) / len(frame)
                # Map ZCR to approximate MIDI (very rough)
                pitch_contour[i] = 60 + zcr * 24
                salience[i] = np.std(frame)
        
        # Normalize salience
        if np.max(salience) > 0:
            salience = salience / np.max(salience)
        
        return MelodyContour(
            timestamps=timestamps,
            salience=salience,
            pitch_contour=pitch_contour,
            confidence=0.3,  # Low confidence for fallback
        )
    
    def extract_melody_conditioning(self, audio: np.ndarray, 
                                     sample_rate: int) -> dict:
        """
        Extract structured conditioning for JASCO generation.
        
        Returns:
            Dictionary with melody info for conditioning
        """
        contour = self.extract(audio, sample_rate)
        
        return {
            'timestamps': contour.timestamps,
            'pitch_contour': contour.pitch_contour,
            'salience': contour.salience,
            'confidence': contour.confidence,
            'note_events': contour.get_note_durations(),
            'midi_notes': contour.to_midi_notes().tolist(),
        }


# Convenience function
def extract_melody(audio: np.ndarray, 
                   sample_rate: int) -> MelodyContour:
    """
    Quick function to extract melody contour.
    
    Args:
        audio: Audio samples
        sample_rate: Sample rate
        
    Returns:
        MelodyContour with extracted melody
    """
    extractor = MelodyExtractor()
    return extractor.extract(audio, sample_rate)
