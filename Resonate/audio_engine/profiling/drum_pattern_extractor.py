"""
Drum Pattern Extractor - Extract drum onsets and patterns for conditioning.

Uses onset detection with beat tracking to identify kick, snare, and hi-hat
patterns. Drum types are classified by spectral characteristics.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


# Drum type names
DRUM_TYPES = ['kick', 'snare', 'hihat', 'tom', 'clap', 'cymbal']


@dataclass
class DrumOnsets:
    """
    Structured output for drum pattern extraction.
    
    Attributes:
        timestamps: Onset times in seconds
        types: Drum type for each onset
        confidence: Confidence score for each onset (0-1)
        pattern_grid: Beat-aligned pattern matrix
    """
    timestamps: np.ndarray
    types: List[str]
    confidence: np.ndarray
    pattern_grid: np.ndarray
    
    def get_kick_times(self) -> np.ndarray:
        """Get timestamps of kick drum onsets."""
        return np.array([t for t, ty in zip(self.timestamps, self.types) if ty == 'kick'])
    
    def get_snare_times(self) -> np.ndarray:
        """Get timestamps of snare drum onsets."""
        return np.array([t for t, ty in zip(self.timestamps, self.types) if ty == 'snare'])
    
    def get_hihat_times(self) -> np.ndarray:
        """Get timestamps of hi-hat onsets."""
        return np.array([t for t, ty in zip(self.timestamps, self.types) if ty == 'hihat'])
    
    def to_beat_grid(self, beat_times: np.ndarray) -> np.ndarray:
        """
        Convert onsets to beat-aligned grid.
        
        Args:
            beat_times: Times of each beat
            
        Returns:
            Binary matrix (drum_types x beats) showing onsets
        """
        num_beats = len(beat_times)
        grid = np.zeros((len(DRUM_TYPES), num_beats))
        
        for onset_time, drum_type in zip(self.timestamps, self.types):
            if drum_type in DRUM_TYPES:
                drum_idx = DRUM_TYPES.index(drum_type)
                # Find nearest beat
                beat_idx = np.argmin(np.abs(beat_times - onset_time))
                if beat_idx < num_beats:
                    grid[drum_idx, beat_idx] = 1.0
        
        return grid
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'timestamps': self.timestamps.tolist(),
            'types': self.types,
            'confidence': self.confidence.tolist(),
            'pattern_grid': self.pattern_grid.tolist() if self.pattern_grid.size > 0 else [],
        }


class DrumPatternExtractor:
    """
    Extract drum onsets and patterns from audio.
    
    Uses onset detection with beat tracking to identify drum hits.
    Drum types are classified by spectral characteristics:
    - Kick: Low frequency energy (<200 Hz), rapid decay
    - Snare: Mid-frequency burst with noise, ~200-2000 Hz
    - Hi-hat: High frequency noise, >4000 Hz
    - Tom: Similar to kick but higher frequency
    - Clap: Broadband percussive burst
    - Cymbal: Sustained high frequency energy
    """
    
    def __init__(self, 
                 frame_length: int = 2048,
                 hop_length: int = 512,
                 n_mels: int = 128):
        """
        Initialize drum pattern extractor.
        
        Args:
            frame_length: FFT frame length
            hop_length: Hop length between frames
            n_mels: Number of mel bands
        """
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.sample_rate = None
        self.beat_times = None
        
    def extract(self, audio: np.ndarray, 
                sample_rate: int,
                min_onset_interval: float = 0.05) -> DrumOnsets:
        """
        Extract drum onsets and patterns from audio.
        
        Args:
            audio: Audio samples (float32, range [-1, 1])
            sample_rate: Sample rate in Hz
            min_onset_interval: Minimum seconds between onsets
            
        Returns:
            DrumOnsets with detected drum hits
        """
        self.sample_rate = sample_rate
        
        try:
            import librosa
            
            # Get beat tracking
            tempo, beat_frames = librosa.beat.beat_track(
                y=audio, sr=sample_rate,
                hop_length=self.hop_length
            )
            self.beat_times = librosa.frames_to_time(
                beat_frames, sr=sample_rate, hop_length=self.hop_length
            )
            
            # Compute onset strength
            onset_env = librosa.onset.onset_strength(
                y=audio, sr=sample_rate,
                hop_length=self.hop_length,
                n_mels=self.n_mels
            )
            
            # Detect onsets
            onset_frames = librosa.onset.onset_detect(
                y=audio, sr=sample_rate,
                hop_length=self.hop_length,
                onset_envelope=onset_env,
                backtrack=True
            )
            
            # Filter close onsets
            onset_frames = self._filter_onset_spacing(
                onset_frames, min_onset_interval
            )
            
            # Convert to times
            onset_times = librosa.frames_to_time(
                onset_frames, sr=sample_rate, hop_length=self.hop_length
            )
            
            # Classify drum types
            drum_types = self._classify_drum_types(audio, onset_frames)
            
            # Compute confidence scores
            confidence = self._compute_confidence(onset_env, onset_frames)
            
            # Create pattern grid
            pattern_grid = self._create_pattern_grid(onset_times, drum_types)
            
            return DrumOnsets(
                timestamps=onset_times,
                types=drum_types,
                confidence=confidence,
                pattern_grid=pattern_grid,
            )
            
        except ImportError:
            logger.warning("librosa not available, using fallback detection")
            return self._extract_fallback(audio, sample_rate)
        except Exception as e:
            logger.error(f"Drum pattern extraction failed: {e}")
            return self._extract_fallback(audio, sample_rate)
    
    def _filter_onset_spacing(self, 
                               onset_frames: np.ndarray,
                               min_interval: float) -> np.ndarray:
        """Filter out onsets that are too close together."""
        if len(onset_frames) == 0:
            return onset_frames
        
        min_frames = int(min_interval * self.sample_rate / self.hop_length)
        
        if min_frames <= 0:
            return onset_frames
        
        filtered = [onset_frames[0]]
        
        for frame in onset_frames[1:]:
            if frame - filtered[-1] >= min_frames:
                filtered.append(frame)
        
        return np.array(filtered)
    
    def _classify_drum_types(self, audio: np.ndarray, 
                              onset_frames: np.ndarray) -> List[str]:
        """
        Classify drum types based on spectral characteristics.
        
        Args:
            audio: Full audio signal
            onset_frames: Frame indices of onsets
            
        Returns:
            List of drum type names
        """
        import librosa
        
        # Compute spectrogram for analysis
        S = np.abs(librosa.stft(
            audio, n_fft=self.frame_length, hop_length=self.hop_length
        ))
        freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=self.frame_length)
        
        drum_types = []
        
        for frame in onset_frames:
            # Get spectrum around onset
            start_frame = max(0, frame - 2)
            end_frame = min(frame + 8, S.shape[1])
            frame_spectrum = np.mean(S[:, start_frame:end_frame], axis=1)
            
            # Normalize
            frame_spectrum = frame_spectrum / (np.max(frame_spectrum) + 1e-10)
            
            # Frequency band energies
            low_energy = np.mean(frame_spectrum[freqs < 200])
            mid_energy = np.mean(frame_spectrum[(freqs >= 200) & (freqs < 2000)])
            high_energy = np.mean(frame_spectrum[(freqs >= 2000) & (freqs < 6000)])
            noise_energy = np.mean(frame_spectrum[freqs >= 6000])
            
            # Classify based on spectral characteristics
            if low_energy > 0.5 and mid_energy < 0.3:
                drum_type = 'kick'
            elif mid_energy > 0.4 and noise_energy > 0.2:
                drum_type = 'snare'
            elif noise_energy > 0.5 and high_energy > 0.3:
                drum_type = 'hihat'
            elif low_energy > 0.3 and mid_energy > 0.3:
                drum_type = 'tom'
            elif mid_energy > 0.5 and noise_energy > 0.3:
                drum_type = 'clap'
            elif high_energy > 0.4 and noise_energy > 0.3:
                drum_type = 'cymbal'
            else:
                # Default classification based on dominant energy
                energies = {
                    'kick': low_energy,
                    'snare': mid_energy,
                    'hihat': noise_energy,
                }
                drum_type = max(energies, key=energies.get)
            
            drum_types.append(drum_type)
        
        return drum_types
    
    def _compute_confidence(self, onset_env: np.ndarray, 
                            onset_frames: np.ndarray) -> np.ndarray:
        """Compute confidence scores for each onset."""
        if len(onset_frames) == 0:
            return np.array([])
        
        # Normalize onset envelope
        onset_max = np.max(onset_env)
        if onset_max > 0:
            normalized = onset_env / onset_max
        else:
            normalized = onset_env
        
        # Get confidence at onset frames
        confidence = np.array([normalized[f] for f in onset_frames])
        
        # Ensure minimum confidence
        confidence = np.clip(confidence, 0.1, 1.0)
        
        return confidence
    
    def _create_pattern_grid(self, 
                              onset_times: np.ndarray,
                              drum_types: List[str]) -> np.ndarray:
        """
        Create beat-aligned pattern grid.
        
        Returns:
            Matrix of (drum_types x beats)
        """
        if len(onset_times) == 0 or self.beat_times is None:
            return np.zeros((len(DRUM_TYPES), 16))  # Default 16-beat grid
        
        num_beats = max(16, len(self.beat_times))
        grid = np.zeros((len(DRUM_TYPES), num_beats))
        
        for onset_time, drum_type in zip(onset_times, drum_types):
            if drum_type in DRUM_TYPES:
                drum_idx = DRUM_TYPES.index(drum_type)
                # Find nearest beat
                beat_idx = np.argmin(np.abs(self.beat_times - onset_time))
                if beat_idx < num_beats:
                    grid[drum_idx, beat_idx] = 1.0
        
        return grid
    
    def _extract_fallback(self, audio: np.ndarray, 
                          sample_rate: int) -> DrumOnsets:
        """Fallback drum extraction using basic signal analysis."""
        # Simple onset detection using amplitude envelope
        frame_length = int(0.02 * sample_rate)  # 20ms frames
        hop_length = frame_length // 2
        
        energy = np.array([
            np.sum(audio[i:i+frame_length]**2) 
            for i in range(0, len(audio) - frame_length, hop_length)
        ])
        
        # Find energy peaks
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(energy, distance=int(0.05 * sample_rate / hop_length))
        
        # Convert to time
        onset_times = peaks * hop_length / sample_rate
        
        # Simple classification based on energy
        drum_types = ['kick'] * len(onset_times)
        confidence = np.clip(energy[peaks] / (np.max(energy) + 1e-10), 0.1, 1.0)
        
        # Default pattern grid
        pattern_grid = np.zeros((len(DRUM_TYPES), 16))
        
        return DrumOnsets(
            timestamps=onset_times,
            types=drum_types,
            confidence=confidence,
            pattern_grid=pattern_grid,
        )
    
    def extract_drum_conditioning(self, audio: np.ndarray, 
                                   sample_rate: int) -> dict:
        """
        Extract structured conditioning for JASCO generation.
        
        Returns:
            Dictionary with drum info for conditioning
        """
        onsets = self.extract(audio, sample_rate)
        
        return {
            'onset_times': onsets.timestamps,
            'drum_types': onsets.types,
            'confidence': onsets.confidence,
            'pattern_grid': onsets.pattern_grid,
            'kick_times': onsets.get_kick_times(),
            'snare_times': onsets.get_snare_times(),
            'hihat_times': onsets.get_hihat_times(),
        }


# Convenience function
def extract_drum_pattern(audio: np.ndarray, 
                         sample_rate: int) -> DrumOnsets:
    """
    Quick function to extract drum patterns.
    
    Args:
        audio: Audio samples
        sample_rate: Sample rate
        
    Returns:
        DrumOnsets with detected drum hits
    """
    extractor = DrumPatternExtractor()
    return extractor.extract(audio, sample_rate)
