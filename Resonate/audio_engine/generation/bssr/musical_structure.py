"""
Musical Structure Analyzer - Extract beat, bar, and section structure.

This module provides the MusicalStructureAnalyzer class, which analyzes audio
to extract a complete musical grid (beats, bars) and structural segmentation
(verse, chorus, etc.) to guide bar-aligned processing.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import librosa

logger = logging.getLogger(__name__)


@dataclass
class MusicalStructure:
    """
    Container for musical structure information.
    
    Attributes:
        beat_times: Timestamps of all beats in seconds
        beat_frames: Frame indices of beats
        tempo: Detected tempo in BPM
        bar_times: Timestamps of bar starts in seconds
        bar_frames: Frame indices of bar starts
        beats_per_bar: Number of beats per bar (time signature numerator)
        section_boundaries: Timestamps of section boundaries (optional)
        section_labels: Labels for sections (optional)
        duration: Total duration of audio in seconds
        total_bars: Total number of bars
        total_beats: Total number of beats
    """
    beat_times: np.ndarray
    beat_frames: np.ndarray
    tempo: float
    bar_times: np.ndarray
    bar_frames: np.ndarray
    beats_per_bar: int
    duration: float
    total_bars: int
    total_beats: int
    section_boundaries: List[float]
    section_labels: List[str]


class MusicalStructureAnalyzer:
    """
    Analyzes audio to extract musical structure (beats, bars, sections).
    
    Uses librosa for beat tracking and structural segmentation to build
    a comprehensive musical grid for the audio.
    """
    
    def __init__(self, hop_length: int = 512, sample_rate: int = 44100):
        """
        Initialize the analyzer.
        
        Args:
            hop_length: FFT hop length for analysis
            sample_rate: Target sample rate for analysis
        """
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        
    def analyze(self, audio: np.ndarray, sample_rate: int = 44100) -> MusicalStructure:
        """
        Extract complete musical structure from audio.
        
        Args:
            audio: Audio samples
            sample_rate: Sample rate of input audio
            
        Returns:
            MusicalStructure object with beat/bar/section info
        """
        logger.info("Analyzing musical structure...")
        
        # Ensure correct sample rate if needed (though librosa handles this usually)
        if sample_rate != self.sample_rate:
            # We trust the input sample rate for conversion functions
            pass
            
        # 1. Beat Tracking
        tempo, beat_frames = librosa.beat.beat_track(
            y=audio, 
            sr=sample_rate, 
            hop_length=self.hop_length
        )
        
        # Convert to times
        beat_times = librosa.frames_to_time(
            beat_frames, 
            sr=sample_rate, 
            hop_length=self.hop_length
        )
        
        # 2. Bar Inference
        # Estimate time signature (beats per bar)
        # For now, we default to 4/4 as it's most common for this use case,
        # but could be enhanced with more advanced meter detection.
        beats_per_bar = 4
        
        # Assume first beat is a downbeat (simplification, can be improved with downbeat tracking)
        # Create bar grid
        bar_indices = np.arange(0, len(beat_times), beats_per_bar)
        bar_times = beat_times[bar_indices]
        bar_frames = beat_frames[bar_indices]
        
        # 3. Section Segmentation (Optional)
        # We can look for spectral novelty to find section boundaries
        section_boundaries = self._find_section_boundaries(audio, sample_rate)
        section_labels = [f"Section {i+1}" for i in range(len(section_boundaries))]
        
        duration = len(audio) / sample_rate
        
        structure = MusicalStructure(
            beat_times=beat_times,
            beat_frames=beat_frames,
            tempo=float(tempo),
            bar_times=bar_times,
            bar_frames=bar_frames,
            beats_per_bar=beats_per_bar,
            duration=duration,
            total_bars=len(bar_times),
            total_beats=len(beat_times),
            section_boundaries=section_boundaries,
            section_labels=section_labels
        )
        
        logger.info(f"Structure analysis complete: {structure.total_bars} bars at {structure.tempo:.1f} BPM")
        return structure
    
    def _find_section_boundaries(self, audio: np.ndarray, sample_rate: int) -> List[float]:
        """
        Detect section boundaries using spectral analysis.
        
        Uses spectral contrast and novelty features to find major transition points.
        """
        try:
            # Compute spectral features
            chroma = librosa.feature.chroma_cqt(y=audio, sr=sample_rate, hop_length=self.hop_length)
            
            # Recurrence matrix
            rec = librosa.segment.recurrence_matrix(chroma, mode='affinity')
            
            # Structure components (simplified segmentation)
            # For now, just return empty list or simple grid as placeholder for advanced segmentation
            # Real implementation would use librosa.segment.agglomerative or similar
            
            # Placeholder: return boundaries every ~30 seconds aligned to bars if we had bar info here
            # But since this is a private helper, we'll return an empty list for now 
            # and let the chunker handle logic if no explicit sections found.
            return []
            
        except Exception as e:
            logger.warning(f"Section segmentation failed: {e}")
            return []
