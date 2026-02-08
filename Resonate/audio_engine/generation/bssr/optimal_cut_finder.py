"""
Optimal Cut Finder - Find the best transition point between overlapping audio chunks.

This module provides the OptimalCutFinder, which analyzes overlapping audio regions
to find the point of minimum spectral distance for seamless crossfading.
"""

import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import librosa

logger = logging.getLogger(__name__)


@dataclass
class CutPoint:
    """
    Represents an optimal cut point between two chunks.
    
    Attributes:
        time: Cut time in seconds relative to the start of the overlap
        frame: Frame index of the cut
        spectral_distance: Spectral distance at the cut point (lower is better)
        is_beat_aligned: Whether the cut is aligned to a beat
        confidence: Confidence score (0-1) based on distance
    """
    time: float
    frame: int
    spectral_distance: float
    is_beat_aligned: bool
    confidence: float


class OptimalCutFinder:
    """
    Finds the optimal point to cut/crossfade between overlapping audio chunks.
    
    Uses spectral distance minimization to find where two audio segments are
    most similar, reducing audible discontinuities.
    """
    
    def __init__(self, sample_rate: int = 44100, n_fft: int = 2048):
        """
        Initialize the finder.
        
        Args:
            sample_rate: Sample rate of audio
            n_fft: FFT window size for spectral analysis
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = n_fft // 4
        
    def find_optimal_cut(self,
                         chunk_a: np.ndarray,
                         chunk_b: np.ndarray,
                         overlap_duration: float,
                         beat_times: np.ndarray) -> CutPoint:
        """
        Find the optimal cut point in the overlap region.
        
        Args:
            chunk_a: First audio chunk (outgoing)
            chunk_b: Second audio chunk (incoming)
            overlap_duration: Duration of overlap in seconds
            beat_times: Array of beat timestamps in the overlap region
            
        Returns:
            CutPoint object with details of the best cut
        """
        # Validate inputs
        if len(chunk_a) == 0 or len(chunk_b) == 0:
            logger.warning("Empty chunks provided to OptimalCutFinder")
            return self._create_default_cut(overlap_duration)
            
        overlap_samples = int(overlap_duration * self.sample_rate)
        
        # Ensure we have enough audio
        if len(chunk_a) < overlap_samples or len(chunk_b) < overlap_samples:
            logger.warning("Chunks shorter than requested overlap")
            overlap_samples = min(len(chunk_a), len(chunk_b))
            
        if overlap_samples == 0:
            return self._create_default_cut(0.0)
            
        # Extract overlap regions
        # Chunk A: End of chunk
        a_overlap = chunk_a[-overlap_samples:]
        
        # Chunk B: Start of chunk
        b_overlap = chunk_b[:overlap_samples]
        
        # Compute spectral representations
        try:
            stft_a = np.abs(librosa.stft(a_overlap, n_fft=self.n_fft, 
                                          hop_length=self.hop_length))
            stft_b = np.abs(librosa.stft(b_overlap, n_fft=self.n_fft,
                                          hop_length=self.hop_length))
            
            # Compute frame-by-frame spectral distance
            distances = self._compute_spectral_distances(stft_a, stft_b)
            
            # Apply bias towards center of overlap (avoid cuts at very edges)
            center_bias = self._create_center_bias(len(distances))
            biased_distances = distances * (1.0 + center_bias)
            
            # Find minimum distance frame
            min_frame = np.argmin(biased_distances)
            min_time = librosa.frames_to_time(min_frame, sr=self.sample_rate,
                                               hop_length=self.hop_length)
            
            # Snap to nearest beat if close enough
            cut_time, is_beat = self._snap_to_beat(min_time, beat_times)
            
            # Recalculate frame if snapped
            if is_beat:
                cut_frame = librosa.time_to_frames(cut_time, sr=self.sample_rate,
                                                    hop_length=self.hop_length)
                # Ensure frame is within bounds
                cut_frame = min(cut_frame, len(distances) - 1)
                final_distance = distances[cut_frame]
            else:
                cut_frame = min_frame
                final_distance = distances[min_frame]
            
            # Calculate confidence (inverse of distance, normalized)
            # Assuming distance is roughly cosine distance (0-1 range usually, but depends on metric)
            confidence = max(0.0, 1.0 - final_distance)
            
            logger.debug(f"Found optimal cut at {cut_time:.3f}s (beat_aligned={is_beat}, conf={confidence:.2f})")
            
            return CutPoint(
                time=cut_time,
                frame=int(cut_frame),
                spectral_distance=float(final_distance),
                is_beat_aligned=is_beat,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Optimal cut finding failed: {e}")
            return self._create_default_cut(overlap_duration)
    
    def _compute_spectral_distances(self, stft_a: np.ndarray, stft_b: np.ndarray) -> np.ndarray:
        """
        Compute spectral distance between corresponding frames.
        Uses cosine distance between magnitude spectra.
        """
        # Shape is (freq_bins, frames)
        frames = min(stft_a.shape[1], stft_b.shape[1])
        distances = np.zeros(frames)
        
        for i in range(frames):
            vec_a = stft_a[:, i]
            vec_b = stft_b[:, i]
            
            # Normalize vectors
            norm_a = np.linalg.norm(vec_a)
            norm_b = np.linalg.norm(vec_b)
            
            if norm_a > 1e-6 and norm_b > 1e-6:
                # Cosine similarity
                similarity = np.dot(vec_a, vec_b) / (norm_a * norm_b)
                # Cosine distance
                distances[i] = 1.0 - similarity
            else:
                distances[i] = 1.0 # Max distance for silent frames vs non-silent
                
        return distances
    
    def _create_center_bias(self, length: int) -> np.ndarray:
        """
        Create a bias curve that penalizes edges.
        Parabolic curve: 0 at center, higher at edges.
        """
        x = np.linspace(-1, 1, length)
        # y = x^2 * strength
        # e.g., at edges (x=1), penalty is +0.5 distance
        return x**2 * 0.5
    
    def _snap_to_beat(self, time: float, beat_times: np.ndarray, 
                       tolerance: float = 0.1) -> Tuple[float, bool]:
        """Snap time to nearest beat if within tolerance."""
        if beat_times is None or len(beat_times) == 0:
            return time, False
        
        # Find nearest beat
        diffs = np.abs(beat_times - time)
        nearest_idx = np.argmin(diffs)
        nearest_diff = diffs[nearest_idx]
        
        if nearest_diff <= tolerance:
            return beat_times[nearest_idx], True
            
        return time, False
        
    def _create_default_cut(self, overlap_duration: float) -> CutPoint:
        """Create a default cut point in the middle of overlap."""
        return CutPoint(
            time=overlap_duration / 2,
            frame=0,
            spectral_distance=1.0,
            is_beat_aligned=False,
            confidence=0.0
        )
