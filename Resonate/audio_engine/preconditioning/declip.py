"""
De-clipper Module
Repairs distorted peaks caused by phone Automatic Gain Control (AGC) saturation.
Part of the Pre-Conditioning Enhancement Pipeline.
"""

from dataclasses import dataclass
from typing import List, Optional
import logging

import numpy as np
from scipy.interpolate import CubicSpline, interp1d

logger = logging.getLogger(__name__)


@dataclass
class DeclipConfig:
    """Configuration for the Declipper."""
    detection_threshold: float = 0.99  # Samples above this amplitude are clipped
    interpolation_method: str = "cubic"  # cubic or linear
    max_clip_duration_ms: float = 5.0  # Maximum clip length to attempt repair
    margin_samples: int = 5  # Context samples on each side for interpolation


@dataclass
class ClipRegion:
    """Represents a detected clipped region."""
    start: int  # Sample index of clip start
    end: int    # Sample index of clip end
    duration_ms: float  # Duration in milliseconds
    severity: float  # 0-1 scale, how "hard" the clip is (ratio of clipped to total samples in region)


class Declipper:
    """
    Handles detection and repair of clipped audio regions.
    Uses spline interpolation to reconstruct saturated peaks.
    """

    def __init__(self, sample_rate: int, config: DeclipConfig = None):
        """
        Initialize the Declipper.

        Args:
            sample_rate: The sample rate of the audio.
            config: Optional configuration. Defaults to DeclipConfig().
        """
        self.sample_rate = sample_rate
        self.config = config or DeclipConfig()

    def detect_clipping(self, audio: np.ndarray) -> List[ClipRegion]:
        """
        Find contiguous regions where abs(samples) >= threshold.

        Args:
            audio: Input audio array.

        Returns:
            List of ClipRegion objects.
        """
        threshold = self.config.detection_threshold
        
        # Identify clipped samples
        is_clipped = np.abs(audio) >= threshold
        
        # Find runs of clipped samples
        # Pad with False to detect edges at start/end
        padded = np.concatenate(([False], is_clipped, [False]))
        diff = np.diff(padded.astype(int))
        
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        
        regions = []
        for start, end in zip(starts, ends):
            # Calculate duration
            length_samples = end - start
            duration_ms = (length_samples / self.sample_rate) * 1000.0
            
            # Skip if too long (likely not AGC clip but intentional or unrecoverable)
            if duration_ms > self.config.max_clip_duration_ms:
                continue
                
            # Calculate severity
            # For contiguous blocks detected by threshold, this is typically 1.0
            # unless the threshold logic changes.
            segment = audio[start:end]
            clipped_count = np.sum(np.abs(segment) >= threshold)
            severity = clipped_count / length_samples if length_samples > 0 else 0.0
            
            regions.append(ClipRegion(
                start=int(start),
                end=int(end),
                duration_ms=float(duration_ms),
                severity=float(severity)
            ))
            
        return regions

    def repair_clip(self, audio: np.ndarray, region: ClipRegion) -> np.ndarray:
        """
        Repair a single clipped region using interpolation.
        
        Args:
            audio: Input audio array.
            region: The ClipRegion to repair.
            
        Returns:
            New audio array with the region repaired.
        """
        # Work on a copy to avoid side effects if called individually
        audio_copy = audio.copy()
        self._repair_region_in_place(audio_copy, region)
        return audio_copy

    def _repair_region_in_place(self, audio: np.ndarray, region: ClipRegion) -> None:
        """
        Helper to repair a region in-place.
        """
        start = region.start
        end = region.end
        margin = self.config.margin_samples
        
        # Define context window indices
        # Ensure we don't go out of bounds
        idx_start_margin = max(0, start - margin)
        idx_end_margin = min(len(audio), end + margin)
        
        # Extract control points (indices)
        # We use points BEFORE the clip and AFTER the clip
        idx_before = np.arange(idx_start_margin, start)
        idx_after = np.arange(end, idx_end_margin)
        
        # We need points on both sides ideally, but at least enough total points
        if len(idx_before) == 0 and len(idx_after) == 0:
            return

        x_control = np.concatenate([idx_before, idx_after])
        y_control = audio[x_control]
        
        # Indices to interpolate
        x_target = np.arange(start, end)
        
        if len(x_target) == 0:
            return

        # Perform interpolation
        try:
            if self.config.interpolation_method == "cubic" and len(x_control) >= 4:
                # Cubic spline interpolation
                cs = CubicSpline(x_control, y_control)
                interpolated = cs(x_target)
            else:
                # Linear interpolation (fallback)
                # fill_value="extrapolate" handles cases where we might be at an edge
                f = interp1d(x_control, y_control, kind='linear', fill_value="extrapolate")
                interpolated = f(x_target)
            
            # Apply repaired values
            # Smooth blending is implicitly handled by using the margins as control points,
            # ensuring the spline passes through or connects to the existing waveform.
            audio[start:end] = interpolated
            
        except Exception as e:
            logger.warning(f"Failed to repair clip at {start}-{end}: {str(e)}")

    def process(self, audio: np.ndarray) -> np.ndarray:
        """
        Main entry point - process audio to remove clipping.
        
        Args:
            audio: Input audio array.
            
        Returns:
            Repaired audio array.
        """
        # Handle stereo/multichannel
        if audio.ndim > 1:
            channels = []
            for ch in range(audio.shape[1]):
                channels.append(self._process_mono(audio[:, ch]))
            return np.column_stack(channels)
        else:
            return self._process_mono(audio)

    def _process_mono(self, audio: np.ndarray) -> np.ndarray:
        """Process a single channel."""
        # Detect regions on the original audio
        regions = self.detect_clipping(audio)
        
        if not regions:
            return audio
            
        # Create a working copy
        processed_audio = audio.copy()
        
        # Repair regions
        # Note: Since we perform 1:1 replacement, indices do not shift.
        # We process in order so that close regions benefit from previous repairs if margins overlap.
        for region in regions:
            self._repair_region_in_place(processed_audio, region)
            
        return processed_audio


def declip_audio(audio: np.ndarray, sample_rate: int, threshold: float = 0.99) -> np.ndarray:
    """
    Convenience function for declipping audio.
    
    Args:
        audio: Input audio array.
        sample_rate: Sample rate of the audio.
        threshold: Detection threshold (0.0-1.0).
        
    Returns:
        Processed audio array.
    """
    config = DeclipConfig(detection_threshold=threshold)
    declipper = Declipper(sample_rate, config)
    return declipper.process(audio)
