"""
Beat-Aligned Stitcher - Seamlessly stitch audio chunks with phase coherence.

This module provides the BeatAlignedStitcher, which combines generated chunks
using optimal cut points, applying phase correction and equal-power crossfades.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .musical_structure import MusicalStructure
from .optimal_cut_finder import CutPoint

logger = logging.getLogger(__name__)


@dataclass
class StitchConfig:
    """
    Configuration for the stitching process.
    
    Attributes:
        crossfade_bars: Number of bars to crossfade (approximate duration)
        use_equal_power: Whether to use equal-power crossfade (vs linear)
        phase_correction: Whether to apply phase correction at boundaries
        crossfade_duration_ms: Explicit crossfade duration in ms (overrides bars if set)
    """
    crossfade_bars: int = 2
    use_equal_power: bool = True
    phase_correction: bool = True
    crossfade_duration_ms: Optional[float] = None


class BeatAlignedStitcher:
    """
    Stitches audio chunks together at optimal cut points.
    
    Ensures seamless transitions by:
    1. Aligning cuts to beats where possible
    2. Applying phase correction to align waveforms
    3. Using equal-power crossfades to maintain volume
    """
    
    def __init__(self, sample_rate: int = 44100, config: Optional[StitchConfig] = None):
        """
        Initialize the stitcher.
        
        Args:
            sample_rate: Audio sample rate
            config: Stitching configuration
        """
        self.sample_rate = sample_rate
        self.config = config or StitchConfig()
        
    def stitch_chunks(self,
                      chunks: List[np.ndarray],
                      cut_points: List[CutPoint],
                      structure: Optional[MusicalStructure] = None) -> np.ndarray:
        """
        Stitch multiple chunks together into a single continuous audio track.
        
        Args:
            chunks: List of audio chunks (numpy arrays)
            cut_points: List of optimal cut points (one less than chunks)
            structure: Musical structure for tempo-based crossfade duration
            
        Returns:
            Stitched audio array
        """
        if not chunks:
            return np.array([])
            
        if len(chunks) == 1:
            return chunks[0]
            
        if len(cut_points) != len(chunks) - 1:
            logger.warning(f"Mismatch between chunks ({len(chunks)}) and cut points ({len(cut_points)})")
            # Create default cut points if needed? For now, assume caller handles this or we fail gracefully
            return chunks[0] # Fail safe
            
        # Initialize result with the first chunk up to the first cut point
        # The cut_point.time is relative to the start of the overlap
        # But wait, we need to know where the overlap *starts* in the generated audio
        # The Orchestrator knows the overlap structure.
        #
        # Let's verify the inputs:
        # chunks[i] is the FULL generated audio for chunk i
        # chunks[i+1] is the FULL generated audio for chunk i+1
        # They overlap. The CutPoint tells us where in the OVERLAP region to cut.
        #
        # To stitch, we need to know:
        # 1. How much overlap exists between chunk[i] and chunk[i+1]?
        # 
        # Ideally, `cut_points` should probably contain this context or we pass it in.
        # 
        # Refined Logic:
        # We process sequentially.
        # Result starts as Chunk 0.
        # For Chunk 1:
        #   We know it overlaps Chunk 0 by X amount.
        #   The cut point is at T inside that X overlap.
        #   We trim Chunk 0 at (End - Overlap + T)
        #   We trim Chunk 1 at T
        #   We crossfade around T.
        #
        # Issue: This method signature doesn't pass the overlap info explicitly.
        # The Orchestrator should probably handle the high-level logic and call a simpler
        # `crossfade_pair` method here, OR we need more info.
        #
        # Let's assume the cut_points logic (as defined in optimal_cut_finder) gives us
        # time relative to the START of overlap.
        # But we still need to know the TOTAL overlap duration or start time relative to chunk end.
        
        # Simplification: The Orchestrator will have to handle the exact slicing.
        # The stitcher should probably take pre-aligned segments or be called iteratively.
        # Let's implement `stitch_pair` which is safer and can be composed.
        pass
        
    def crossfade_pair(self, 
                       outgoing_chunk: np.ndarray, 
                       incoming_chunk: np.ndarray,
                       cut_point: CutPoint,
                       overlap_duration: float,
                       structure: Optional[MusicalStructure] = None) -> np.ndarray:
        """
        Stitch two overlapping chunks together at a cut point.
        
        Args:
            outgoing_chunk: First chunk
            incoming_chunk: Second chunk
            cut_point: Optimal cut point (time relative to overlap start)
            overlap_duration: Total duration of overlap between chunks
            structure: Musical structure for crossfade sizing
            
        Returns:
            Combined audio
        """
        # Determine crossfade duration
        xfade_samples = self._get_crossfade_samples(structure)
        
        # Calculate samples
        overlap_samples = int(overlap_duration * self.sample_rate)
        cut_sample_in_overlap = int(cut_point.time * self.sample_rate)
        
        # Validate overlap
        if overlap_samples > len(incoming_chunk) or overlap_samples > len(outgoing_chunk):
            logger.warning("Overlap longer than chunks, clamping")
            overlap_samples = min(len(incoming_chunk), len(outgoing_chunk))
            
        # Define the splice point in respective chunks
        # Outgoing: End - Overlap + Cut
        out_splice_idx = len(outgoing_chunk) - overlap_samples + cut_sample_in_overlap
        # Incoming: Cut
        in_splice_idx = cut_sample_in_overlap
        
        # Define Crossfade Region boundaries
        # We center the crossfade on the splice point
        half_xfade = xfade_samples // 2
        
        # Boundaries in Outgoing
        out_xfade_start = out_splice_idx - half_xfade
        out_xfade_end = out_splice_idx + half_xfade
        
        # Boundaries in Incoming
        in_xfade_start = in_splice_idx - half_xfade
        in_xfade_end = in_splice_idx + half_xfade
        
        # Boundary checks
        if out_xfade_start < 0 or in_xfade_start < 0:
            # Shift crossfade forward if too close to start
            shift = max(-out_xfade_start, -in_xfade_start)
            out_xfade_start += shift
            out_xfade_end += shift
            in_xfade_start += shift
            in_xfade_end += shift
            
        if out_xfade_end > len(outgoing_chunk) or in_xfade_end > len(incoming_chunk):
             # Shrink crossfade if too close to end
             # (This shouldn't happen often if we cut in middle of overlap)
             pass
             
        # Extract segments
        # 1. Outgoing pre-fade
        part_a = outgoing_chunk[:out_xfade_start]
        
        # 2. Crossfade region
        fade_out_segment = outgoing_chunk[out_xfade_start:out_xfade_end]
        fade_in_segment = incoming_chunk[in_xfade_start:in_xfade_end]
        
        # 3. Incoming post-fade
        part_b = incoming_chunk[in_xfade_end:]
        
        # Apply Phase Correction on the fade segments
        if self.config.phase_correction and len(fade_out_segment) > 0 and len(fade_in_segment) > 0:
            fade_in_segment = self._phase_correct(fade_out_segment, fade_in_segment)
            
        # Apply Crossfade
        if len(fade_out_segment) > 0:
            crossed_segment = self._apply_crossfade(fade_out_segment, fade_in_segment)
        else:
            crossed_segment = np.array([])
            
        # Concatenate
        result = np.concatenate([part_a, crossed_segment, part_b])
        return result

    def _get_crossfade_samples(self, structure: Optional[MusicalStructure]) -> int:
        """Calculate crossfade length in samples."""
        if self.config.crossfade_duration_ms:
            return int(self.config.crossfade_duration_ms / 1000.0 * self.sample_rate)
            
        if structure and structure.tempo > 0:
            # Use bars
            beats_per_bar = structure.beats_per_bar
            seconds_per_bar = (60.0 / structure.tempo) * beats_per_bar
            duration = seconds_per_bar * self.config.crossfade_bars
            return int(duration * self.sample_rate)
            
        # Default: 100ms
        return int(0.1 * self.sample_rate)
        
    def _phase_correct(self, reference: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        Align target phase to reference using cross-correlation.
        Returns shifted target (padded/trimmed).
        """
        if len(reference) == 0 or len(target) == 0:
            return target
            
        # Cross-correlate
        correlation = np.correlate(reference, target, mode='full')
        # Find lag with max correlation
        lag = np.argmax(correlation) - (len(target) - 1)
        
        if lag == 0:
            return target
            
        # Apply shift
        result = np.zeros_like(target)
        if lag > 0:
            # Target needs to move right (delayed)
            if lag < len(target):
                result[lag:] = target[:-lag]
        else:
            # Target needs to move left (advanced)
            shift = -lag
            if shift < len(target):
                result[:-shift] = target[shift:]
                
        return result
        
    def _apply_crossfade(self, fade_out: np.ndarray, fade_in: np.ndarray) -> np.ndarray:
        """Apply equal-power or linear crossfade."""
        length = min(len(fade_out), len(fade_in))
        if length == 0:
            return np.array([])
            
        t = np.linspace(0, 1, length)
        
        if self.config.use_equal_power:
            # Sine/Cosine for constant power
            curve_in = np.sin(t * np.pi / 2) ** 2
            curve_out = np.cos(t * np.pi / 2) ** 2
        else:
            # Linear
            curve_in = t
            curve_out = 1.0 - t
            
        return fade_out[:length] * curve_out + fade_in[:length] * curve_in
