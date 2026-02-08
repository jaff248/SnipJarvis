"""
Blender - Audio crossfade blending for stem regeneration.

This module provides the Blender class for smooth blending of regenerated
audio segments back into original audio. It includes crossfading,
loudness matching, and edge smoothing to prevent artifacts at boundaries.

Features:
- Equal-power crossfades for consistent loudness
- Multiple segment blending with independent crossfades
- RMS loudness matching between original and regenerated
- Edge smoothing to prevent clicking/popping
- Optional phase alignment at boundaries
"""

import logging
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class Blender:
    """
    Handles audio blending operations for stem regeneration.
    
    Provides smooth transitions between original and regenerated audio
    using crossfades, loudness matching, and edge smoothing.
    
    Features:
    - Equal-power crossfades (squared sine) for consistent loudness
    - Multiple regenerated segment blending
    - Automatic loudness matching
    - Edge smoothing at transition boundaries
    - Optional phase alignment
    
    Example:
        >>> blender = Blender()
        >>> result = blender.crossfade(original, regenerated, 2.0, 4.0, fade_duration=0.1)
    """
    
    # Default crossfade duration in seconds
    DEFAULT_FADE_DURATION = 0.1  # 100ms
    
    # Default edge smoothing duration
    DEFAULT_EDGE_DURATION = 0.05  # 50ms
    
    def __init__(self):
        """Initialize the blender."""
        logger.info("Initialized Blender")
    
    def crossfade(
        self,
        original: np.ndarray,
        regenerated: np.ndarray,
        start_time: float,
        end_time: float,
        fade_duration: float = 0.1,
    ) -> np.ndarray:
        """
        Create a smooth crossfade between original and regenerated audio.
        
        Uses an equal-power crossfade (squared sine curve) to ensure
        consistent perceived loudness throughout the transition.
        
        Args:
            original: Original audio samples
            regenerated: Regenerated audio samples
            start_time: Start time of crossfade region in seconds
            end_time: End time of crossfade region in seconds
            fade_duration: Duration of crossfade in seconds
            
        Returns:
            Audio with crossfade applied in the specified region
            
        Note:
            The crossfade creates three regions:
            1. Before fade_start: original audio unchanged
            2. Fade region: equal-power crossfade
            3. After fade_end: regenerated audio unchanged
        """
        # Validate inputs
        if len(original) == 0 or len(regenerated) == 0:
            logger.warning("Empty audio in crossfade, returning original")
            return original.copy()
        
        # Ensure regenerated matches original length if needed
        if len(regenerated) < len(original):
            # Pad regenerated to match
            padding = np.zeros(len(original) - len(regenerated), dtype=np.float32)
            regenerated = np.concatenate([regenerated, padding])
        elif len(regenerated) > len(original):
            # Trim regenerated
            regenerated = regenerated[:len(original)]
        
        result = original.copy()
        
        # Convert times to sample indices
        sample_rate = 44100  # Assume standard sample rate
        
        # Calculate fade boundaries
        fade_samples = int(fade_duration * sample_rate)
        
        # Fade starts at start_time and ends at start_time + fade_duration
        fade_start_sample = int(start_time * sample_rate)
        fade_end_sample = int((start_time + fade_duration) * sample_rate)
        
        # Fade-in region: end_time - fade_duration to end_time
        fade_in_start = int((end_time - fade_duration) * sample_rate)
        fade_in_end = int(end_time * sample_rate)
        
        # Apply crossfade
        fade_start_idx = max(0, fade_start_sample)
        fade_end_idx = min(len(result), fade_end_sample)
        fade_in_start_idx = max(0, fade_in_start)
        fade_in_end_idx = min(len(result), fade_in_end)
        
        # Fade-out original (crossfade in regenerated)
        if fade_end_idx > fade_start_idx:
            fade_out_length = fade_end_idx - fade_start_idx
            if fade_out_length > 0:
                # Create equal-power crossfade curve (squared sine)
                fade_curve = self._create_crossfade_curve(fade_out_length)
                
                # Apply fade out to original
                result[fade_start_idx:fade_end_idx] = (
                    result[fade_start_idx:fade_end_idx] * (1 - fade_curve) +
                    regenerated[fade_start_idx:fade_end_idx] * fade_curve
                )
        
        # Fade-in regenerated (crossfade from original)
        if fade_in_end_idx > fade_in_start_idx:
            fade_in_length = fade_in_end_idx - fade_in_start_idx
            if fade_in_length > 0:
                # Create equal-power crossfade curve
                fade_curve = self._create_crossfade_curve(fade_in_length)
                
                # Apply fade in to regenerated
                result[fade_in_start_idx:fade_in_end_idx] = (
                    original[fade_in_start_idx:fade_in_end_idx] * (1 - fade_curve) +
                    regenerated[fade_in_start_idx:fade_in_end_idx] * fade_curve
                )
        
        # Apply direct replacement in the middle region
        middle_start = fade_end_sample
        middle_end = max(fade_in_start, fade_end_sample)
        
        if middle_end > middle_start:
            result[middle_start:middle_end] = regenerated[middle_start:middle_end]
        
        return result
    
    def _create_crossfade_curve(self, length: int) -> np.ndarray:
        """
        Create an equal-power crossfade curve.
        
        Uses a squared sine curve for equal power, which ensures
        the total power remains constant during crossfade.
        
        Args:
            length: Number of samples for the curve
            
        Returns:
            Crossfade curve array (0 to 1)
        """
        if length <= 0:
            return np.array([])
        
        # Create linear fade (0 to 1)
        linear = np.linspace(0, 1, length)
        
        # Apply squared sine for equal power
        # This gives a smooth curve that maintains constant power
        curve = np.sin(linear * np.pi / 2) ** 2
        
        return curve
    
    def blend_regions(
        self,
        original: np.ndarray,
        regenerated_segments: List[Tuple[np.ndarray, float, float]],
        sample_rate: int = 44100,
        fade_duration: Optional[float] = None,
    ) -> np.ndarray:
        """
        Blend multiple regenerated segments into original audio.
        
        Each segment is blended with its own crossfade. Segments are
        processed in order, with later segments taking precedence
        for overlapping regions.
        
        Args:
            original: Original audio samples
            regenerated_segments: List of (audio, start_time, end_time) tuples
            sample_rate: Sample rate in Hz
            fade_duration: Crossfade duration (default: 0.1s)
            
        Returns:
            Original audio with regenerated segments blended in
            
        Example:
            >>> segments = [
            ...     (regenerated_drums, 2.0, 4.0),
            ...     (regenerated_vocals, 1.5, 3.5),
            ... ]
            >>> result = blender.blend_regions(original, segments, 44100)
        """
        if not regenerated_segments:
            logger.warning("No segments to blend, returning original")
            return original.copy()
        
        fade_duration = fade_duration or self.DEFAULT_FADE_DURATION
        
        # Start with original
        result = original.copy()
        
        # Sort segments by start time
        sorted_segments = sorted(
            regenerated_segments,
            key=lambda x: x[1]  # Sort by start_time
        )
        
        # Process each segment
        for i, (segment_audio, start_time, end_time) in enumerate(sorted_segments):
            try:
                # Apply crossfade for this segment
                result = self.crossfade(
                    original=result,
                    regenerated=segment_audio,
                    start_time=start_time,
                    end_time=end_time,
                    fade_duration=fade_duration,
                )
                
                logger.debug(
                    f"Blended segment {i+1}/{len(sorted_segments)}: "
                    f"{start_time:.2f}s - {end_time:.2f}s"
                )
                
            except Exception as e:
                logger.error(f"Failed to blend segment {i}: {e}")
                continue
        
        return result
    
    def smooth_edges(
        self,
        audio: np.ndarray,
        regions: List[Tuple[float, float]],
        edge_duration: float = 0.05,
        sample_rate: int = 44100,
    ) -> np.ndarray:
        """
        Apply short smoothing at region boundaries.
        
        Applies a very short fade (50ms default) at the start and end
        of each region to prevent clicking and popping artifacts.
        
        Args:
            audio: Audio samples
            regions: List of (start_time, end_time) tuples for regions
            edge_duration: Duration of smoothing in seconds
            sample_rate: Sample rate in Hz
            
        Returns:
            Audio with smoothed edges at region boundaries
            
        Note:
            This applies:
            - Fade-in at region start (removes click onset)
            - Fade-out at region end (removes click offset)
        """
        if not regions:
            return audio.copy()
        
        result = audio.copy()
        edge_samples = int(edge_duration * sample_rate)
        
        for start_time, end_time in regions:
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            
            # Fade-in at start
            fade_in_end = min(start_sample + edge_samples, len(result))
            if fade_in_end > start_sample:
                fade_length = fade_in_end - start_sample
                if fade_length > 0:
                    # Create smooth fade-in curve
                    fade_curve = self._create_fade_in_curve(fade_length)
                    result[start_sample:fade_in_end] *= fade_curve
            
            # Fade-out at end
            fade_out_start = max(end_sample - edge_samples, 0)
            if fade_out_start < end_sample:
                fade_length = end_sample - fade_out_start
                if fade_length > 0:
                    # Create smooth fade-out curve
                    fade_curve = self._create_fade_out_curve(fade_length)
                    result[fade_out_start:end_sample] *= fade_curve
        
        return result
    
    def _create_fade_in_curve(self, length: int) -> np.ndarray:
        """
        Create a smooth fade-in curve.
        
        Args:
            length: Number of samples
            
        Returns:
            Fade-in curve (0 to 1)
        """
        if length <= 0:
            return np.array([])
        
        # Use cosine curve for smooth fade-in
        curve = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, length)))
        return curve
    
    def _create_fade_out_curve(self, length: int) -> np.ndarray:
        """
        Create a smooth fade-out curve.
        
        Args:
            length: Number of samples
            
        Returns:
            Fade-out curve (1 to 0)
        """
        if length <= 0:
            return np.array([])
        
        # Use cosine curve for smooth fade-out
        curve = 0.5 * (1 + np.cos(np.pi * np.linspace(0, 1, length)))
        return curve
    
    def match_loudness(
        self,
        original_region: np.ndarray,
        regenerated_region: np.ndarray,
        target_db: float = -3.0,
    ) -> np.ndarray:
        """
        Match RMS loudness between original and regenerated audio.
        
        Calculates the RMS level of both signals and applies gain
        to the regenerated region to match the original's loudness.
        
        Args:
            original_region: Original audio samples
            regenerated_region: Regenerated audio samples
            target_db: Target loudness in dB (relative to peak)
            
        Returns:
            Loudness-matched regenerated audio
            
        Note:
            Uses RMS (Root Mean Square) for loudness measurement,
            which correlates well with perceived loudness.
        """
        if len(regenerated_region) == 0:
            logger.warning("Empty regenerated region, cannot match loudness")
            return regenerated_region
        
        # Calculate RMS levels
        original_rms = self._rms_level(original_region)
        regenerated_rms = self._rms_level(regenerated_region)
        
        if original_rms < 1e-10:
            logger.warning("Original region too quiet, skipping loudness match")
            return regenerated_region
        
        if regenerated_rms < 1e-10:
            logger.warning("Regenerated region silent, cannot match loudness")
            return regenerated_region
        
        # Calculate gain to match loudness
        gain = original_rms / regenerated_rms
        
        # Apply gain
        matched = regenerated_region * gain
        
        # Log the adjustment
        original_db = 20 * np.log10(original_rms + 1e-10)
        matched_db = 20 * np.log10(np.abs(matched).max() + 1e-10)
        logger.debug(
            f"Loudness match: {original_db:.1f}dB -> {matched_db:.1f}dB "
            f"(gain: {20*np.log10(gain):.1f}dB)"
        )
        
        return matched
    
    def _rms_level(self, audio: np.ndarray) -> float:
        """
        Calculate RMS level of audio.
        
        Args:
            audio: Audio samples
            
        Returns:
            RMS level (0 to 1)
        """
        return np.sqrt(np.mean(audio.astype(np.float32) ** 2 + 1e-10))
    
    def phase_align(
        self,
        original: np.ndarray,
        regenerated: np.ndarray,
        sample_rate: int = 44100,
    ) -> np.ndarray:
        """
        Align phase at boundaries for smoother transitions.
        
        Attempts to align the phase of original and regenerated audio
        at the boundaries to reduce discontinuities. Uses cross-correlation
        to find the optimal offset.
        
        Args:
            original: Original audio samples
            regenerated: Regenerated audio samples
            sample_rate: Sample rate in Hz
            
        Returns:
            Phase-aligned regenerated audio
            
        Note:
            This is a best-effort alignment. It may not work well if
            the original and regenerated audio are very different.
        """
        if len(original) == 0 or len(regenerated) == 0:
            return regenerated.copy()
        
        # Use a short window for alignment (100ms)
        window_size = int(0.1 * sample_rate)
        
        # Take the end of original and start of regenerated
        original_end = original[-window_size:] if len(original) > window_size else original
        regenerated_start = regenerated[:window_size] if len(regenerated) > window_size else regenerated
        
        # Cross-correlation to find optimal offset
        correlation = np.correlate(original_end, regenerated_start, mode='full')
        optimal_offset = np.argmax(correlation) - (len(regenerated_start) - 1)
        
        if optimal_offset > 0:
            # Shift regenerated forward (add silence at start)
            shifted = np.zeros_like(regenerated)
            shifted[optimal_offset:] = regenerated[:-optimal_offset]
            logger.debug(f"Phase aligned: shifted forward by {optimal_offset} samples")
            return shifted
        elif optimal_offset < 0:
            # Shift regenerated backward (trim start)
            shifted = regenerated[-optimal_offset:]
            logger.debug(f"Phase aligned: shifted backward by {-optimal_offset} samples")
            # Pad to original length if needed
            if len(shifted) < len(regenerated):
                padding = np.zeros(len(regenerated) - len(shifted), dtype=np.float32)
                shifted = np.concatenate([shifted, padding])
            return shifted[:len(regenerated)]
        else:
            logger.debug("No phase alignment needed")
            return regenerated.copy()
    
    def blend_with_loudness_match(
        self,
        original: np.ndarray,
        regenerated: np.ndarray,
        start_time: float,
        end_time: float,
        fade_duration: float = 0.1,
        sample_rate: int = 44100,
    ) -> np.ndarray:
        """
        Blend regenerated region with loudness matching.
        
        Combines crossfading and loudness matching for the best
        possible transition between original and regenerated audio.
        
        Args:
            original: Original audio samples
            regenerated: Regenerated audio samples
            start_time: Start of region in seconds
            end_time: End of region in seconds
            fade_duration: Crossfade duration in seconds
            sample_rate: Sample rate in Hz
            
        Returns:
            Blended audio with loudness-matched crossfade
        """
        # Extract regions
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        
        original_region = original[start_sample:end_sample]
        
        if len(original_region) == 0 or len(regenerated) == 0:
            return original.copy()
        
        # Match loudness first
        regenerated_matched = self.match_loudness(original_region, regenerated)
        
        # Apply phase alignment
        regenerated_aligned = self.phase_align(original_region, regenerated_matched, sample_rate)
        
        # Now crossfade
        result = self.crossfade(
            original=original,
            regenerated=regenerated_aligned,
            start_time=start_time,
            end_time=end_time,
            fade_duration=fade_duration,
        )
        
        return result
    
    def blend_multiple_with_priority(
        self,
        original: np.ndarray,
        segments: List[Tuple[np.ndarray, float, float]],
        sample_rate: int = 44100,
        fade_duration: float = 0.1,
        priority: str = "last",
    ) -> np.ndarray:
        """
        Blend multiple segments with overlap handling.
        
        When segments overlap, uses priority strategy to determine
        which segment takes precedence.
        
        Args:
            original: Original audio samples
            segments: List of (audio, start_time, end_time) tuples
            sample_rate: Sample rate in Hz
            fade_duration: Crossfade duration
            priority: "first", "last", or "loudest"
            
        Returns:
            Blended audio with overlaps resolved
        """
        if not segments:
            return original.copy()
        
        result = original.copy()
        
        # Sort by start time
        sorted_segments = sorted(segments, key=lambda x: x[1])
        
        # Find and resolve overlaps
        resolved_segments = self._resolve_overlaps(sorted_segments, priority)
        
        # Blend resolved segments
        result = self.blend_regions(
            original=result,
            regenerated_segments=resolved_segments,
            sample_rate=sample_rate,
            fade_duration=fade_duration,
        )
        
        return result
    
    def _resolve_overlaps(
        self,
        segments: List[Tuple[np.ndarray, float, float]],
        priority: str = "last",
    ) -> List[Tuple[np.ndarray, float, float]]:
        """
        Resolve overlapping segments based on priority.
        
        Args:
            segments: List of (audio, start_time, end_time) tuples
            priority: "first", "last", or "loudest"
            
        Returns:
            Segments with overlaps resolved
        """
        if len(segments) <= 1:
            return segments
        
        resolved = []
        
        for i, (audio, start, end) in enumerate(segments):
            # Check if this segment overlaps with any previous
            overlap_start = start
            overlap_end = end
            
            for j in range(len(resolved)):
                prev_audio, prev_start, prev_end = resolved[j]
                
                if start < prev_end and end > prev_start:
                    # Overlap detected
                    overlap_start = max(start, prev_start)
                    overlap_end = min(end, prev_end)
                    
                    if priority == "first":
                        # Keep previous, skip this overlap
                        start = prev_end
                    elif priority == "loudest":
                        # Compare loudness
                        prev_rms = self._rms_level(prev_audio)
                        curr_rms = self._rms_level(audio)
                        
                        if curr_rms > prev_rms:
                            # Replace previous with current
                            resolved[j] = (audio, prev_start, prev_end)
                            start = prev_end
                        else:
                            # Skip current overlap
                            start = prev_end
            
            if end > start:
                resolved.append((audio, start, end))
        
        return resolved


# =============================================================================
# Convenience Functions
# =============================================================================

def create_crossfade(
    original: np.ndarray,
    regenerated: np.ndarray,
    start_time: float,
    end_time: float,
    fade_duration: float = 0.1,
) -> np.ndarray:
    """
    Quick crossfade function.
    
    Args:
        original: Original audio
        regenerated: Regenerated audio
        start_time: Start of crossfade
        end_time: End of crossfade
        fade_duration: Crossfade duration
        
    Returns:
        Crossfaded audio
    """
    blender = Blender()
    return blender.crossfade(original, regenerated, start_time, end_time, fade_duration)


def match_and_blend(
    original: np.ndarray,
    regenerated: np.ndarray,
    start_time: float,
    end_time: float,
    fade_duration: float = 0.1,
    sample_rate: int = 44100,
) -> np.ndarray:
    """
    Match loudness and blend regenerated audio.
    
    Args:
        original: Original audio
        regenerated: Regenerated audio
        start_time: Start of region
        end_time: End of region
        fade_duration: Crossfade duration
        sample_rate: Sample rate
        
    Returns:
        Blended audio with loudness matching
    """
    blender = Blender()
    return blender.blend_with_loudness_match(
        original=original,
        regenerated=regenerated,
        start_time=start_time,
        end_time=end_time,
        fade_duration=fade_duration,
        sample_rate=sample_rate,
    )
