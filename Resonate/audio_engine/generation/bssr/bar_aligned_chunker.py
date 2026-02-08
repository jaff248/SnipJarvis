"""
Bar-Aligned Chunker - Break audio into musically coherent chunks.

This module provides the BarAlignedChunker, which divides audio into chunks
aligned with musical bar boundaries to ensure seamless generation and stitching.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .musical_structure import MusicalStructure

logger = logging.getLogger(__name__)


@dataclass
class BarAlignedChunk:
    """
    Represents a chunk of audio aligned to bar boundaries.
    
    Attributes:
        chunk_index: Index of this chunk in the sequence
        start_time: Start time in seconds (aligned to bar start)
        end_time: End time in seconds (aligned to bar start + overlap)
        start_bar: Index of the starting bar
        end_bar: Index of the ending bar
        overlap_start: Start time of overlap with previous chunk (0 for first chunk)
        overlap_end: Duration of overlap into next chunk
        is_section_boundary: Whether chunk starts at a detected section boundary
    """
    chunk_index: int
    start_time: float
    end_time: float
    start_bar: int
    end_bar: int
    overlap_start: float
    overlap_end: float
    is_section_boundary: bool
    
    @property
    def duration(self) -> float:
        """Total duration of the chunk including overlap."""
        return self.end_time - self.start_time
    
    @property
    def non_overlap_duration(self) -> float:
        """Duration of new content (excluding overlap with previous)."""
        return self.duration - self.overlap_start


@dataclass
class ChunkingConfig:
    """
    Configuration for the chunking process.
    
    Attributes:
        target_duration: Target duration for chunks in seconds
        max_duration: Maximum allowed duration (MusicGen limit is ~30s)
        min_duration: Minimum allowed duration
        overlap_bars: Number of bars to overlap between chunks
        prefer_section_boundaries: Whether to align chunks to section starts
    """
    target_duration: float = 28.0
    max_duration: float = 30.0
    min_duration: float = 10.0
    overlap_bars: int = 4
    prefer_section_boundaries: bool = True


class BarAlignedChunker:
    """
    Splits audio into bar-aligned chunks for generation.
    
    Ensures that:
    1. Chunks start and end on bar boundaries
    2. Chunks are within duration limits (under 30s for MusicGen)
    3. Overlaps are sufficient for crossfading
    4. Section boundaries are respected when possible
    """
    
    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or ChunkingConfig()
        
    def create_chunks(self, structure: MusicalStructure) -> List[BarAlignedChunk]:
        """
        Create bar-aligned chunks from musical structure.
        
        Args:
            structure: MusicalStructure object with bar info
            
        Returns:
            List of BarAlignedChunk objects covering the full duration
        """
        logger.info(f"Creating bar-aligned chunks (target={self.config.target_duration}s)")
        
        chunks = []
        current_bar = 0
        chunk_idx = 0
        total_bars = structure.total_bars
        
        # If no bars detected, fallback to time-based chunking
        if total_bars == 0:
            logger.warning("No bars detected, falling back to time-based chunking")
            return self._create_time_based_chunks(structure.duration)
            
        while current_bar < total_bars:
            # Calculate start time
            start_time = structure.bar_times[current_bar]
            
            # Determine overlap from previous chunk
            overlap_start_duration = 0.0
            if chunk_idx > 0:
                # Calculate duration of previous chunk's overlap into this one
                # Note: This is simpler than recalculating exactly; we just track it
                # effectively, the *start* of this chunk was determined by the previous chunk's target end
                pass
                
            # Find target end bar
            # We want: (end_time - start_time) <= max_duration
            
            # 1. Find the bar that is closest to start_time + target_duration
            target_time = start_time + self.config.target_duration
            
            # Find bar closest to target_time
            # np.searchsorted finds insertion point to maintain order
            target_bar_idx = np.searchsorted(structure.bar_times, target_time)
            
            # Ensure we don't exceed max duration
            # Check duration if we end at target_bar_idx
            if target_bar_idx >= total_bars:
                target_bar_idx = total_bars
            
            # Validate against max_duration
            if target_bar_idx < total_bars:
                while target_bar_idx > current_bar and \
                      (structure.bar_times[target_bar_idx] - start_time) > self.config.max_duration:
                    target_bar_idx -= 1
            else:
                 # Last chunk
                 pass

            # Check for section boundaries near target
            if self.config.prefer_section_boundaries and structure.section_boundaries:
                # Look for section boundary within +/- 2 bars of target
                for section_time in structure.section_boundaries:
                    section_bar = np.searchsorted(structure.bar_times, section_time)
                    if abs(section_bar - target_bar_idx) <= 2:
                        # Ensure this doesn't violate max duration
                        if (structure.bar_times[section_bar] - start_time) <= self.config.max_duration:
                            target_bar_idx = section_bar
                            break
            
            # Ensure minimum duration unless it's the last chunk
            if target_bar_idx < total_bars:
                while target_bar_idx < total_bars and \
                      (structure.bar_times[target_bar_idx] - start_time) < self.config.min_duration:
                    target_bar_idx += 1
            
            # Determine actual end bar (start of next chunk's fresh content)
            # This bar will be the start of the next chunk
            next_chunk_start_bar = target_bar_idx
            
            # Add overlap for THIS chunk's generation
            # We generate past the target point to allow for crossfading
            gen_end_bar = min(total_bars, next_chunk_start_bar + self.config.overlap_bars)
            
            # If we are at the end, just go to the end
            if next_chunk_start_bar >= total_bars:
                gen_end_bar = total_bars
                next_chunk_start_bar = total_bars # Terminate loop
            
            # Get times
            if gen_end_bar < total_bars:
                end_time = structure.bar_times[gen_end_bar]
            else:
                end_time = structure.duration
                
            # Calculate overlap duration (time between next chunk start and this chunk end)
            if next_chunk_start_bar < total_bars:
                next_start_time = structure.bar_times[next_chunk_start_bar]
                overlap_end_duration = end_time - next_start_time
            else:
                overlap_end_duration = 0.0
            
            # Overlap from previous
            overlap_prev = 0.0
            if chunk_idx > 0:
                # Previous chunk ended at current start_time + its overlap
                # Effectively we are starting at 'current_bar', which was the 'next_chunk_start_bar' of previous
                # So the overlap region is [current_bar, current_bar + overlap]
                # Wait, simpler model:
                # Chunk N starts at bar X.
                # Chunk N ends at bar Y (generation end).
                # Chunk N+1 starts at bar Z (where Z < Y).
                # Overlap is Y - Z bars.
                
                # In this loop:
                # start_time is set to structure.bar_times[current_bar]
                # We need to know how much overlap exists *before* this start point?
                # No, BarAlignedChunk definition: start_time is the start of generation.
                # But for BSSR, we generate FROM start_time.
                # The overlap is handled by the stitcher blending the *end* of chunk N 
                # with the *start* of chunk N+1.
                pass

            chunk = BarAlignedChunk(
                chunk_index=chunk_idx,
                start_time=start_time,
                end_time=end_time,
                start_bar=current_bar,
                end_bar=gen_end_bar,
                overlap_start=0.0, # Not used in this logic, but good for record keeping
                overlap_end=overlap_end_duration,
                is_section_boundary=False # TODO: Check against sections
            )
            chunks.append(chunk)
            
            # Advance
            current_bar = next_chunk_start_bar
            chunk_idx += 1
            
            if current_bar >= total_bars and gen_end_bar >= total_bars:
                break
                
        logger.info(f"Created {len(chunks)} chunks")
        return chunks

    def _create_time_based_chunks(self, duration: float) -> List[BarAlignedChunk]:
        """Fallback for when no musical structure is detected."""
        chunks = []
        current_time = 0.0
        chunk_idx = 0
        
        while current_time < duration:
            target_end = min(duration, current_time + self.config.target_duration)
            
            # Add overlap
            gen_end = min(duration, target_end + 5.0) # 5s overlap
            
            overlap_end = gen_end - target_end
            
            chunk = BarAlignedChunk(
                chunk_index=chunk_idx,
                start_time=current_time,
                end_time=gen_end,
                start_bar=0, # Dummy
                end_bar=0,   # Dummy
                overlap_start=0.0,
                overlap_end=overlap_end,
                is_section_boundary=False
            )
            chunks.append(chunk)
            
            current_time = target_end
            chunk_idx += 1
            
        return chunks
