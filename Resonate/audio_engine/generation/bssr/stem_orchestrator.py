"""
Stem Sequential Orchestrator - Coordinate the BSSR pipeline.

This module provides the StemSequentialOrchestrator, which manages the entire
Beat-Synchronous Stem-Sequential Regeneration process, including:
1. Orchestrating stem generation order (Drums -> Bass -> Other -> Vocals)
2. Managing chunks and overlaps
3. Stitching results into full-length audio
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from .autoregressive_generator import AutoregressiveGenerator, ContinuationContext
from .bar_aligned_chunker import BarAlignedChunker
from .beat_aligned_stitcher import BeatAlignedStitcher
from .musical_structure import MusicalStructure
from .optimal_cut_finder import OptimalCutFinder

logger = logging.getLogger(__name__)


@dataclass
class StemDependency:
    """Defines a stem's generation priority and dependencies."""
    stem_type: str
    depends_on: List[str]
    conditioning_weight: float


# Default generation order
STEM_ORDER = [
    StemDependency('drums', [], 1.0),           # First - sets rhythm
    StemDependency('bass', ['drums'], 0.8),     # Follows drums + chords
    StemDependency('other', ['drums', 'bass'], 0.6),  # Follows all
    StemDependency('vocals', ['drums', 'bass', 'other'], 0.4),  # Last
]


class StemSequentialOrchestrator:
    """
    Orchestrates the full BSSR pipeline for long-form audio generation.
    """
    
    def __init__(self,
                 chunker: BarAlignedChunker,
                 generator: AutoregressiveGenerator,
                 stitcher: Optional[BeatAlignedStitcher] = None,
                 cut_finder: Optional[OptimalCutFinder] = None):
        """
        Initialize the orchestrator.
        
        Args:
            chunker: Configured BarAlignedChunker
            generator: Configured AutoregressiveGenerator
            stitcher: Optional BeatAlignedStitcher
            cut_finder: Optional OptimalCutFinder
        """
        self.chunker = chunker
        self.generator = generator
        self.stitcher = stitcher or BeatAlignedStitcher()
        self.cut_finder = cut_finder or OptimalCutFinder()
        self.stem_order = STEM_ORDER
        
    def regenerate_all_stems(self,
                             original_stems: Dict[str, np.ndarray],
                             musical_profile: Dict[str, Any],
                             structure: MusicalStructure,
                             callbacks: Optional[Any] = None) -> Dict[str, np.ndarray]:
        """
        Regenerate all stems in dependency order using BSSR.
        
        Args:
            original_stems: Dictionary of original audio stems
            musical_profile: Extracted musical profile
            structure: Extracted musical structure
            callbacks: Progress callbacks
            
        Returns:
            Dictionary of regenerated stems
        """
        logger.info("Starting BSSR regeneration pipeline")
        
        # 1. Create chunks
        chunks = self.chunker.create_chunks(structure)
        logger.info(f"Split audio into {len(chunks)} bar-aligned chunks")
        
        regenerated_stems = {}
        
        # 2. Process stems in order
        total_stems = len([s for s in self.stem_order if s.stem_type in original_stems])
        completed_stems = 0
        
        for stem_dep in self.stem_order:
            stem_type = stem_dep.stem_type
            
            # Skip if not in original stems (unless we want to generate new stems?)
            # For "Taylor's Version", we typically replace existing stems.
            if stem_type not in original_stems and stem_type != 'all':
                continue
            
            logger.info(f"Processing stem: {stem_type} (Step {completed_stems+1}/{total_stems})")
            if callbacks:
                callbacks.report_progress(
                    int(completed_stems / total_stems * 100),
                    f"Regenerating {stem_type}..."
                )
            
            # TODO: Get conditioning from already-generated stems
            # This would involve mixing the generated stems to create a "backing track"
            # context for the current stem. For now, we rely on the symbolic profile (chords/key/tempo).
            
            # Regenerate this stem
            regenerated = self._regenerate_stem(
                chunks=chunks,
                stem_type=stem_type,
                musical_profile=musical_profile,
                structure=structure,
                callbacks=callbacks
            )
            
            regenerated_stems[stem_type] = regenerated
            completed_stems += 1
            
        logger.info("BSSR pipeline complete")
        return regenerated_stems
        
    def _regenerate_stem(self,
                         chunks: List[Any],
                         stem_type: str,
                         musical_profile: Dict[str, Any],
                         structure: MusicalStructure,
                         callbacks: Optional[Any] = None) -> np.ndarray:
        """
        Regenerate a single stem across all chunks and stitch them.
        """
        generated_chunks = []
        context = None
        
        for i, chunk in enumerate(chunks):
            logger.debug(f"Generating chunk {i+1}/{len(chunks)}")
            
            # Generate chunk
            result = self.generator.generate_chunk(
                chunk=chunk,
                context=context,
                musical_profile=musical_profile,
                stem_type=stem_type
            )
            
            if not result.success or result.audio is None:
                logger.error(f"Failed to generate chunk {i}: {result.error}")
                # Fallback: Silence or handle error
                # For now, return empty array to signal failure upstream
                return np.array([])
                
            generated_chunks.append(result.audio)
            
            # Update context for next chunk
            # Use the last N seconds of THIS chunk as context for the NEXT
            # Note: We need to be careful with the overlap.
            # The chunk we just generated includes the overlap region at the end.
            # We want the next chunk to continue smoothly from that overlap.
            
            # For context, we just grab the very end of the generated audio.
            # MusicGen will try to match this.
            context_audio = result.audio
            
            context = ContinuationContext(
                previous_audio=context_audio,
                sample_rate=result.sample_rate,
                duration=2.0, # Use last 2 seconds as prompt
                tempo=structure.tempo,
                key=musical_profile.get('key'),
                chords=musical_profile.get('chords')
            )
            
        # Stitch chunks
        return self._stitch_stem(generated_chunks, chunks, structure)
        
    def _stitch_stem(self,
                     audio_chunks: List[np.ndarray],
                     chunk_defs: List[Any],
                     structure: MusicalStructure) -> np.ndarray:
        """
        Stitch generated audio chunks together using optimal cuts.
        """
        if len(audio_chunks) == 1:
            return audio_chunks[0]
            
        # Find optimal cut points for each overlap
        cut_points = []
        
        for i in range(len(audio_chunks) - 1):
            outgoing = audio_chunks[i]
            incoming = audio_chunks[i+1]
            chunk_def = chunk_defs[i] # Definition for outgoing chunk
            
            # Overlap duration is defined in the chunk definition
            # chunk_def.overlap_end is the duration of overlap into the next chunk
            overlap_duration = chunk_def.overlap_end
            
            # Get beat times in the overlap region to snap cuts
            # Outgoing chunk ends at chunk_def.end_time
            # Overlap starts at chunk_def.end_time - overlap_duration
            overlap_start_time = chunk_def.end_time - overlap_duration
            
            # Find beats in this window
            beat_indices = np.where(
                (structure.beat_times >= overlap_start_time) & 
                (structure.beat_times <= chunk_def.end_time)
            )[0]
            
            if len(beat_indices) > 0:
                # Get times relative to start of overlap
                beats_in_overlap = structure.beat_times[beat_indices] - overlap_start_time
            else:
                beats_in_overlap = np.array([])
            
            # Find optimal cut
            cut_point = self.cut_finder.find_optimal_cut(
                chunk_a=outgoing,
                chunk_b=incoming,
                overlap_duration=overlap_duration,
                beat_times=beats_in_overlap
            )
            cut_points.append(cut_point)
            
        # Stitch
        # Note: The Stitcher's stitch_chunks API is a bit high level.
        # We might need to iterate and stitch pairs if we want precise control.
        # But let's use the provided stitcher method if possible, or iterate here.
        
        # Iterative stitching:
        result = audio_chunks[0]
        
        for i in range(len(cut_points)):
            incoming = audio_chunks[i+1]
            cut_point = cut_points[i]
            chunk_def = chunk_defs[i]
            overlap_duration = chunk_def.overlap_end
            
            # Use stitcher to blend
            # We treat 'result' as the outgoing chunk (accumulated)
            result = self.stitcher.crossfade_pair(
                outgoing_chunk=result,
                incoming_chunk=incoming,
                cut_point=cut_point,
                overlap_duration=overlap_duration,
                structure=structure
            )
            
        return result
