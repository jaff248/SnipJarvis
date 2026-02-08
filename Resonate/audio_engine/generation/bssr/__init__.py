"""
Beat-Synchronous Stem-Sequential Regeneration (BSSR)

A research-grade pipeline for long-form AI audio generation that overcomes
the ~30s limit of MusicGen/JASCO models through:

1. Bar-Aligned Chunking
2. Autoregressive Continuation
3. Stem-Sequential Dependencies
4. Optimal Cut-Point Selection
5. Beat-Aligned Stitching
"""

from .musical_structure import MusicalStructureAnalyzer, MusicalStructure
from .bar_aligned_chunker import BarAlignedChunker, ChunkingConfig, BarAlignedChunk
from .optimal_cut_finder import OptimalCutFinder, CutPoint
from .beat_aligned_stitcher import BeatAlignedStitcher, StitchConfig
from .autoregressive_generator import AutoregressiveGenerator, ContinuationContext
from .stem_orchestrator import StemSequentialOrchestrator, StemDependency

__all__ = [
    'MusicalStructureAnalyzer',
    'MusicalStructure',
    'BarAlignedChunker',
    'ChunkingConfig',
    'BarAlignedChunk',
    'OptimalCutFinder',
    'CutPoint',
    'BeatAlignedStitcher',
    'StitchConfig',
    'AutoregressiveGenerator',
    'ContinuationContext',
    'StemSequentialOrchestrator',
    'StemDependency',
]
