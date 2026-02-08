#!/usr/bin/env python3
"""
Test BSSR Pipeline - Verify long-form audio generation.

This script tests the Beat-Synchronous Stem-Sequential Regeneration (BSSR) pipeline
by generating a synthetic long audio file and running it through the Orchestrator
with a mock generator.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

import numpy as np
import soundfile as sf

# Add Resonate directory to path
sys.path.insert(0, str(Path(__file__).parent / "Resonate"))

# Mock missing dependencies
import sys
from unittest.mock import MagicMock

# Mock modules that might be missing
for mod in ['demucs', 'demucs.separate', 'audiocraft', 'audiocraft.models', 'audiocraft.data.audio']:
    sys.modules[mod] = MagicMock()

# Now import BSSR components
from audio_engine.generation.bssr import (
    MusicalStructureAnalyzer, 
    BarAlignedChunker, 
    StemSequentialOrchestrator,
    AutoregressiveGenerator,
    ContinuationContext
)
from audio_engine.generation.jasco_generator import GenerationResult, JASCOGenerator, GenerationConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("test_bssr")

# Mock Generator that produces synthetic audio
class MockJASCOGenerator:
    def __init__(self):
        self.config = GenerationConfig(sample_rate=44100)
        self.is_loaded = True
        self._model = MagicMock() # Mock internal model
        
    def load_model(self):
        return True
        
    def _build_description(self, **kwargs):
        return ["mock description"]
        
    def generate_from_profile(self, profile, stem_type, duration, callbacks=None, **kwargs):
        logger.info(f"   [MockGen] Generating {duration}s of {stem_type}...")
        
        # Generate sine wave at different frequency per stem
        freqs = {"drums": 100, "bass": 200, "other": 400, "vocals": 800}
        freq = freqs.get(stem_type, 440)
        
        sr = 44100
        t = np.linspace(0, duration, int(duration * sr))
        audio = np.sin(2 * np.pi * freq * t) * 0.5
        
        # Add fade in/out
        fade_len = int(0.1 * sr)
        if len(audio) > 2 * fade_len:
            audio[:fade_len] *= np.linspace(0, 1, fade_len)
            audio[-fade_len:] *= np.linspace(1, 0, fade_len)
            
        return GenerationResult(
            audio=audio,
            sample_rate=sr,
            duration=duration,
            success=True
        )

# Subclass AutoregressiveGenerator to use Mock
class MockAutoregressiveGenerator(AutoregressiveGenerator):
    def __init__(self):
        self.generator = MockJASCOGenerator()
        
    def _generate_with_continuation(self, chunk, context, profile, stem_type, callbacks=None):
        logger.info(f"   [MockAuto] Continuing chunk {chunk.chunk_index} ({chunk.duration:.1f}s)")
        # Just call fresh generation for mock, but log it
        return self._generate_fresh(chunk, profile, stem_type, callbacks)

def generate_synthetic_audio(duration_s: float, sr: int = 44100, bpm: float = 120.0) -> np.ndarray:
    """Generate synthetic audio with clear beat structure."""
    logger.info(f"Generating {duration_s}s synthetic audio at {bpm} BPM...")
    
    t = np.linspace(0, duration_s, int(duration_s * sr))
    
    # 1. Metronome click on beats
    beat_interval = 60.0 / bpm
    clicks = np.zeros_like(t)
    
    beat_indices = np.arange(0, len(t), int(beat_interval * sr))
    # Make clicks 50ms long
    click_len = int(0.05 * sr)
    
    for idx in beat_indices:
        if idx + click_len < len(clicks):
            # Higher pitch for downbeat (every 4th)
            freq = 880.0 if (idx // int(beat_interval * sr)) % 4 == 0 else 440.0
            click_t = np.linspace(0, 0.05, click_len)
            clicks[idx:idx+click_len] = np.sin(2 * np.pi * freq * click_t) * 0.8
            
    return clicks

def test_bssr_logic(audio: np.ndarray, sr: int, stem_type: str = "drums"):
    """Test BSSR logic flow without heavy models."""
    logger.info(f"\n--- Testing BSSR Logic (Stem: {stem_type}) ---")
    
    # 1. Analyze
    logger.info("1. Structure Analysis...")
    analyzer = MusicalStructureAnalyzer()
    structure = analyzer.analyze(audio, sr)
    logger.info(f"   Detected {structure.total_bars} bars at {structure.tempo:.1f} BPM")
    
    # 2. Chunk
    logger.info("2. Chunking...")
    chunker = BarAlignedChunker()
    chunks = chunker.create_chunks(structure)
    logger.info(f"   Created {len(chunks)} chunks")
    
    # 3. Orchestrate with Mock
    logger.info("3. Orchestrating...")
    mock_gen = MockAutoregressiveGenerator()
    orchestrator = StemSequentialOrchestrator(
        chunker=chunker,
        generator=mock_gen
    )
    
    # Create dummy profile
    profile = {"tempo": structure.tempo, "key": "C", "chords": []}
    
    # Run
    start_time = time.time()
    results = orchestrator.regenerate_all_stems(
        original_stems={stem_type: audio},
        musical_profile=profile,
        structure=structure
    )
    
    elapsed = time.time() - start_time
    logger.info(f"Orchestration complete in {elapsed:.2f}s")
    
    # Verify
    result_audio = results.get(stem_type)
    if result_audio is not None:
        logger.info(f"Output shape: {result_audio.shape}")
        logger.info(f"Output duration: {len(result_audio)/sr:.2f}s")
        logger.info(f"Target duration: {len(audio)/sr:.2f}s")
        
        diff = abs(len(result_audio) - len(audio)) / sr
        if diff < 1.0:
            logger.info("✅ Duration matches target")
        else:
            logger.warning(f"⚠️ Duration mismatch: {diff:.2f}s difference")
            
        # Save
        out_path = f"bssr_mock_output_{stem_type}.wav"
        sf.write(out_path, result_audio, sr)
        logger.info(f"Saved mock output to {out_path}")
    else:
        logger.error("❌ No output generated")

def main():
    parser = argparse.ArgumentParser(description="Test BSSR Pipeline Logic")
    parser.add_argument("--duration", type=float, default=65.0, help="Duration of synthetic audio")
    
    args = parser.parse_args()
    
    sr = 44100
    audio = generate_synthetic_audio(args.duration, sr=sr)
    
    try:
        test_bssr_logic(audio, sr)
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
