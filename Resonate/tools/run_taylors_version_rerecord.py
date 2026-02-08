#!/usr/bin/env python3
"""
Taylor's Version Rerecording - Full AI Audio Resynthesis

This script creates a "Taylor's Version" of an input track: a full studio-quality 
rerecording using AI-generated instruments that closely follow the original performance.

Key features:
- 100% AI generation (blend=1.0)
- Enhanced conditioning (Timbre, Articulation, Melody, Rhythm)
- Iterative refinement for similarity
- Studio quality mastering
- BSSR Pipeline for long-form generation (>30s)
"""

import sys
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import soundfile as sf
from dataclasses import dataclass

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import base class
from tools.run_studio_ai_remaster_v2 import StudioAIRemasterV2, AIRemasterResult

@dataclass
class TaylorsVersionResult(AIRemasterResult):
    """Result of Taylor's Version rerecording."""
    similarity_score: float = 0.0
    iterations: int = 0
    timbre_matched: bool = False
    articulation_matched: bool = False

class TaylorsVersionRerecord(StudioAIRemasterV2):
    """
    Full AI Rerecording Engine ("Taylor's Version" Mode).
    
    Generates 100% AI audio that closely mimics the original performance.
    """
    
    def __init__(self,
                 similarity_target: float = 0.85,
                 max_iterations: int = 1,
                 match_timbre: bool = True,
                 match_articulation: bool = True,
                 blend_ratio: float = 1.0,
                 **kwargs):
        """
        Initialize Taylor's Version engine.
        
        Args:
            similarity_target: Target similarity score (0-1)
            max_iterations: Max refinement attempts
            match_timbre: Use spectral characteristics for conditioning
            match_articulation: Use expressive details for conditioning
            blend_ratio: AI/original blend (1.0 = 100% AI, 0.3 = 30% AI/70% original)
            **kwargs: Arguments passed to StudioAIRemasterV2
        """
        # Set blend ratio (default 100% AI for true "rerecording")
        kwargs['blend_ratio'] = blend_ratio
        kwargs['force_jasco'] = True
        kwargs['regenerate_minor'] = True
        
        super().__init__(**kwargs)
        
        self.similarity_target = similarity_target
        self.max_iterations = max_iterations
        self.match_timbre = match_timbre
        self.match_articulation = match_articulation
        
        logger.info("=" * 60)
        logger.info("🎙️ Taylor's Version Rerecording Engine Initialized")
        logger.info(f"  Similarity Target: {similarity_target}")
        logger.info(f"  Timbre Matching: {match_timbre}")
        logger.info(f"  Articulation Matching: {match_articulation}")
        logger.info("  Mode: 100% AI Regeneration (BSSR Enabled)")
        logger.info("=" * 60)

    def regenerate_damaged_stems(self, stems: Dict[str, np.ndarray],
                                  reports: Dict[str, Dict],
                                  sr: int,
                                  original_audio: np.ndarray) -> Tuple[Dict[str, np.ndarray], List[str]]:
        """
        Override to use enhanced extractors and full rerecording logic with BSSR.
        """
        logger.info("Step 4: AI RERECORDING (Taylor's Version)")
        
        regenerated_stems = []
        
        try:
            from audio_engine.generation.stem_regenerator import StemRegenerator, RegenerationRegion
            from audio_engine.profiling.quality_detector import DamageLevel, StemQualityReport
            from audio_engine.profiling.melody_extractor import MelodyExtractor
            from audio_engine.profiling.drum_pattern_extractor import DrumPatternExtractor
            from audio_engine.profiling.timbre_analyzer import TimbreAnalyzer
            from audio_engine.profiling.articulation_detector import ArticulationDetector
            from audio_engine.generation.jasco_generator import MelodyConditioning, DrumConditioning
            
            # Initialize extractors
            melody_extractor = MelodyExtractor()
            drum_extractor = DrumPatternExtractor()
            timbre_analyzer = TimbreAnalyzer() if self.match_timbre else None
            articulation_detector = ArticulationDetector() if self.match_articulation else None
            
            # Initialize Regenerator with BSSR enabled
            regenerator = StemRegenerator(use_bssr=True)
            
            # Configure internal generator
            regenerator.generator.config.model_variant = self.jasco_model
            regenerator.generator.config.guidance_scale = 8.0
            
            import torch
            if torch.backends.mps.is_available() and not self.force_cpu:
                regenerator.generator.config.device = "mps"
            else:
                regenerator.generator.config.device = "cpu"
            
            # Process each stem
            for stem_name, original_stem in stems.items():
                logger.info(f"  🎙️ Rerecording {stem_name}...")
                
                # Build Musical Profile
                musical_profile = {
                    "tempo": 120.0, # Default, BSSR analyzer will refine this from audio
                    "key": "C",
                    "chords": [] # BSSR might not use this yet if not provided
                }
                
                # 1. Analyze Timbre
                if self.match_timbre:
                    timbre_profile = timbre_analyzer.analyze(original_stem, sr, stem_name)
                    musical_profile['timbre'] = timbre_profile
                    logger.info(f"    Timbre: {timbre_profile.texture_description}")
                    
                # 2. Analyze Articulation
                if self.match_articulation:
                    articulation_profile = articulation_detector.detect(original_stem, sr, stem_name)
                    musical_profile['articulation'] = articulation_profile
                    logger.info(f"    Articulation: {articulation_profile.articulation_description}")
                
                # 3. Analyze Melody/Drums
                try:
                    if stem_name == 'drums':
                        onsets = drum_extractor.extract(original_stem, sr)
                        # Create DrumConditioning object
                        musical_profile['drums'] = DrumConditioning(
                            onset_times=onsets.timestamps,
                            tempo_bpm=120.0 # Placeholder
                        )
                    else:
                        contour = melody_extractor.extract(original_stem, sr)
                        # Create MelodyConditioning object
                        musical_profile['melody'] = MelodyConditioning(
                            contour=contour.salience
                        )
                except Exception as e:
                    logger.warning(f"    Conditioning extraction failed: {e}")
                
                # Create a full-length regeneration region
                # This triggers "should_regenerate_entire_stem" logic in StemRegenerator
                # which then routes to BSSR if duration > 30s
                duration = len(original_stem) / sr
                
                dummy_report = StemQualityReport(
                    damage_level=DamageLevel.CRITICAL, 
                    confidence=1.0, 
                    needs_regeneration=True,
                    regenerate_regions=[(0.0, duration)]
                )
                
                region = RegenerationRegion(
                    start_time=0.0, 
                    end_time=duration, 
                    stem_name=stem_name,
                    damage_level=DamageLevel.CRITICAL,
                    quality_report=dummy_report
                )
                
                # Execute Regeneration
                try:
                    generated = regenerator.regenerate_regions(
                        audio=original_stem,
                        regions=[region],
                        musical_profile=musical_profile,
                        stem_type=stem_name if stem_name != "vocals" else "vocals",
                        sample_rate=sr
                    )
                    
                    # Update stem
                    stems[stem_name] = generated.astype(np.float32)
                    regenerated_stems.append(stem_name)
                    logger.info(f"    ✅ Rerecorded {stem_name} ({len(generated)/sr:.1f}s)")
                    
                except Exception as e:
                    logger.error(f"    ❌ Rerecording failed for {stem_name}: {e}")
                    # Keep original on failure
            
            return stems, regenerated_stems
            
        except Exception as e:
            logger.error(f"  Rerecording setup failed: {e}")
            import traceback
            traceback.print_exc()
            return stems, []

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Taylor's Version Rerecording - 100% AI Audio Resynthesis"
    )
    parser.add_argument("input_file", help="Input audio file")
    parser.add_argument("--output", "-o", help="Output file path")
    parser.add_argument("--genre", default="studio", help="Target genre")
    parser.add_argument("--blend", type=float, default=1.0,
                       help="AI blend ratio (1.0=100%% AI, 0.3=30%% AI/70%% original, default: 1.0)")
    parser.add_argument("--similarity-target", type=float, default=0.85, help="Target similarity (0.85)")
    parser.add_argument("--no-timbre", action="store_true", help="Disable timbre matching")
    parser.add_argument("--no-articulation", action="store_true", help="Disable articulation matching")
    parser.add_argument("--vocal-boost", type=float, default=1.0, help="Vocal boost")
    
    # Standard args needed for base class
    parser.add_argument("--demucs-model", default="htdemucs_ft")
    parser.add_argument("--jasco-model", default="medium")
    parser.add_argument("--force-cpu", action="store_true")
    
    args = parser.parse_args()
    
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Not found: {input_path}")
        sys.exit(1)
        
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_name(f"{input_path.stem}_taylors_version.wav")
        
    engine = TaylorsVersionRerecord(
        similarity_target=args.similarity_target,
        match_timbre=not args.no_timbre,
        match_articulation=not args.no_articulation,
        blend_ratio=args.blend,
        demucs_model=args.demucs_model,
        jasco_model=args.jasco_model,
        force_cpu=args.force_cpu,
        genre=args.genre,
        vocal_boost=args.vocal_boost,
        precondition_strength=0.5 # Default stronger cleaning for rerecording
    )
    
    engine.process(str(input_path), str(output_path))

if __name__ == "__main__":
    main()
