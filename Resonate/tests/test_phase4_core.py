"""
Phase 4 Core Functionality Test Script

Tests the JASCO stem regeneration feature modules for Resonate.
Verifies imports, basic functionality, and integration.

Run with: python tests/test_phase4_core.py
"""

import sys
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all Phase 4 modules can be imported."""
    logger.info("=" * 60)
    logger.info("Testing module imports...")
    logger.info("=" * 60)
    
    imports = [
        # Profiling module
        ("audio_engine.profiling.quality_detector", ["QualityDetector", "DamageLevel", "StemQualityReport"]),
        ("audio_engine.profiling.chord_extractor", ["ChordExtractor", "ChordProgressions"]),
        ("audio_engine.profiling.tempo_key_analyzer", ["TempoKeyAnalyzer", "TempoKeyInfo"]),
        ("audio_engine.profiling.melody_extractor", ["MelodyExtractor", "MelodyContour"]),
        ("audio_engine.profiling.drum_pattern_extractor", ["DrumPatternExtractor", "DrumOnsets"]),
        ("audio_engine.profiling", ["extract_all_profiles"]),
        
        # Generation module
        ("audio_engine.generation.jasco_generator", ["JASCOGenerator", "GenerationConfig", "GenerationResult"]),
        ("audio_engine.generation.stem_regenerator", ["StemRegenerator", "RegenerationRegion"]),
        ("audio_engine.generation.blender", ["Blender", "create_crossfade"]),
        ("audio_engine.generation", ["create_generator", "create_regenerator"]),
        
        # Polish module
        ("audio_engine.polish.mbd_enhancer", ["MBDEnhancer"]),
    ]
    
    passed = 0
    failed = 0
    
    for module_name, expected_names in imports:
        try:
            module = __import__(module_name, fromlist=expected_names)
            
            # Check expected names exist
            missing = []
            for name in expected_names:
                if not hasattr(module, name):
                    missing.append(name)
            
            if missing:
                logger.error(f"  ❌ {module_name}: missing {missing}")
                failed += 1
            else:
                logger.info(f"  ✅ {module_name}: all imports OK")
                passed += 1
                
        except ImportError as e:
            logger.error(f"  ❌ {module_name}: {e}")
            failed += 1
    
    logger.info(f"\nImport results: {passed} passed, {failed} failed")
    return failed == 0


def test_quality_detection():
    """Test quality detection on synthetic audio."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing quality detection...")
    logger.info("=" * 60)
    
    try:
        from audio_engine.profiling.quality_detector import QualityDetector
        
        # Create synthetic test audio (2 seconds of clean sine wave)
        sample_rate = 44100
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
        clean_audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        
        # Add some noise for testing
        noise = np.random.randn(len(t)).astype(np.float32) * 0.01
        test_audio = clean_audio + noise
        
        detector = QualityDetector()
        report = detector.analyze(test_audio, sample_rate)
        
        logger.info(f"  Damage level: {report.damage_level.value}")
        logger.info(f"  Confidence: {report.confidence:.2f}")
        logger.info(f"  Needs regeneration: {report.needs_regeneration}")
        logger.info(f"  Regenerate regions: {len(report.regenerate_regions)}")
        
        # Test with heavily clipped audio
        clipped_audio = np.clip(test_audio * 2.0, -1.0, 1.0)
        report_clipped = detector.analyze(clipped_audio, sample_rate)
        logger.info(f"  Clipped audio damage level: {report_clipped.damage_level.value}")
        
        logger.info("  ✅ Quality detection test passed")
        return True
        
    except Exception as e:
        logger.error(f"  ❌ Quality detection test failed: {e}")
        return False


def test_chord_extraction():
    """Test chord extraction on synthetic audio."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing chord extraction...")
    logger.info("=" * 60)
    
    try:
        from audio_engine.profiling.chord_extractor import ChordExtractor
        
        # Create synthetic test audio (simple chord progression)
        sample_rate = 44100
        duration = 4.0
        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
        
        # Create a simple audio with harmonic content
        # C major chord (C, E, G frequencies)
        c4, e4, g4 = 261.63, 329.63, 392.00
        audio = (np.sin(2 * np.pi * c4 * t) + 
                 np.sin(2 * np.pi * e4 * t) + 
                 np.sin(2 * np.pi * g4 * t)) * 0.3
        
        extractor = ChordExtractor()
        chords = extractor.extract(audio, sample_rate)
        
        logger.info(f"  Key detected: {chords.key}")
        logger.info(f"  Tempo detected: {chords.tempo:.1f} BPM")
        logger.info(f"  Chord timeline: {chords.get_chord_timeline()[:3]}...")
        
        logger.info("  ✅ Chord extraction test passed")
        return True
        
    except Exception as e:
        logger.error(f"  ❌ Chord extraction test failed: {e}")
        return False


def test_tempo_key_analysis():
    """Test tempo and key analysis."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing tempo/key analysis...")
    logger.info("=" * 60)
    
    try:
        from audio_engine.profiling.tempo_key_analyzer import TempoKeyAnalyzer
        
        # Create synthetic test audio with tempo (beat-like pulses)
        sample_rate = 44100
        duration = 4.0
        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
        
        # Create beat-like pulses at 120 BPM
        bpm = 120
        beat_interval = 60 / bpm
        beats = np.zeros_like(t)
        for i in range(int(duration / beat_interval)):
            beat_time = i * beat_interval
            beat_sample = int(beat_time * sample_rate)
            if beat_sample < len(beats):
                beats[beat_sample:beat_sample + 100] = 1.0
        
        audio = beats * 0.5
        
        analyzer = TempoKeyAnalyzer()
        info = analyzer.analyze(audio, sample_rate)
        
        logger.info(f"  Tempo detected: {info.tempo:.1f} BPM")
        logger.info(f"  Key detected: {info.key}")
        logger.info(f"  Time signature: {info.time_signature}")
        logger.info(f"  Confidence: {info.confidence:.2f}")
        
        logger.info("  ✅ Tempo/key analysis test passed")
        return True
        
    except Exception as e:
        logger.error(f"  ❌ Tempo/key analysis test failed: {e}")
        return False


def test_jasco_generator_init():
    """Test JASCO generator initialization."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing JASCO generator initialization...")
    logger.info("=" * 60)
    
    try:
        from audio_engine.generation.jasco_generator import (
            JASCOGenerator, 
            GenerationConfig,
            ModelLoadError
        )
        
        # Test configuration creation
        config = GenerationConfig(
            device="cpu",  # Use CPU for testing
            stem_type="drums",
            duration=5,
            guidance_scale=7.0,
            num_steps=20
        )
        
        logger.info(f"  Config created: device={config.device}, stem={config.stem_type}")
        
        # Test generator creation
        generator = JASCOGenerator(config)
        logger.info(f"  Generator created: {generator}")
        
        # Test model info (without loading model)
        info = generator.get_model_info()
        logger.info(f"  Model info (unloaded): {info}")
        
        logger.info("  ✅ JASCO generator initialization test passed")
        return True
        
    except Exception as e:
        logger.error(f"  ❌ JASCO generator initialization test failed: {e}")
        return False


def test_stem_regenerator():
    """Test stem regenerator region detection."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing stem regenerator...")
    logger.info("=" * 60)
    
    try:
        from audio_engine.generation.stem_regenerator import (
            StemRegenerator,
            RegenerationRegion
        )
        from audio_engine.generation.jasco_generator import GenerationConfig
        
        # Create test audio with damaged regions
        sample_rate = 44100
        duration = 5.0
        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        
        # Create regenerator
        config = GenerationConfig(device="cpu", duration=5)
        regenerator = StemRegenerator(config)
        
        # Test region identification
        regions = regenerator.identify_regions(audio, "test_stem", sample_rate)
        logger.info(f"  Identified {len(regions)} regions for clean audio")
        
        # Create a damaged region manually
        damaged_region = RegenerationRegion(
            start_time=1.0,
            end_time=2.0,
            stem_name="test_stem",
            damage_level=None,  # Will be set internally
        )
        logger.info(f"  Created test region: {damaged_region.duration:.2f}s")
        
        # Test regeneration summary
        summary = regenerator.get_regeneration_summary(
            [damaged_region], audio, sample_rate
        )
        logger.info(f"  Summary: {summary.regions_regenerated} regions, {summary.percent_regenerated:.1f}% regenerated")
        
        logger.info("  ✅ Stem regenerator test passed")
        return True
        
    except Exception as e:
        logger.error(f"  ❌ Stem regenerator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_blender():
    """Test blender crossfade functionality."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing blender crossfade...")
    logger.info("=" * 60)
    
    try:
        from audio_engine.generation.blender import Blender
        
        # Create test audio segments
        sample_rate = 44100
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
        
        original = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz
        regenerated = 0.5 * np.sin(2 * np.pi * 880 * t)  # 880 Hz
        
        blender = Blender()
        
        # Test crossfade
        result = blender.crossfade(
            original, regenerated, 
            start_time=0.5, 
            end_time=1.5, 
            fade_duration=0.1
        )
        
        logger.info(f"  Original shape: {original.shape}")
        logger.info(f"  Regenerated shape: {regenerated.shape}")
        logger.info(f"  Result shape: {result.shape}")
        
        # Test blend_regions
        segments = [(regenerated, 0.0, 1.0)]
        blended = blender.blend_regions(original, segments, sample_rate)
        logger.info(f"  Blended result shape: {blended.shape}")
        
        # Test loudness matching
        loudness_matched = blender.match_loudness(original, regenerated)
        logger.info(f"  Loudness matched shape: {loudness_matched.shape}")
        
        logger.info("  ✅ Blender test passed")
        return True
        
    except Exception as e:
        logger.error(f"  ❌ Blender test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mbd_enhancer():
    """Test MBD enhancer initialization (without model loading)."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing MBD enhancer...")
    logger.info("=" * 60)
    
    try:
        from audio_engine.polish.mbd_enhancer import MBDEnhancer
        
        # Test enhancer creation (will try to load model, but handle gracefully)
        enhancer = MBDEnhancer(sample_rate=24000)
        
        logger.info(f"  MBD available: {enhancer.available}")
        
        # Create test audio
        sample_rate = 24000
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        
        # Test processing (will return original if MBD unavailable)
        result = enhancer.process(audio, intensity=0.5)
        
        logger.info(f"  Input shape: {audio.shape}")
        logger.info(f"  Output shape: {result.shape}")
        
        logger.info("  ✅ MBD enhancer test passed")
        return True
        
    except Exception as e:
        logger.error(f"  ❌ MBD enhancer test failed: {e}")
        return False


def run_all_tests():
    """Run all Phase 4 tests."""
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 4 CORE FUNCTIONALITY TESTS")
    logger.info("=" * 60)
    
    tests = [
        ("Module Imports", test_imports),
        ("Quality Detection", test_quality_detection),
        ("Chord Extraction", test_chord_extraction),
        ("Tempo/Key Analysis", test_tempo_key_analysis),
        ("JASCO Generator Init", test_jasco_generator_init),
        ("Stem Regenerator", test_stem_regenerator),
        ("Blender Crossfade", test_blender),
        ("MBD Enhancer", test_mbd_enhancer),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            logger.error(f"Test {name} crashed: {e}")
            results.append((name, False))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for _, r in results if r)
    failed = sum(1 for _, r in results if not r)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"  {status}: {name}")
    
    logger.info(f"\nTotal: {passed} passed, {failed} failed")
    
    if failed == 0:
        logger.info("\n🎉 All Phase 4 core tests passed!")
        return True
    else:
        logger.info(f"\n⚠️ {failed} test(s) failed. See details above.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
