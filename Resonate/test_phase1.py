#!/usr/bin/env python3
"""
Resonate Phase 1 Test Suite
Tests core pipeline components to verify implementation
"""

import sys
import os
import time
import logging
import numpy as np
import soundfile as sf
from pathlib import Path

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_utils():
    """Test utility functions."""
    logger.info("=" * 60)
    logger.info("TESTING UTILITY FUNCTIONS")
    logger.info("=" * 60)
    
    from audio_engine.utils import (
        db_to_gain, gain_to_db, normalize_peak, 
        estimate_snr, ensure_mono, format_duration
    )
    
    try:
        # Test conversions
        assert abs(db_to_gain(0) - 1.0) < 1e-10, "0 dB should be 1.0"
        assert abs(db_to_gain(-6) - 0.5) < 1e-3, "-6 dB should be 0.5"
        assert abs(gain_to_db(1.0) - 0) < 1e-10, "1.0 gain should be 0 dB"
        
        # Test normalization
        audio = np.array([0.5, -0.5, 0.3, -0.3])
        normalized = normalize_peak(audio, target_db=-1.0)
        assert abs(np.max(np.abs(normalized)) - db_to_gain(-1.0)) < 1e-3, "Normalization failed"
        
        # Test SNR
        snr = estimate_snr(np.array([0.5, 0.5, 0.5]))
        assert snr > 20, "SNR calculation failed"
        
        # Test mono conversion
        stereo = np.array([[0.5, 0.5], [0.3, 0.3], [0.1, 0.1]])
        mono = ensure_mono(stereo)
        assert mono.shape == (3,), "Stereo to mono failed"
        
        # Test duration formatting
        assert format_duration(65) == "01:05", "Duration formatting failed"
        
        logger.info("✅ All utility tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Utility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_device_management():
    """Test MPS device management."""
    logger.info("=" * 60)
    logger.info("TESTING DEVICE MANAGEMENT")
    logger.info("=" * 60)
    
    from audio_engine.device import DeviceManager, get_device
    
    try:
        manager = DeviceManager(memory_fraction=0.75)
        logger.info(f"Device: {manager.device}")
        logger.info(f"Device type: {manager.device_type}")
        
        info = manager.get_info()
        logger.info(f"Device info: {info}")
        
        # Test device access
        device = get_device()
        logger.info(f"Global device: {device}")
        
        logger.info("✅ Device management test passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Device management test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ingest():
    """Test audio ingestion."""
    logger.info("=" * 60)
    logger.info("TESTING AUDIO INGESTION")
    logger.info("=" * 60)
    
    from audio_engine.ingest import AudioIngest, AudioBuffer
    
    try:
        # Create test audio
        sample_rate = 44100
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        test_audio = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.randn(len(t))
        
        # Save test file
        test_file = "test_ingest.wav"
        sf.write(test_file, test_audio, sample_rate)
        
        # Test ingest
        ingest = AudioIngest(target_sample_rate=44100, normalize=True)
        buffer = ingest.load(test_file)
        
        logger.info(f"Loaded buffer: {buffer}")
        logger.info(f"Metadata: {buffer.metadata.to_dict()}")
        
        if buffer.analysis:
            logger.info(f"Analysis: {buffer.analysis.to_dict()}")
        
        # Test test tone creation
        test_buffer = ingest.create_test_tone(duration=1.0, frequency=440)
        logger.info(f"Test tone: {test_buffer}")
        
        # Cleanup
        ingest.cleanup()
        os.remove(test_file)
        
        logger.info("✅ Ingestion test passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Ingestion test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_enhancers():
    """Test individual stem enhancers."""
    logger.info("=" * 60)
    logger.info("TESTING STEM ENHANCERS")
    logger.info("=" * 60)
    
    from audio_engine.enhancers.vocal import VocalEnhancer
    from audio_engine.enhancers.drums import DrumEnhancer
    from audio_engine.enhancers.bass import BassEnhancer
    from audio_engine.enhancers.instruments import InstrumentEnhancer
    
    try:
        sample_rate = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Test vocal enhancer
        logger.info("Testing VocalEnhancer...")
        vocal_audio = 0.4 * np.sin(2 * np.pi * 440 * t) + 0.05 * np.random.randn(len(t))
        vocal_result = VocalEnhancer().enhance(vocal_audio, sample_rate)
        logger.info(f"  ✅ Vocal: {vocal_result.enhancements_applied}")
        
        # Test drum enhancer
        logger.info("Testing DrumEnhancer...")
        drum_audio = 0.5 * np.sin(2 * np.pi * 50 * t) * np.exp(-t) + 0.05 * np.random.randn(len(t))
        drum_result = DrumEnhancer().enhance(drum_audio, sample_rate)
        logger.info(f"  ✅ Drums: {drum_result.enhancements_applied}")
        
        # Test bass enhancer
        logger.info("Testing BassEnhancer...")
        bass_audio = 0.4 * np.sin(2 * np.pi * 80 * t) + 0.03 * np.random.randn(len(t))
        bass_result = BassEnhancer().enhance(bass_audio, sample_rate)
        logger.info(f"  ✅ Bass: {bass_result.enhancements_applied}")
        
        # Test instrument enhancer
        logger.info("Testing InstrumentEnhancer...")
        inst_audio = 0.3 * np.sin(2 * np.pi * 330 * t) + 0.02 * np.random.randn(len(t))
        inst_result = InstrumentEnhancer().enhance(inst_audio, sample_rate)
        logger.info(f"  ✅ Instruments: {inst_result.enhancements_applied}")
        
        logger.info("✅ All enhancer tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Enhancer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mixing():
    """Test stem mixing."""
    logger.info("=" * 60)
    logger.info("TESTING STEM MIXING")
    logger.info("=" * 60)
    
    from audio_engine.mixing import StemMixer
    
    try:
        sample_rate = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Create test stems
        stems = {
            'vocals': 0.4 * np.sin(2 * np.pi * 440 * t),
            'drums': 0.5 * np.sin(2 * np.pi * 50 * t) * np.exp(-t % 1),
            'bass': 0.4 * np.sin(2 * np.pi * 80 * t),
            'other': 0.3 * np.sin(2 * np.pi * 330 * t)
        }
        
        # Test mixer
        mixer = StemMixer()
        logger.info(f"Mixer: {mixer}")
        
        result = mixer.mix(stems, sample_rate)
        logger.info(f"Mix result: {result.to_dict()}")
        
        # Test auto balance
        logger.info("Testing auto-balance...")
        balance_result = mixer.auto_balance(stems, sample_rate)
        logger.info(f"Auto-balanced levels: {balance_result.stem_levels}")
        
        # Save mix for verification
        sf.write("test_mix.wav", result.mixed_audio, sample_rate)
        os.remove("test_mix.wav")
        
        logger.info("✅ Mixing test passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Mixing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mastering():
    """Test audio mastering."""
    logger.info("=" * 60)
    logger.info("TESTING AUDIO MASTERING")
    logger.info("=" * 60)
    
    from audio_engine.mastering import AudioMaster
    
    try:
        sample_rate = 44100
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Create test audio
        audio = (
            0.5 * np.sin(2 * np.pi * 440 * t) +
            0.3 * np.sin(2 * np.pi * 880 * t) +
            0.02 * np.random.randn(len(t))
        )
        
        # Test master
        master = AudioMaster()
        logger.info(f"Master: {master}")
        
        result = master.master(audio, sample_rate, "test_master.wav")
        logger.info(f"Master result: {result.to_dict()}")
        
        # Verify file was created
        assert os.path.exists("test_master.wav"), "Mastering output file not created"
        
        # Check loudness
        assert result.loudness_lufs > -20 and result.loudness_lufs < -10, "Loudness out of range"
        
        # Cleanup
        os.remove("test_master.wav")
        
        logger.info("✅ Mastering test passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Mastering test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pipeline():
    """Test the complete pipeline."""
    logger.info("=" * 60)
    logger.info("TESTING COMPLETE PIPELINE")
    logger.info("=" * 60)
    
    from audio_engine.pipeline import AudioPipeline, create_pipeline
    
    try:
        # Create test audio
        sample_rate = 44100
        duration = 3.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Create test mix
        test_audio = (
            0.3 * np.sin(2 * np.pi * 440 * t) +  # Vocals
            0.2 * np.sin(2 * np.pi * 50 * t) * np.exp(-t % 0.5) +  # Drums
            0.2 * np.sin(2 * np.pi * 80 * t) +  # Bass
            0.15 * np.sin(2 * np.pi * 330 * t) +  # Guitar
            0.05 * np.random.randn(len(t))  # Noise
        )
        
        # Normalize
        test_audio = test_audio / np.max(np.abs(test_audio)) * 0.8
        
        # Save test file
        test_file = "test_pipeline_input.wav"
        sf.write(test_file, test_audio, sample_rate)
        
        # Create pipeline
        pipeline = create_pipeline(mode="preview")
        logger.info(f"Pipeline: {pipeline}")
        
        # Get info
        info = pipeline.get_info()
        logger.info(f"Pipeline config: {info['config']}")
        
        # Process
        logger.info("Running preview pipeline...")
        start_time = time.time()
        result = pipeline.process(test_file, "test_pipeline_output.wav")
        elapsed = time.time() - start_time
        
        if result.success:
            logger.info(f"✅ Pipeline completed in {elapsed:.1f}s")
            logger.info(f"Output file: {result.output_file}")
            logger.info(f"Stage times: {result.stage_times}")
            
            # Verify output
            assert os.path.exists(result.output_file), "Output file not created"
            output_audio = sf.read(result.output_file)[0]
            logger.info(f"Output audio shape: {output_audio.shape}")
            
            # Cleanup
            os.remove(result.output_file)
            pipeline.cleanup()
        else:
            logger.error(f"❌ Pipeline failed: {result.error_message}")
        
        os.remove(test_file)
        
        logger.info("✅ Pipeline test passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all Phase 1 tests."""
    logger.info("=" * 60)
    logger.info("🚀 STARTING RESONATE PHASE 1 TEST SUITE")
    logger.info("=" * 60)
    
    tests = [
        ("Utilities", test_utils),
        ("Device Management", test_device_management),
        ("Ingestion", test_ingest),
        ("Enhancers", test_enhancers),
        ("Mixing", test_mixing),
        ("Mastering", test_mastering),
        ("Complete Pipeline", test_pipeline),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            logger.error(f"Test {name} crashed: {e}")
            results.append((name, False))
    
    # Summary
    logger.info("=" * 60)
    logger.info("📊 TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{status}: {name}")
    
    logger.info("=" * 60)
    logger.info(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED! Ready for Phase 2.")
        return True
    else:
        logger.error(f"⚠️ {total - passed} tests failed. Fix issues before Phase 2.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
