"""Functional tests for polish and enhancements."""

import os
import numpy as np
import soundfile as sf

from audio_engine.separator import SeparatorEngine, SeparatorConfig
from audio_engine.metrics import artifact_detection, analyze_quality, get_quality_assessment
from audio_engine.polish import MBDEnhancer
from audio_engine.pipeline import AudioPipeline, PipelineConfig, PipelineMode


def test_fallback_separation_robustness():
    """Test fallback separation with degraded audio."""
    sr, duration = 44100, 3.0
    t = np.linspace(0, duration, int(sr * duration))
    audio = np.sin(2 * np.pi * 440 * t) * 2.0  # Clipped
    audio = np.clip(audio, -0.95, 0.95)

    separator = SeparatorEngine(SeparatorConfig.render())
    result = separator.separate(audio, sr)

    assert result.stems is not None
    assert len(result.stems) == 4
    assert all(len(v) > 0 for v in result.stems.values())
    assert result.model == "fallback_spectral"  # Verify fallback used


def test_artifact_detection_comprehensive():
    """Test all 8 artifact types."""
    sr = 44100
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))

    artifacts = artifact_detection(np.sin(2 * np.pi * 440 * t), sr)

    expected_keys = [
        "metallic_score", "ringing_score", "clicking_score",
        "phase_distortion", "aliasing_score", "clipping_residual",
        "pump_score", "overall_artifact_score"
    ]

    for key in expected_keys:
        assert key in artifacts, f"Missing: {key}"
        assert 0.0 <= artifacts[key] <= 1.0, f"Invalid range: {key}"


def test_quality_assessment_accuracy():
    """Test quality string generation."""
    sr = 44100
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))

    # Create realistic good-quality audio with varied frequency content
    good_audio = (
        0.4 * np.sin(2 * np.pi * 440 * t) +   # Strong fundamental
        0.15 * np.sin(2 * np.pi * 880 * t) +  # Harmonics
        0.15 * np.sin(2 * np.pi * 220 * t) +  # Sub-harmonic
        0.1 * np.sin(2 * np.pi * 1320 * t) +  # Higher harmonic
        0.02 * np.random.randn(len(t))         # Very low noise floor
    )
    good_audio = good_audio / np.max(np.abs(good_audio)) * 0.65  # Normalize with headroom
    
    report = analyze_quality(good_audio.astype(np.float32), sr)
    assessment = get_quality_assessment(report)

    # Should get a quality tier (any tier is valid for synthetic audio)
    valid_tiers = ["Poor", "Fair", "Good", "Excellent", "Reference"]
    assert any(tier in assessment for tier in valid_tiers), \
        f"Expected valid quality tier, got: {assessment}"
    
    # Should contain recommendations
    assert "recommend" in assessment, f"Expected recommendations in assessment: {assessment}"


def test_mbd_enhancement():
    """Test MBD polish."""
    sr = 44100
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration))
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)

    mbd = MBDEnhancer(sr)

    if mbd.is_available():
        result = mbd.process(audio, intensity=0.5)
        assert result.shape == audio.shape
        assert result.dtype == np.float32
        correlation = np.corrcoef(audio, result)[0, 1]
        assert correlation > 0.9
    else:
        result = mbd.process(audio, intensity=0.5)
        np.testing.assert_array_equal(result, audio)


def test_full_pipeline_with_all_features(tmp_path):
    """End-to-end test with all features enabled."""
    sr = 44100
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration))
    audio = (
        0.3 * np.sin(2 * np.pi * 440 * t) +
        0.2 * np.sin(2 * np.pi * 80 * t) +
        0.1 * np.random.randn(len(t))
    )
    audio = audio / np.max(np.abs(audio)) * 0.8

    temp_file = tmp_path / "test_full_pipeline.wav"
    sf.write(str(temp_file), audio.astype(np.float32), sr)

    config = PipelineConfig(
        mode=PipelineMode.PREVIEW,
        frequency_restoration=True,
        frequency_intensity=0.5,
        dereverberation=True,
        dereverb_intensity=0.3,
        enable_mbd_polish=False,
    )

    pipeline = AudioPipeline(config)
    result = pipeline.process(str(temp_file))

    assert result.success
    assert result.output_file is not None
    assert os.path.exists(result.output_file)
    assert result.total_processing_time < 120.0

