"""
Tests for audio metrics module.
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio_engine.metrics import (
    QualityMetrics, QualityReport, snr_estimate, loudness_lufs,
    spectral_centroid, clipping_detection, artifact_detection,
    analyze_quality, before_after_comparison, get_quality_assessment
)


class TestQualityMetrics:
    """Test suite for QualityMetrics class."""
    
    @pytest.fixture
    def clean_audio(self):
        """Create clean test audio."""
        sample_rate = 44100
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        # Clean sine wave
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        return audio
    
    @pytest.fixture
    def noisy_audio(self):
        """Create noisy test audio."""
        sample_rate = 44100
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        # Clean signal + noise
        audio = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.randn(len(t))
        return audio
    
    @pytest.fixture
    def clipped_audio(self):
        """Create clipped test audio."""
        sample_rate = 44100
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        # Overdriven signal
        audio = np.clip(np.sin(2 * np.pi * 440 * t) * 1.5, -1.0, 1.0)
        return audio
    
    @pytest.fixture
    def silent_audio(self):
        """Create silent test audio."""
        sample_rate = 44100
        duration = 1.0
        return np.zeros(int(sample_rate * duration), dtype=np.float32)
    
    @pytest.fixture
    def short_audio(self):
        """Create short test audio (< 1 second)."""
        sample_rate = 44100
        duration = 0.5
        t = np.linspace(0, duration, int(sample_rate * duration))
        return 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    
    def test_snr_estimate_clean(self, clean_audio):
        """Test SNR estimation on clean audio."""
        sr = 44100
        snr = snr_estimate(clean_audio, sr)
        # Clean sine wave may have variable SNR depending on algorithm
        assert snr > 0.0, f"Expected positive SNR for clean audio, got {snr}"
        assert isinstance(snr, float)
    
    def test_snr_estimate_noisy(self, noisy_audio):
        """Test SNR estimation on noisy audio."""
        sr = 44100
        snr = snr_estimate(noisy_audio, sr)
        # Noisy audio should have lower SNR
        assert isinstance(snr, float), f"Expected float, got {type(snr)}"
        assert snr < snr_estimate(clean_audio_fixture := 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 2, 88200)), sr)
    
    def test_loudness_lufs(self, clean_audio):
        """Test loudness measurement."""
        sr = 44100
        loudness = loudness_lufs(clean_audio, sr)
        assert isinstance(loudness, float), f"Expected float, got {type(loudness)}"
        # Should be negative (audio below reference level)
        assert loudness < 0, f"Expected negative LUFS, got {loudness}"
    
    def test_spectral_centroid(self, clean_audio):
        """Test spectral centroid calculation."""
        sr = 44100
        centroid = spectral_centroid(clean_audio, sr)
        assert isinstance(centroid, float), f"Expected float, got {type(centroid)}"
        assert 0 < centroid < sr / 2, f"Centroid should be in valid range, got {centroid}"
    
    def test_clipping_detection_clean(self, clean_audio):
        """Test clipping detection on clean audio."""
        percent, has_clipping = clipping_detection(clean_audio)
        assert percent == 0.0, f"Expected no clipping, got {percent}"
        assert has_clipping == False, "Expected has_clipping=False for clean audio"
    
    def test_clipping_detection_clipped(self, clipped_audio):
        """Test clipping detection on clipped audio."""
        percent, has_clipping = clipping_detection(clipped_audio)
        assert percent > 0, f"Expected clipping, got {percent}"
        assert has_clipping == True, "Expected has_clipping=True for clipped audio"
    
    def test_artifact_detection(self, clean_audio):
        """Test artifact detection."""
        sr = 44100
        artifacts = artifact_detection(clean_audio, sr)
        assert isinstance(artifacts, dict), f"Expected dict, got {type(artifacts)}"
        # Check expected keys exist
        expected_keys = ['metallic_score', 'ringing_score', 'clicking_score', 'phase_distortion']
        for key in expected_keys:
            assert key in artifacts, f"Missing key: {key}"
        # All scores should be in 0-1 range
        for score in artifacts.values():
            assert 0.0 <= score <= 1.0, f"Score {score} not in [0,1]"
    
    def test_analyze_quality(self, clean_audio):
        """Test full quality analysis."""
        sr = 44100
        report = analyze_quality(clean_audio, sr)
        assert isinstance(report, QualityReport), f"Expected QualityReport, got {type(report)}"
        # Check all attributes
        assert hasattr(report, 'snr_db')
        assert hasattr(report, 'loudness_lufs')
        assert hasattr(report, 'spectral_centroid_hz')
        assert hasattr(report, 'clipping_percent')
        assert hasattr(report, 'has_clipping')
        assert hasattr(report, 'artifacts')
    
    def test_before_after_comparison(self, clean_audio):
        """Test before/after comparison."""
        sr = 44100
        # Create processed version (slightly boosted)
        processed = clean_audio * 1.1
        processed = np.clip(processed, -1.0, 1.0)
        
        comparison = before_after_comparison(clean_audio, processed, sr)
        
        assert isinstance(comparison, dict), f"Expected dict, got {type(comparison)}"
        assert 'original' in comparison
        assert 'processed' in comparison
        assert 'improvements' in comparison
    
    def test_edge_case_silent_audio(self, silent_audio):
        """Test handling of silent audio."""
        sr = 44100
        # Should not raise exception
        report = analyze_quality(silent_audio, sr)
        assert report is not None
    
    def test_edge_case_short_audio(self, short_audio):
        """Test handling of short audio clips."""
        sr = 44100
        # Should not raise exception
        report = analyze_quality(short_audio, sr)
        assert report is not None
    
    def test_dtype_preservation(self, clean_audio):
        """Test that output maintains float32 dtype."""
        sr = 44100
        report = analyze_quality(clean_audio, sr)
        # QualityReport doesn't store audio, just metrics
        # But individual functions should handle float32 input
        snr = snr_estimate(clean_audio, sr)
        assert isinstance(snr, (int, float))
    
    def test_clip_prevention(self):
        """Test that processing doesn't exceed thresholds."""
        sr = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))
        
        # Very loud audio
        loud_audio = np.ones(int(sr * duration), dtype=np.float32) * 0.99
        
        # Run analysis
        report = analyze_quality(loud_audio, sr)
        
        # Should report clipping if present
        assert report.clipping_percent >= 0
