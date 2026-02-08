"""
Integration tests for Phase 2 & 3 features.

Tests the full pipeline with:
- Quality metrics analysis
- Frequency restoration
- Dereverberation
- Cache system with segment caching
- Error handling and graceful degradation
"""

import pytest
import numpy as np
import sys
import os
import tempfile
import shutil

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio_engine.pipeline import AudioPipeline, PipelineConfig, PipelineMode
from audio_engine.cache import CacheManager, get_cache_manager
from audio_engine.metrics import QualityMetrics, analyze_quality
from audio_engine.restoration import FrequencyRestorer, Dereverberator


class TestPhase2Integration:
    """Integration tests for Phase 2 features (quality & polish)."""
    
    @pytest.fixture
    def test_audio(self):
        """Create test audio file."""
        sample_rate = 44100
        duration = 3.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Create test mix
        audio = (
            0.3 * np.sin(2 * np.pi * 440 * t) +  # Vocals (simulated)
            0.2 * np.sin(2 * np.pi * 50 * t) * np.exp(-t % 1) +  # Drums
            0.2 * np.sin(2 * np.pi * 80 * t) +  # Bass
            0.15 * np.sin(2 * np.pi * 330 * t) +  # Guitar
            0.05 * np.random.randn(len(t))  # Noise
        )
        
        # Normalize
        audio = audio / np.max(np.abs(audio)) * 0.8
        
        return audio.astype(np.float32), sample_rate
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        temp_dir = tempfile.mkdtemp(prefix="test_cache_")
        yield temp_dir
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    def test_quality_metrics_analysis(self, test_audio):
        """Test quality metrics analysis on audio."""
        audio, sr = test_audio
        
        # Analyze quality
        report = analyze_quality(audio, sr)
        
        # Verify report structure
        assert report.snr_db is not None
        assert report.loudness_lufs is not None
        assert report.spectral_centroid_hz is not None
        assert report.clipping_percent is not None
        assert report.artifacts is not None
        
        # Verify reasonable values
        assert report.snr_db > 0
        assert report.spectral_centroid_hz > 0
        assert report.spectral_centroid_hz < sr / 2
        assert report.clipping_percent >= 0
    
    def test_before_after_comparison(self, test_audio):
        """Test before/after comparison functionality."""
        audio, sr = test_audio
        
        # Create processed version
        processed = audio * 0.9  # Simple processing
        
        # Compare
        comparison = QualityMetrics.before_after_comparison(audio, processed, sr)
        
        # Verify comparison structure
        assert 'original' in comparison
        assert 'processed' in comparison
        assert 'improvements' in comparison
        assert 'snr_improvement_db' in comparison['improvements']
        assert 'loudness_shift_db' in comparison['improvements']
    
    def test_cache_manager_initialization(self, temp_cache_dir):
        """Test cache manager initialization."""
        cache = CacheManager(cache_dir=temp_cache_dir, max_size_gb=0.1)
        
        assert cache.cache_dir.exists()
        assert cache.stems_dir.exists()
        assert cache.enhanced_dir.exists()
        assert cache.metadata_dir.exists()
    
    def test_cache_stems(self, temp_cache_dir, test_audio):
        """Test caching stems."""
        audio, sr = test_audio
        
        cache = CacheManager(cache_dir=temp_cache_dir)
        
        # Create test stems
        stems = {
            'vocals': audio[:len(audio)//4],
            'drums': audio[len(audio)//4:len(audio)//2],
            'bass': audio[len(audio)//2:3*len(audio)//4],
            'other': audio[3*len(audio)//4:]
        }
        
        # Cache stems
        audio_hash = "test_audio_hash_123"
        cache_key = cache.cache_stems(audio_hash, stems, metadata={"test": True})
        
        assert cache_key is not None
        assert len(cache_key) > 0
        
        # Retrieve stems
        retrieved = cache.get_stems(audio_hash)
        
        assert retrieved is not None
        assert set(retrieved.keys()) == set(stems.keys())
    
    def test_segment_caching(self, temp_cache_dir, test_audio):
        """Test segment caching functionality."""
        audio, sr = test_audio
        
        cache = CacheManager(cache_dir=temp_cache_dir)
        
        # Split into segments
        segment_length = len(audio) // 3
        segments = [
            audio[:segment_length],
            audio[segment_length:2*segment_length],
            audio[2*segment_length:]
        ]
        
        base_key = "test_segment_key"
        
        # Cache segments
        keys = cache.cache_segments(base_key, segments)
        
        assert len(keys) == 3
        
        # Retrieve segments
        retrieved = cache.get_segments(base_key, 3)
        
        assert retrieved is not None
        assert len(retrieved) == 3
        
        # Verify content
        for i, (orig, retr) in enumerate(zip(segments, retrieved)):
            np.testing.assert_array_almost_equal(orig, retr)
    
    def test_cache_version_invalidation(self, temp_cache_dir):
        """Test cache version-based invalidation."""
        cache = CacheManager(cache_dir=temp_cache_dir)
        
        # Add a test entry
        test_audio = np.random.randn(44100).astype(np.float32)
        cache.cache_segment("test_key", 0, test_audio)
        
        # Invalidate old versions
        cache.invalidate_by_version("3.0")
        
        # Entry should be invalidated (version 2.0 < 3.0)
        result = cache.get_segment("test_key", 0)
        assert result is None


class TestPhase3Integration:
    """Integration tests for Phase 3 features (advanced restoration)."""
    
    @pytest.fixture
    def test_audio(self):
        """Create test audio for restoration."""
        sample_rate = 44100
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Audio with limited frequency content
        audio = (
            0.4 * np.sin(2 * np.pi * 440 * t) +   # Midrange
            0.2 * np.sin(2 * np.pi * 1000 * t) +  # High-mid
            0.1 * np.sin(2 * np.pi * 150 * t)     # Bass
        )
        
        return audio.astype(np.float32), sample_rate
    
    def test_frequency_restoration(self, test_audio):
        """Test frequency restoration module."""
        audio, sr = test_audio
        
        # Create restorer
        restorer = FrequencyRestorer(sr, intensity=0.5)
        
        # Process
        result = restorer.process(audio)
        
        # Verify output
        assert result.shape == audio.shape
        assert result.dtype == np.float32
        assert np.max(np.abs(result)) <= 1.0
    
    def test_dereverberation(self, test_audio):
        """Test dereverberation module."""
        audio, sr = test_audio
        
        # Create dereverberator
        dereverb = Dereverberator(sr, intensity=0.3)
        
        # Process
        result = dereverb.process(audio)
        
        # Verify output
        assert result.shape == audio.shape
        assert result.dtype == np.float32
        assert np.max(np.abs(result)) <= 1.0
    
    def test_sequential_restoration(self, test_audio):
        """Test applying both restoration modules sequentially."""
        audio, sr = test_audio
        
        # Apply frequency restoration
        freq_restorer = FrequencyRestorer(sr, intensity=0.5)
        audio = freq_restorer.process(audio)
        
        # Apply dereverberation
        dereverb = Dereverberator(sr, intensity=0.3)
        audio = dereverb.process(audio)
        
        # Verify output
        assert audio.shape == (int(sr * 2.0),)
        assert audio.dtype == np.float32
        assert np.max(np.abs(audio)) <= 1.0


class TestPipelineIntegration:
    """Integration tests for pipeline with Phase 2 & 3 features."""
    
    @pytest.fixture
    def test_audio_file(self):
        """Create test audio file."""
        sample_rate = 44100
        duration = 3.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        audio = (
            0.3 * np.sin(2 * np.pi * 440 * t) +
            0.2 * np.sin(2 * np.pi * 80 * t) +
            0.1 * np.random.randn(len(t))
        )
        audio = audio / np.max(np.abs(audio)) * 0.8
        
        # Save to temp file
        import soundfile as sf
        temp_dir = tempfile.mkdtemp()
        temp_file = os.path.join(temp_dir, "test_input.wav")
        sf.write(temp_file, audio.astype(np.float32), sample_rate)
        
        yield temp_file
        
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    def test_pipeline_with_restoration_settings(self, test_audio_file):
        """Test pipeline with restoration configuration."""
        # Create pipeline with restoration enabled
        config = PipelineConfig(
            mode=PipelineMode.PREVIEW,
            frequency_restoration=True,
            frequency_intensity=0.5,
            dereverberation=True,
            dereverb_intensity=0.3
        )
        
        pipeline = AudioPipeline(config)
        
        # Verify config
        assert pipeline.config.frequency_restoration is True
        assert pipeline.config.frequency_intensity == 0.5
        assert pipeline.config.dereverberation is True
        assert pipeline.config.dereverb_intensity == 0.3
    
    def test_pipeline_get_info_includes_restoration(self):
        """Test that pipeline info includes restoration settings."""
        config = PipelineConfig(
            frequency_restoration=True,
            frequency_intensity=0.7,
            dereverberation=True,
            dereverb_intensity=0.4
        )
        
        pipeline = AudioPipeline(config)
        info = pipeline.get_info()
        
        # Info should include restoration settings
        assert 'config' in info
        assert 'frequency_restoration' in info['config']
        assert 'frequency_intensity' in info['config']
        assert 'dereverberation' in info['config']
        assert 'dereverb_intensity' in info['config']


class TestErrorHandling:
    """Tests for error handling and graceful degradation."""
    
    def test_silent_audio_metrics(self):
        """Test metrics on silent audio."""
        sr = 44100
        silent = np.zeros(int(sr * 1.0), dtype=np.float32)
        
        # Should not raise exception
        report = analyze_quality(silent, sr)
        
        assert report is not None
    
    def test_short_audio_metrics(self):
        """Test metrics on short audio (< 1 second)."""
        sr = 44100
        short = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 0.5, int(sr * 0.5)))
        
        # Should not raise exception
        report = analyze_quality(short.astype(np.float32), sr)
        
        assert report is not None
    
    def test_clipped_audio_metrics(self):
        """Test metrics on heavily clipped audio."""
        sr = 44100
        t = np.linspace(0, 1.0, sr)
        clipped = np.clip(np.sin(2 * np.pi * 440 * t) * 2.0, -1.0, 1.0)
        
        # Should detect clipping
        percent, has_clipping = QualityMetrics.clipping_detection(clipped)
        
        assert has_clipping == True
        assert percent > 0
    
    def test_dereverb_empty_audio(self):
        """Test dereverberation on empty audio."""
        sr = 44100
        empty = np.array([], dtype=np.float32)
        
        dereverb = Dereverberator(sr, intensity=0.5)
        result = dereverb.process(empty)
        
        assert result.shape == (0,)
    
    def test_frequency_empty_audio(self):
        """Test frequency restoration on empty audio."""
        sr = 44100
        empty = np.array([], dtype=np.float32)
        
        restorer = FrequencyRestorer(sr, intensity=0.5)
        result = restorer.process(empty)
        
        assert result.shape == (0,)


class TestEdgeCases:
    """Edge case tests."""
    
    def test_audio_dtype_preservation(self):
        """Test that audio dtype is preserved through processing."""
        sr = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))
        
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        audio = audio.astype(np.float32)
        
        # Process through metrics
        report = analyze_quality(audio, sr)
        
        # Process through restoration
        restorer = FrequencyRestorer(sr, intensity=0.5)
        result = restorer.process(audio)
        
        assert result.dtype == np.float32
    
    def test_max_amplitude_not_exceeded(self):
        """Test that processing never exceeds amplitude threshold."""
        sr = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))
        
        # Create audio close to max
        audio = np.ones(int(sr * duration), dtype=np.float32) * 0.95
        
        # Apply restoration
        restorer = FrequencyRestorer(sr, intensity=1.0)
        result = restorer.process(audio)
        
        # Should not exceed 1.0
        assert np.max(np.abs(result)) <= 1.0
    
    def test_normalization_preserves_shape(self):
        """Test that audio shape is preserved."""
        sr = 44100
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))
        
        audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        
        # Process through restoration
        restorer = FrequencyRestorer(sr, intensity=0.5)
        result = restorer.process(audio)
        
        assert result.shape == audio.shape
