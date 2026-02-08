"""
Tests for restoration modules (frequency restoration and dereverberation).
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio_engine.restoration.frequency import (
    FrequencyRestorer, FrequencyRestorationConfig, restore_frequency
)
from audio_engine.restoration.dereverberation import (
    Dereverberator, DereverberationConfig, dereverberate
)


class TestFrequencyRestorer:
    """Test suite for FrequencyRestorer class."""
    
    @pytest.fixture
    def test_audio(self):
        """Create test audio with limited frequency content."""
        sample_rate = 44100
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        # Audio with limited frequency content (midrange only)
        audio = (
            0.4 * np.sin(2 * np.pi * 440 * t) +   # Midrange
            0.2 * np.sin(2 * np.pi * 1000 * t) +  # High-mid
            0.1 * np.sin(2 * np.pi * 150 * t)     # Bass
        )
        return audio.astype(np.float32)
    
    @pytest.fixture
    def clean_audio(self):
        """Create clean test audio."""
        sample_rate = 44100
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        return 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    
    @pytest.fixture
    def silent_audio(self):
        """Create silent test audio."""
        sample_rate = 44100
        duration = 1.0
        return np.zeros(int(sample_rate * duration), dtype=np.float32)
    
    def test_initialization(self):
        """Test restorer initialization."""
        sr = 44100
        restorer = FrequencyRestorer(sr, intensity=0.5)
        assert restorer.sample_rate == sr
        assert restorer.config.intensity == 0.5
    
    def test_intensity_clamping(self):
        """Test that intensity is properly clamped."""
        sr = 44100
        # Test negative intensity
        restorer = FrequencyRestorer(sr, intensity=-0.5)
        assert restorer.config.intensity == 0.0
        
        # Test > 1 intensity
        restorer = FrequencyRestorer(sr, intensity=1.5)
        assert restorer.config.intensity == 1.0
    
    def test_process_clean_audio(self, clean_audio):
        """Test processing clean audio."""
        sr = 44100
        restorer = FrequencyRestorer(sr, intensity=0.5)
        result = restorer.process(clean_audio)
        
        assert result.shape == clean_audio.shape
        assert result.dtype == np.float32
        # Should not be significantly louder
        assert np.max(np.abs(result)) <= 1.0
    
    def test_process_returns_float32(self, clean_audio):
        """Test that output is float32."""
        sr = 44100
        restorer = FrequencyRestorer(sr, intensity=0.5)
        result = restorer.process(clean_audio)
        
        assert result.dtype == np.float32
    
    def test_intensity_zero_returns_original(self, clean_audio):
        """Test that intensity=0 returns original audio."""
        sr = 44100
        restorer = FrequencyRestorer(sr, intensity=0.0)
        result = restorer.process(clean_audio)
        
        # With very low intensity, should be close to original
        np.testing.assert_array_almost_equal(result, clean_audio, decimal=1)
    
    def test_intensity_one_applies_full(self, clean_audio):
        """Test that intensity=1 applies full restoration."""
        sr = 44100
        restorer = FrequencyRestorer(sr, intensity=1.0)
        result = restorer.process(clean_audio)
        
        assert result.shape == clean_audio.shape
        assert np.max(np.abs(result)) <= 1.0
    
    def test_silent_audio_handling(self, silent_audio):
        """Test handling of silent audio."""
        sr = 44100
        restorer = FrequencyRestorer(sr, intensity=0.5)
        result = restorer.process(silent_audio)
        
        assert result.shape == silent_audio.shape
        assert np.all(result == 0)
    
    def test_empty_audio(self):
        """Test handling of empty audio."""
        sr = 44100
        restorer = FrequencyRestorer(sr, intensity=0.5)
        empty = np.array([], dtype=np.float32)
        result = restorer.process(empty)
        
        assert result.shape == empty.shape
    
    def test_no_clipping(self, clean_audio):
        """Test that processing doesn't cause clipping."""
        sr = 44100
        restorer = FrequencyRestorer(sr, intensity=1.0)
        result = restorer.process(clean_audio)
        
        assert np.max(np.abs(result)) <= 1.0
    
    def test_get_info(self):
        """Test getting restorer info."""
        sr = 44100
        restorer = FrequencyRestorer(sr, intensity=0.5)
        info = restorer.get_info()
        
        assert 'sample_rate' in info
        assert 'config' in info
        assert info['sample_rate'] == sr
        assert 'intensity' in info['config']
    
    def test_repr(self):
        """Test string representation."""
        sr = 44100
        restorer = FrequencyRestorer(sr, intensity=0.5)
        repr_str = repr(restorer)
        
        assert 'FrequencyRestorer' in repr_str
        assert str(sr) in repr_str
    
    def test_convenience_function(self, clean_audio):
        """Test convenience function."""
        sr = 44100
        result = restore_frequency(clean_audio, sr, intensity=0.5)
        
        assert result.shape == clean_audio.shape
        assert result.dtype == np.float32


class TestDereverberator:
    """Test suite for Dereverberator class."""
    
    @pytest.fixture
    def reverberant_audio(self):
        """Create audio with simulated reverb."""
        sample_rate = 44100
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Clean signal
        clean = 0.5 * np.sin(2 * np.pi * 440 * t)
        
        # Add reverb (delayed copies)
        reverb = clean.copy()
        for delay_ms in [20, 50, 100, 200]:
            delay_samples = int(delay_ms * sample_rate / 1000)
            reverb += 0.15 * np.roll(clean, delay_samples)
        
        return reverb.astype(np.float32)
    
    @pytest.fixture
    def clean_audio(self):
        """Create clean test audio."""
        sample_rate = 44100
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        return 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    
    @pytest.fixture
    def silent_audio(self):
        """Create silent test audio."""
        sample_rate = 44100
        duration = 1.0
        return np.zeros(int(sample_rate * duration), dtype=np.float32)
    
    def test_initialization(self):
        """Test dereverberator initialization."""
        sr = 44100
        dereverb = Dereverberator(sr, intensity=0.3)
        assert dereverb.sample_rate == sr
        assert dereverb.config.intensity == 0.3
    
    def test_intensity_clamping(self):
        """Test that intensity is properly clamped."""
        sr = 44100
        dereverb = Dereverberator(sr, intensity=-0.5)
        assert dereverb.config.intensity == 0.0
        
        dereverb = Dereverberator(sr, intensity=1.5)
        assert dereverb.config.intensity == 1.0
    
    def test_process_clean_audio(self, clean_audio):
        """Test processing clean audio."""
        sr = 44100
        dereverb = Dereverberator(sr, intensity=0.3)
        result = dereverb.process(clean_audio)
        
        assert result.shape == clean_audio.shape
        assert result.dtype == np.float32
        # Should not be significantly louder
        assert np.max(np.abs(result)) <= 1.0
    
    def test_process_reverberant_audio(self, reverberant_audio):
        """Test processing reverberant audio."""
        sr = 44100
        dereverb = Dereverberator(sr, intensity=0.3)
        result = dereverb.process(reverberant_audio)
        
        assert result.shape == reverberant_audio.shape
        assert result.dtype == np.float32
        assert np.max(np.abs(result)) <= 1.0
    
    def test_process_returns_float32(self, clean_audio):
        """Test that output is float32."""
        sr = 44100
        dereverb = Dereverberator(sr, intensity=0.3)
        result = dereverb.process(clean_audio)
        
        assert result.dtype == np.float32
    
    def test_intensity_zero_returns_original(self, clean_audio):
        """Test that intensity=0 returns original audio."""
        sr = 44100
        dereverb = Dereverberator(sr, intensity=0.0)
        result = dereverb.process(clean_audio)
        
        # With zero intensity, should be close to original
        np.testing.assert_array_almost_equal(result, clean_audio, decimal=1)
    
    def test_silent_audio_handling(self, silent_audio):
        """Test handling of silent audio."""
        sr = 44100
        dereverb = Dereverberator(sr, intensity=0.3)
        result = dereverb.process(silent_audio)
        
        assert result.shape == silent_audio.shape
        assert np.all(result == 0)
    
    def test_empty_audio(self):
        """Test handling of empty audio."""
        sr = 44100
        dereverb = Dereverberator(sr, intensity=0.3)
        empty = np.array([], dtype=np.float32)
        result = dereverb.process(empty)
        
        assert result.shape == empty.shape
    
    def test_no_clipping(self, clean_audio):
        """Test that processing doesn't cause clipping."""
        sr = 44100
        dereverb = Dereverberator(sr, intensity=1.0)
        result = dereverb.process(clean_audio)
        
        assert np.max(np.abs(result)) <= 1.0
    
    def test_get_info(self):
        """Test getting dereverberator info."""
        sr = 44100
        dereverb = Dereverberator(sr, intensity=0.3)
        info = dereverb.get_info()
        
        assert 'sample_rate' in info
        assert 'config' in info
        assert info['sample_rate'] == sr
        assert 'intensity' in info['config']
    
    def test_repr(self):
        """Test string representation."""
        sr = 44100
        dereverb = Dereverberator(sr, intensity=0.3)
        repr_str = repr(dereverb)
        
        assert 'Dereverberator' in repr_str
        assert str(sr) in repr_str
    
    def test_convenience_function(self, clean_audio):
        """Test convenience function."""
        sr = 44100
        result = dereverberate(clean_audio, sr, intensity=0.3)
        
        assert result.shape == clean_audio.shape
        assert result.dtype == np.float32


class TestRestorationIntegration:
    """Integration tests for restoration modules."""
    
    def test_both_restorers_sequential(self):
        """Test using both restorers sequentially."""
        sr = 44100
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))
        
        # Create test audio
        audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        
        # Apply frequency restoration
        freq_restorer = FrequencyRestorer(sr, intensity=0.5)
        audio = freq_restorer.process(audio)
        
        # Apply dereverberation
        dereverb = Dereverberator(sr, intensity=0.3)
        audio = dereverb.process(audio)
        
        # Verify output
        assert audio.shape == (int(sr * duration),)
        assert audio.dtype == np.float32
        assert np.max(np.abs(audio)) <= 1.0
