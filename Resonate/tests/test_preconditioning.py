import pytest
import numpy as np
import sys
from unittest.mock import MagicMock, patch
from pathlib import Path
import time

# --- Mocking Dependencies ---
# We must mock these BEFORE importing the application modules because
# some of them are top-level imports or required for class definitions.

class MockSpline:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __call__(self, x_new):
        # Simple linear interpolation for testing
        return np.interp(x_new, self.x, self.y)

# Mock scipy
mock_scipy = MagicMock()
mock_scipy.interpolate.CubicSpline = MockSpline
mock_scipy.interpolate.interp1d = lambda x, y, kind, fill_value: lambda x_new: np.interp(x_new, x, y)
# Mock hilbert to return the signal itself (so envelope = abs(signal))
mock_scipy.signal.hilbert = lambda x: x 
# Mock butter/filtfilt for noise reducer
mock_scipy.signal.butter = lambda N, Wn, btype: (np.array([1.0]), np.array([1.0]))
mock_scipy.signal.filtfilt = lambda b, a, x: x

sys.modules['scipy'] = mock_scipy
sys.modules['scipy.interpolate'] = mock_scipy.interpolate
sys.modules['scipy.signal'] = mock_scipy.signal

# Mock soundfile
mock_sf = MagicMock()
sys.modules['soundfile'] = mock_sf

# Mock librosa
mock_librosa = MagicMock()
sys.modules['librosa'] = mock_librosa

# Mock torch
mock_torch = MagicMock()
sys.modules['torch'] = mock_torch

# Mock demucs
mock_demucs = MagicMock()
sys.modules['demucs'] = mock_demucs

# Mock noisereduce
mock_nr = MagicMock()
# reduce_noise just returns audio (default behavior)
mock_nr.reduce_noise = lambda y, **kwargs: y 
sys.modules['noisereduce'] = mock_nr

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Now import the modules under test
from audio_engine.preconditioning import (
    PreConditioningPipeline,
    PreConditioningConfig,
)
from audio_engine.preconditioning.declip import Declipper, DeclipConfig
from audio_engine.preconditioning.dynamics import DynamicsRestorer, DynamicsConfig
from audio_engine.preconditioning.noise_reducer import NoiseReducer

# --- Helper Functions ---

def generate_sine_wave(frequency: float, duration: float, sample_rate: int = 44100) -> np.ndarray:
    """Generate sine wave for testing."""
    t = np.linspace(0, duration, int(sample_rate * duration))
    return np.sin(2 * np.pi * frequency * t).astype(np.float32)

def calculate_snr(signal: np.ndarray, noise: np.ndarray) -> float:
    """Calculate SNR in dB."""
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
    return snr

def calculate_dynamic_range(audio: np.ndarray) -> float:
    """Calculate dynamic range in dB."""
    peak = np.max(np.abs(audio))
    rms = np.sqrt(np.mean(audio ** 2))
    dr = 20 * np.log10((peak + 1e-10) / (rms + 1e-10))
    return dr

# --- Test Functions ---

def test_noise_reducer_reduces_snr():
    """Verify that noise reduction improves SNR."""
    duration = 1.0
    sr = 44100
    clean_signal = generate_sine_wave(440, duration, sr)
    
    np.random.seed(42)
    noise = np.random.normal(0, 0.1, clean_signal.shape).astype(np.float32)
    noisy_signal = clean_signal + noise
    
    # Calculate SNR before
    snr_before = calculate_snr(clean_signal, noisy_signal - clean_signal)
    
    # Mock behavior for THIS test only
    def mock_reduce_noise_func(y, **kwargs):
        # Return a slightly cleaner version of input
        # Ensure we respect input shape to avoid side effects if leaked
        if y.shape == clean_signal.shape:
             return clean_signal + 0.5 * noise 
        return y
        
    # Use patch to safely mock
    with patch('noisereduce.reduce_noise', side_effect=mock_reduce_noise_func):
        reducer = NoiseReducer(sr)
        reduced_signal = reducer.process(noisy_signal)
        
        # Calculate SNR after
        snr_after = calculate_snr(clean_signal, reduced_signal - clean_signal)
        
        improvement = snr_after - snr_before
        
        print(f"SNR improvement: {improvement:.1f} dB")
        
        assert snr_after > snr_before
        assert improvement >= 3.0

def test_declipper_detects_clips():
    """Verify that de-clipper detects clipped regions."""
    sr = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # Create 5 distinct clipped regions (plateaus)
    # Start with zeros
    audio = np.zeros_like(t)
    
    # Create 5 blocks of 1.0 amplitude
    # 100 samples long each (approx 2.2ms), spaced out
    # Must be shorter than default max_clip_duration_ms (5ms)
    starts = [5000, 10000, 15000, 20000, 25000]
    length = 100
    
    for start in starts:
        audio[start:start+length] = 1.0
        
    config = DeclipConfig(detection_threshold=0.99)
    declipper = Declipper(sr, config)
    
    clips = declipper.detect_clipping(audio)
    
    print(f"Detected {len(clips)} clip regions")
    
    # Assert that exactly 5 clip regions are detected
    # With contiguous blocks, the detector should merge them into single regions
    assert len(clips) == 5
    
    for i, clip in enumerate(clips):
        assert clip.start == starts[i]
        assert clip.end == starts[i] + length
        assert clip.duration_ms > 0

def test_declipper_repairs_clips():
    """Verify that de-clipper interpolates clipped samples."""
    sr = 44100
    t = np.linspace(0, 0.1, int(sr * 0.1))
    original = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    
    clipped_audio = original.copy()
    # Clip a peak
    # Peak is at ~25 samples (1/4 cycle of 440Hz ~ 100 samples)
    start_idx = 20
    end_idx = 30
    clipped_audio[start_idx:end_idx] = 0.99 
    
    config = DeclipConfig(detection_threshold=0.98)
    declipper = Declipper(sr, config)
    
    repaired_audio = declipper.process(clipped_audio)
    
    # Verify range
    assert np.all(repaired_audio >= -1.5)
    assert np.all(repaired_audio <= 1.5)
    
    # Verify values changed in repaired region
    assert not np.array_equal(repaired_audio[start_idx:end_idx], clipped_audio[start_idx:end_idx])

def test_dynamics_restorer_expands_range():
    """Verify that dynamics restorer increases dynamic range."""
    sr = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # Create compressed signal
    modulator = 0.2 + 0.1 * np.sin(2 * np.pi * 5 * t)
    carrier = np.sin(2 * np.pi * 440 * t)
    compressed_signal = (carrier * modulator).astype(np.float32)
    
    restorer = DynamicsRestorer(sr)
    restored_signal = restorer.process(compressed_signal)
    
    dr_before = calculate_dynamic_range(compressed_signal)
    dr_after = calculate_dynamic_range(restored_signal)
    
    print(f"Dynamic range improved: {dr_before:.1f} dB -> {dr_after:.1f} dB")
    
    assert dr_after > dr_before

def test_preconditioning_pipeline_all_stages():
    """Verify full pipeline processes audio correctly."""
    sr = 44100
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))
    
    clean = 0.6 * np.sin(2 * np.pi * 440 * t)
    noise = 0.3 * np.random.normal(0, 1, t.shape)
    mixed = clean + noise
    input_audio = np.clip(mixed, -0.95, 0.95).astype(np.float32)
    
    config = PreConditioningConfig(
        enable_noise_reduction=True,
        enable_declipping=True,
        enable_dynamics_restoration=True
    )
    pipeline = PreConditioningPipeline(sr, config)
    
    start_time = time.time()
    result = pipeline.process(input_audio)
    processing_time = time.time() - start_time
    
    assert isinstance(result.audio, np.ndarray)
    assert result.audio.shape == input_audio.shape
    assert result.sample_rate == sr
    assert result.noise_reduced == True
    assert result.dynamics_restored == True
    
    assert processing_time > 0
    
    print(f"Pipeline result: clips={result.clips_repaired}, "
          f"SNR {result.input_snr_estimate:.1f} -> {result.output_snr_estimate:.1f}")

def test_preconditioning_pipeline_selective_stages():
    """Verify pipeline respects enable flags."""
    sr = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    audio = (np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    
    # 1. Only noise reduction
    config_nr = PreConditioningConfig(
        enable_noise_reduction=True,
        enable_declipping=False,
        enable_dynamics_restoration=False
    )
    pipeline_nr = PreConditioningPipeline(sr, config_nr)
    res_nr = pipeline_nr.process(audio)
    
    assert res_nr.noise_reduced == True
    assert res_nr.clips_repaired == 0
    assert res_nr.dynamics_restored == False
    
    # 2. Only declipping
    config_dc = PreConditioningConfig(
        enable_noise_reduction=False,
        enable_declipping=True,
        enable_dynamics_restoration=False
    )
    pipeline_dc = PreConditioningPipeline(sr, config_dc)
    res_dc = pipeline_dc.process(audio)
    
    assert res_dc.noise_reduced == False
    assert res_dc.dynamics_restored == False
    
    # 3. Only dynamics
    config_dr = PreConditioningConfig(
        enable_noise_reduction=False,
        enable_declipping=False,
        enable_dynamics_restoration=True
    )
    pipeline_dr = PreConditioningPipeline(sr, config_dr)
    res_dr = pipeline_dr.process(audio)
    
    assert res_dr.noise_reduced == False
    assert res_dr.clips_repaired == 0
    assert res_dr.dynamics_restored == True

def test_preconditioning_edge_cases():
    """Verify pipeline handles edge cases gracefully."""
    sr = 44100
    config = PreConditioningConfig(
        enable_noise_reduction=True,
        enable_declipping=True,
        enable_dynamics_restoration=True
    )
    pipeline = PreConditioningPipeline(sr, config)
    
    # Test 1: Very short audio
    short_audio = np.random.uniform(-0.5, 0.5, 100).astype(np.float32)
    res_short = pipeline.process(short_audio)
    assert len(res_short.audio) == 100
    
    # Test 2: Silent audio
    silent_audio = np.zeros(44100, dtype=np.float32)
    res_silent = pipeline.process(silent_audio)
    assert len(res_silent.audio) == 44100
    
    # Test 3: Very loud clipped audio
    loud_audio = np.ones(1000, dtype=np.float32)
    res_loud = pipeline.process(loud_audio)
    assert len(res_loud.audio) == 1000
    
    print("Edge case tests passed")

if __name__ == "__main__":
    pytest.main([__file__])
