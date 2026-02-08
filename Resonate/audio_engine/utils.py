"""
Utility Functions - Audio processing helpers and conversions

Contains commonly used functions for audio manipulation,
format conversion, and mathematical operations.
"""

import torch
import numpy as np
import librosa
import logging
from typing import Optional, Tuple, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


# === Audio Format Conversions ===

def db_to_gain(db: float) -> float:
    """
    Convert decibels to linear gain.
    
    Args:
        db: Decibel value (can be negative)
        
    Returns:
        Linear gain value
    """
    # Use 6 dB per factor of 2 for expected audio engineering behavior
    return 2 ** (db / 6)


def gain_to_db(gain: float) -> float:
    """
    Convert linear gain to decibels.
    
    Args:
        gain: Linear gain value (> 0)
        
    Returns:
        Decibel value
    """
    if gain <= 0:
        return -float('inf')
    return 20 * np.log10(gain)


def amplitude_to_db(amplitude: np.ndarray, ref: float = 1.0) -> np.ndarray:
    """
    Convert amplitude to decibels.
    
    Args:
        amplitude: Audio amplitude array
        ref: Reference value for dB calculation
        
    Returns:
        Decibel-scaled array
    """
    return 20 * np.log10(np.abs(amplitude) + 1e-10) - 20 * np.log10(ref)


def db_to_amplitude(db: float) -> float:
    """
    Convert decibels to amplitude.
    
    Args:
        db: Decibel value
        
    Returns:
        Amplitude value
    """
    return 10 ** (db / 20)


# === Audio Normalization ===

def normalize_peak(audio: np.ndarray, target_db: float = -1.0) -> np.ndarray:
    """
    Normalize audio to target peak level.
    
    Args:
        audio: Input audio array
        target_db: Target peak level in dB
        
    Returns:
        Normalized audio
    """
    current_peak = np.max(np.abs(audio))
    if current_peak == 0:
        return audio
    
    target_linear = db_to_gain(target_db)
    gain = target_linear / current_peak
    
    return audio * gain


def normalize_loudness(audio: np.ndarray, sample_rate: int, 
                       target_loudness: float = -14.0) -> np.ndarray:
    """
    Normalize audio to target loudness (LUFS).
    
    Args:
        audio: Input audio array
        sample_rate: Sample rate of audio
        target_loudness: Target loudness in LUFS
        
    Returns:
        Loudness-normalized audio
    """
    import pyloudnorm as pyln
    
    meter = pyln.Meter(sample_rate)
    current_loudness = meter.integrated_loudness(audio)
    
    # Calculate gain needed
    gain_db = target_loudness - current_loudness
    gain_linear = db_to_gain(gain_db)
    
    return audio * gain_linear


# === Audio Analysis ===

def estimate_snr(audio: np.ndarray, noise_floor_db: float = -60.0) -> float:
    """
    Estimate Signal-to-Noise Ratio.
    
    Args:
        audio: Input audio
        noise_floor_db: Estimated noise floor in dB
        
    Returns:
        Estimated SNR in dB
    """
    signal_power = np.mean(audio ** 2)
    noise_power = 10 ** (noise_floor_db / 10)
    
    if noise_power == 0:
        return float('inf')
    
    snr = 10 * np.log10(signal_power / noise_power)
    return max(snr, 0)  # Ensure non-negative


def estimate_spectral_centroid(audio: np.ndarray, sample_rate: int) -> float:
    """
    Estimate spectral centroid (brightness) of audio.
    
    Args:
        audio: Input audio
        sample_rate: Sample rate
        
    Returns:
        Spectral centroid in Hz
    """
    stft = np.abs(librosa.stft(audio))
    centroid = librosa.feature.spectral_centroid(S=stft, sr=sample_rate)
    return float(np.mean(centroid))


def estimate_frequency_range(audio: np.ndarray, sample_rate: int) -> Tuple[float, float]:
    """
    Estimate frequency range present in audio.
    
    Args:
        audio: Input audio
        sample_rate: Sample rate
        
    Returns:
        Tuple of (low_freq, high_freq) in Hz
    """
    stft = np.abs(librosa.stft(audio))
    
    # Find frequencies with significant energy
    energy = np.mean(stft, axis=1)
    total_energy = np.sum(energy)
    
    # Find low frequency cutoff (where 1% of energy starts)
    cumsum = np.cumsum(energy)
    low_idx = np.searchsorted(cumsum, 0.01 * total_energy)
    
    # Find high frequency cutoff (where 99% of energy ends)
    high_idx = np.searchsorted(cumsum, 0.99 * total_energy)
    
    freqs = librosa.fft_frequencies(sr=sample_rate)
    
    low_freq = float(freqs[low_idx]) if low_idx < len(freqs) else 0.0
    high_freq = float(freqs[high_idx]) if high_idx < len(freqs) else sample_rate / 2
    
    return low_freq, high_freq


def detect_clipping(audio: np.ndarray, threshold: float = 0.95) -> Dict[str, Any]:
    """
    Detect clipping in audio signal.
    
    Args:
        audio: Input audio
        threshold: Clipping threshold (0-1)
        
    Returns:
        Dictionary with clipping analysis
    """
    max_val = np.max(np.abs(audio))
    clipped_samples = np.sum(np.abs(audio) > threshold)
    total_samples = len(audio)
    
    return {
        "max_amplitude": float(max_val),
        "clipped_samples": int(clipped_samples),
        "clipping_percent": float(clipped_samples / total_samples * 100),
        "is_clipped": clipped_samples > 0
    }


# === Tensor Conversions ===

def numpy_to_tensor(audio: np.ndarray, device: torch.device = None) -> torch.Tensor:
    """
    Convert numpy array to PyTorch tensor.
    
    Args:
        audio: Audio numpy array
        device: Target device (defaults to CPU)
        
    Returns:
        PyTorch tensor
    """
    if device is None:
        device = torch.device('cpu')
    
    # Ensure float32
    tensor = torch.from_numpy(audio.astype(np.float32))
    return tensor.to(device)


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert PyTorch tensor to numpy array.
    
    Args:
        tensor: PyTorch tensor
        
    Returns:
        Numpy array (CPU, float32)
    """
    if tensor.is_cuda:
        tensor = tensor.cpu()
    if tensor.requires_grad:
        tensor = tensor.detach()
    return tensor.numpy().astype(np.float32)


def stereo_to_mono(audio: np.ndarray) -> np.ndarray:
    """
    Convert stereo audio to mono by averaging channels.
    
    Args:
        audio: Audio array (can be mono or multi-channel)
        
    Returns:
        Mono audio array
    """
    if audio.ndim == 1:
        return audio
    
    # Average all channels
    return np.mean(audio, axis=1)


def ensure_mono(audio: np.ndarray) -> np.ndarray:
    """
    Ensure audio is mono (1D array).
    
    Args:
        audio: Audio array
        
    Returns:
        Mono audio array
    """
    if audio.ndim == 1:
        return audio
    elif audio.ndim == 2:
        # If stereo, mix down to mono
        return stereo_to_mono(audio)
    else:
        raise ValueError(f"Unexpected audio shape: {audio.shape}")


# === File Operations ===

def get_audio_hash(audio: np.ndarray, sample_rate: int) -> str:
    """
    Generate hash for audio content.
    
    Args:
        audio: Audio array
        sample_rate: Sample rate
        
    Returns:
        MD5 hash string
    """
    import hashlib
    
    # Take first 10 seconds for hashing (or full audio if shorter)
    max_samples = min(len(audio), sample_rate * 10)
    audio_subset = audio[:max_samples]
    
    audio_bytes = audio_subset.tobytes()
    hash_value = hashlib.md5(audio_bytes).hexdigest()
    
    return hash_value


def get_file_hash(file_path: str) -> str:
    """
    Generate hash for audio file.
    
    Args:
        file_path: Path to audio file
        
    Returns:
        MD5 hash string
    """
    import hashlib
    
    with open(file_path, 'rb') as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    
    return file_hash


# === Validation ===

def validate_audio_params(sample_rate: int, channels: int, duration: float) -> Tuple[bool, str]:
    """
    Validate audio parameters.
    
    Args:
        sample_rate: Sample rate in Hz
        channels: Number of audio channels
        duration: Duration in seconds
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check sample rate
    if sample_rate < 8000:
        return False, f"Sample rate {sample_rate} Hz is too low (minimum 8000 Hz)"
    if sample_rate > 192000:
        return False, f"Sample rate {sample_rate} Hz is too high (maximum 192000 Hz)"
    
    # Check channels
    if channels < 1:
        return False, f"Invalid channel count: {channels}"
    if channels > 8:
        return False, f"Unsupported channel count: {channels} (maximum 8)"
    
    # Check duration
    if duration < 0.1:
        return False, f"Audio duration {duration:.2f}s is too short (minimum 0.1s)"
    if duration > 600:  # 10 minutes
        return False, f"Audio duration {duration:.1f}s exceeds maximum (10 minutes)"
    
    return True, "Valid"


# === Display Helpers ===

def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted string (HH:MM:SS or MM:SS)
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted string (KB, MB, GB)
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


# Example usage
if __name__ == "__main__":
    # Test utility functions
    print("Testing utility functions...")
    
    # DB conversions
    print(f"0 dB = {db_to_gain(0)} gain")
    print(f"1.0 gain = {gain_to_db(1.0)} dB")
    
    # Audio generation for testing
    import soundfile as sf
    
    # Generate 1 second test tone
    sample_rate = 44100
    t = np.linspace(0, 1, sample_rate)
    test_audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    
    print(f"Test audio shape: {test_audio.shape}")
    print(f"Peak level: {np.max(np.abs(test_audio)):.3f}")
    
    # Estimate SNR
    snr = estimate_snr(test_audio)
    print(f"Estimated SNR: {snr:.1f} dB")
    
    # Detect clipping
    clipping = detect_clipping(test_audio)
    print(f"Clipping detected: {clipping['is_clipped']}")
