"""
Audio Ingest Module - Load, validate, and preprocess audio files

Handles audio file loading, format validation, quality analysis,
and initial preprocessing for the reconstruction pipeline.
"""

import os
import logging
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

import numpy as np
import soundfile as sf
import librosa
from scipy import signal

from .utils import (
    normalize_peak, ensure_mono, validate_audio_params,
    estimate_snr, estimate_spectral_centroid, estimate_frequency_range,
    detect_clipping, get_audio_hash, get_file_hash, format_duration
)
from .device import get_device

logger = logging.getLogger(__name__)


@dataclass
class AudioMetadata:
    """Metadata extracted from audio file."""
    sample_rate: int
    channels: int
    duration: float
    bit_depth: Optional[int]
    format: str
    file_size: int
    file_hash: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "duration": round(self.duration, 3),
            "bit_depth": self.bit_depth,
            "format": self.format,
            "file_size": self.file_size,
            "file_hash": self.file_hash,
            "duration_formatted": format_duration(self.duration)
        }


@dataclass
class AudioAnalysis:
    """Analysis results for audio quality assessment."""
    snr_db: float
    spectral_centroid_hz: float
    frequency_range_hz: Tuple[float, float]
    is_clipped: bool
    clipping_percent: float
    dynamic_range_db: float
    loudness_lufs: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "snr_db": round(self.snr_db, 1),
            "spectral_centroid_hz": round(self.spectral_centroid_hz, 0),
            "frequency_range_hz": [round(self.frequency_range_hz[0], 0), 
                                   round(self.frequency_range_hz[1], 0)],
            "is_clipped": self.is_clipped,
            "clipping_percent": round(self.clipping_percent, 3),
            "dynamic_range_db": round(self.dynamic_range_db, 1),
            "loudness_lufs": round(self.loudness_lufs, 1)
        }


class AudioBuffer:
    """
    In-memory audio buffer with metadata.
    
    Represents loaded audio data ready for processing.
    """
    
    def __init__(self, audio: np.ndarray, sample_rate: int, 
                 metadata: AudioMetadata, analysis: AudioAnalysis = None):
        """
        Initialize audio buffer.
        
        Args:
            audio: Audio data as numpy array (float32, range [-1, 1])
            sample_rate: Sample rate in Hz
            metadata: Audio file metadata
            analysis: Optional audio analysis results
        """
        self.audio = audio
        self.sample_rate = sample_rate
        self.metadata = metadata
        self.analysis = analysis
        
        # Validate audio shape
        if audio.ndim == 1:
            self.channels = 1
        elif audio.ndim == 2:
            self.channels = audio.shape[1]
        else:
            raise ValueError(f"Invalid audio shape: {audio.shape}")
        
        # Ensure float32
        if audio.dtype != np.float32:
            self.audio = audio.astype(np.float32)
    
    @property
    def duration(self) -> float:
        """Get duration in seconds."""
        return len(self.audio) / self.sample_rate
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Get audio shape."""
        return self.audio.shape
    
    def __repr__(self) -> str:
        return (f"AudioBuffer(shape={self.shape}, sr={self.sample_rate}, "
                f"duration={format_duration(self.duration)})")
    
    def copy(self) -> 'AudioBuffer':
        """Create a copy of the audio buffer."""
        return AudioBuffer(
            audio=self.audio.copy(),
            sample_rate=self.sample_rate,
            metadata=self.metadata,
            analysis=self.analysis
        )


class AudioIngest:
    """
    Audio ingestion engine for loading and preprocessing audio.
    
    Handles:
    - File loading (WAV, MP3, FLAC, OGG)
    - Format validation
    - Quality analysis (SNR, frequency range, clipping)
    - Preprocessing (normalization, mono conversion)
    - Temporary file management
    """
    
    # Supported audio formats
    SUPPORTED_FORMATS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aiff'}
    
    # Preferred sample rates for processing
    PREFERRED_SAMPLE_RATES = [44100, 48000]
    
    # Maximum file size (100 MB)
    MAX_FILE_SIZE = 100 * 1024 * 1024
    
    def __init__(self, target_sample_rate: int = 44100, 
                 normalize: bool = True,
                 convert_to_mono: bool = True):
        """
        Initialize audio ingest.
        
        Args:
            target_sample_rate: Target sample rate for processing
                               (will resample if different)
            normalize: Whether to normalize audio to -1 dB peak
            convert_to_mono: Whether to convert stereo to mono
        """
        self.target_sample_rate = target_sample_rate
        self.normalize = normalize
        self.convert_to_mono = convert_to_mono
        
        # Create temp directory for processing
        self.temp_dir = tempfile.mkdtemp(prefix="resonate_")
        logger.info(f"Created temp directory: {self.temp_dir}")
    
    def load(self, file_path: str, analyze: bool = True) -> AudioBuffer:
        """
        Load audio file and prepare for processing.
        
        Args:
            file_path: Path to audio file
            analyze: Whether to perform quality analysis
            
        Returns:
            AudioBuffer with loaded audio and metadata
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is unsupported or invalid
            RuntimeError: If audio loading fails
        """
        file_path = Path(file_path)
        
        # Validate file exists
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        # Validate file size
        file_size = file_path.stat().st_size
        if file_size > self.MAX_FILE_SIZE:
            raise ValueError(f"File too large: {format_file_size(file_size)} "
                           f"(maximum: {format_file_size(self.MAX_FILE_SIZE)})")
        
        # Validate file extension
        if file_path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {file_path.suffix}. "
                           f"Supported: {', '.join(self.SUPPORTED_FORMATS)}")
        
        logger.info(f"Loading audio: {file_path.name}")
        
        try:
            # Load audio with soundfile (preserves native format)
            audio, sr = sf.read(str(file_path))
            
            logger.info(f"  Loaded: {audio.shape}, {sr} Hz")
            
            # Get file hash
            file_hash = get_file_hash(str(file_path))
            
            # Get format info
            format_info = sf.info(str(file_path))
            
            # Create metadata
            duration = len(audio) / sr
            metadata = AudioMetadata(
                sample_rate=sr,
                channels=audio.shape[1] if audio.ndim > 1 else 1,
                duration=duration,
                bit_depth=getattr(format_info, 'subtype_info', None),
                format=file_path.suffix.lower(),
                file_size=file_size,
                file_hash=file_hash
            )
            
            # Validate parameters
            is_valid, error_msg = validate_audio_params(
                sr, 
                metadata.channels, 
                duration
            )
            if not is_valid:
                raise ValueError(f"Audio validation failed: {error_msg}")
            
            # Preprocessing pipeline
            audio = self._preprocess(audio, sr)
            
            # Update metadata after preprocessing
            metadata = AudioMetadata(
                sample_rate=self.target_sample_rate,
                channels=1 if self.convert_to_mono else metadata.channels,
                duration=len(audio) / self.target_sample_rate,
                bit_depth='float32',
                format=metadata.format,
                file_size=file_size,
                file_hash=file_hash
            )
            
            # Analysis (optional)
            analysis = None
            if analyze:
                analysis = self._analyze(audio, self.target_sample_rate)
                logger.info(f"  Analysis: SNR={analysis.snr_db:.1f} dB, "
                          f"Centroid={analysis.spectral_centroid_hz:.0f} Hz")
            
            return AudioBuffer(
                audio=audio,
                sample_rate=self.target_sample_rate,
                metadata=metadata,
                analysis=analysis
            )
            
        except Exception as e:
            logger.error(f"Failed to load audio: {e}")
            raise RuntimeError(f"Audio loading failed: {e}") from e
    
    def _preprocess(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Apply preprocessing pipeline to audio.
        
        Pipeline:
        1. Convert to mono (if configured)
        2. Resample to target sample rate
        3. Normalize peak level
        4. Validate range [-1, 1]
        
        Args:
            audio: Input audio
            sample_rate: Original sample rate
            
        Returns:
            Preprocessed audio
        """
        # Step 1: Convert to mono
        if self.convert_to_mono and audio.ndim > 1:
            audio = ensure_mono(audio)
            logger.debug("  Converted to mono")
        
        # Step 2: Resample if needed
        if sample_rate != self.target_sample_rate:
            audio = librosa.resample(audio, orig_sr=sample_rate, 
                                    target_sr=self.target_sample_rate)
            logger.debug(f"  Resampled: {sample_rate} Hz -> {self.target_sample_rate} Hz")
        
        # Step 3: Normalize
        if self.normalize:
            audio = normalize_peak(audio, target_db=-1.0)
            logger.debug("  Normalized to -1 dB peak")
        
        # Step 4: Validate range
        if np.max(np.abs(audio)) > 1.0:
            audio = np.clip(audio, -1.0, 1.0)
            logger.warning("  Clipped audio to [-1, 1] range")
        
        return audio
    
    def _analyze(self, audio: np.ndarray, sample_rate: int) -> AudioAnalysis:
        """
        Analyze audio quality.
        
        Args:
            audio: Audio to analyze
            sample_rate: Sample rate
            
        Returns:
            AudioAnalysis with quality metrics
        """
        # SNR estimation
        snr_db = estimate_snr(audio)
        
        # Spectral centroid
        spectral_centroid_hz = estimate_spectral_centroid(audio, sample_rate)
        
        # Frequency range
        freq_range = estimate_frequency_range(audio, sample_rate)
        
        # Clipping detection
        clipping = detect_clipping(audio)
        
        # Dynamic range
        peaks = []
        window_size = sample_rate  # 1 second windows
        for i in range(0, len(audio), window_size):
            window = audio[i:i+window_size]
            if len(window) > 0:
                peaks.append(np.max(np.abs(window)))
        
        if peaks:
            rms_level = np.sqrt(np.mean(audio ** 2))
            peak_db = 20 * np.log10(max(peaks) + 1e-10)
            rms_db = 20 * np.log10(rms_level + 1e-10)
            dynamic_range_db = max(0, peak_db - rms_db)
        else:
            dynamic_range_db = 0
        
        # Loudness (approximate)
        loudness_lufs = -24.0  # Placeholder - would use pyloudnorm for accuracy
        
        return AudioAnalysis(
            snr_db=snr_db,
            spectral_centroid_hz=spectral_centroid_hz,
            frequency_range_hz=freq_range,
            is_clipped=clipping['is_clipped'],
            clipping_percent=clipping['clipping_percent'],
            dynamic_range_db=dynamic_range_db,
            loudness_lufs=loudness_lufs
        )
    
    def create_test_tone(self, duration: float = 1.0, 
                         frequency: float = 440.0,
                         sample_rate: int = None) -> AudioBuffer:
        """
        Create test tone for debugging and validation.
        
        Args:
            duration: Duration in seconds
            frequency: Tone frequency in Hz
            sample_rate: Sample rate (uses target if None)
            
        Returns:
            AudioBuffer with test tone
        """
        sr = sample_rate or self.target_sample_rate
        
        t = np.linspace(0, duration, int(sr * duration))
        audio = 0.5 * np.sin(2 * np.pi * frequency * t)
        
        metadata = AudioMetadata(
            sample_rate=sr,
            channels=1,
            duration=duration,
            bit_depth='float32',
            format='.wav',
            file_size=int(duration * sr * 4),
            file_hash='test_tone'
        )
        
        return AudioBuffer(audio=audio, sample_rate=sr, metadata=metadata)
    
    def cleanup(self):
        """Clean up temporary files."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            logger.info(f"Cleaned up temp directory: {self.temp_dir}")
    
    def __del__(self):
        """Cleanup on destruction."""
        self.cleanup()


def format_file_size(size_bytes: int) -> str:
    """Format file size for display."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test ingest
    ingest = AudioIngest(target_sample_rate=44100, normalize=True)
    
    # Create test tone
    buffer = ingest.create_test_tone(duration=2.0, frequency=440)
    print(f"Created: {buffer}")
    print(f"Metadata: {buffer.metadata.to_dict()}")
    
    if buffer.analysis:
        print(f"Analysis: {buffer.analysis.to_dict()}")
    
    # Cleanup
    ingest.cleanup()
