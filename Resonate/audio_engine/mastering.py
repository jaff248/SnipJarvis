"""
Mastering Module - Final polish and export for reconstructed audio

Handles the mastering stage of the reconstruction pipeline, applying
loudness normalization, limiting, dithering, and format export.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from enum import Enum
from pathlib import Path

import numpy as np
import soundfile as sf

from .utils import db_to_gain, format_duration

logger = logging.getLogger(__name__)


class OutputFormat(Enum):
    """Supported output audio formats."""
    WAV = "wav"      # Uncompressed, highest quality
    FLAC = "flac"    # Compressed lossless
    MP3 = "mp3"      # Compressed lossy (requires lame)


@dataclass
class MasteringConfig:
    """Configuration for audio mastering."""
    # Target loudness
    target_loudness_lufs: float = -14.0  # Spotify/streaming standard
    loudness_range_max_lufs: float = 5.0  # Max loudness range
    
    # Peak limiting
    true_peak_limit_db: float = -1.0     # Hard limit for streaming
    limiter_threshold_db: float = -2.0   # Soft limit threshold
    limiter_release_ms: float = 20.0
    
    # EQ
    master_eq_low_db: float = 0.0        # Low shelf adjustment
    master_eq_mid_db: float = 0.0        # Mid boost/cut
    master_eq_high_db: float = 0.0       # High shelf adjustment
    
    # Output settings
    output_format: OutputFormat = OutputFormat.WAV
    output_bit_depth: int = 24           # 16, 24, or 32 (float)
    
    # Dithering
    apply_dither: bool = True
    dither_type: str = "shibata"         # "rectangular", "triangular", "shibata"
    
    # Stereo processing
    stereo_width: float = 1.0            # 0 = mono, 1 = original, >1 = widened
    
    # Smoothing
    apply_smoothing: bool = True
    smoothing_window_ms: float = 5.0


@dataclass
class MasteringResult:
    """Result of audio mastering."""
    audio: np.ndarray
    sample_rate: int
    format: OutputFormat
    bit_depth: int
    file_path: str
    file_size_bytes: int
    loudness_lufs: float
    true_peak_db: float
    dynamic_range_db: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "audio_shape": self.audio.shape,
            "sample_rate": self.sample_rate,
            "duration_seconds": len(self.audio) / self.sample_rate,
            "duration_formatted": format_duration(len(self.audio) / self.sample_rate),
            "format": self.format.value,
            "bit_depth": self.bit_depth,
            "file_size_mb": self.file_size_bytes / (1024 * 1024),
            "loudness_lufs": round(self.loudness_lufs, 1),
            "true_peak_db": round(self.true_peak_db, 1),
            "dynamic_range_db": round(self.dynamic_range_db, 1)
        }


class AudioMaster:
    """
    Audio mastering engine for final polish.
    
    Handles:
    - Loudness normalization (LUFS)
    - True peak limiting
    - Master EQ adjustments
    - Stereo width processing
    - Dithering for bit-depth reduction
    - Export to various formats
    """
    
    def __init__(self, config: MasteringConfig = None):
        """
        Initialize mastering engine.
        
        Args:
            config: Mastering configuration (uses defaults if None)
        """
        self.config = config or MasteringConfig()
        
        logger.info(f"Initialized AudioMaster: loudness={self.config.target_loudness_lufs} LUFS, "
                   f"format={self.config.output_format.value}")
    
    def master(self, audio: np.ndarray, sample_rate: int,
              output_path: str = None) -> MasteringResult:
        """
        Apply full mastering chain to audio.
        
        Pipeline:
        1. Master EQ (subtle adjustments)
        2. Stereo width processing
        3. Loudness normalization (LUFS)
        4. True peak limiting
        5. Smoothing (optional)
        6. Export to file
        
        Args:
            audio: Input audio (float32, range [-1, 1])
            sample_rate: Sample rate
            output_path: Path for output file (auto-generated if None)
            
        Returns:
            MasteringResult with mastered audio and metadata
        """
        import time
        start_time = time.time()
        
        logger.info("Starting mastering...")
        
        processed = audio.copy()
        
        # 1. Master EQ
        if (self.config.master_eq_low_db != 0 or 
            self.config.master_eq_mid_db != 0 or 
            self.config.master_eq_high_db != 0):
            processed = self._apply_master_eq(processed, sample_rate)
            logger.debug("Applied master EQ")
        
        # 2. Stereo width
        if self.config.stereo_width != 1.0:
            processed = self._adjust_stereo_width(processed, self.config.stereo_width)
            logger.debug(f"Adjusted stereo width to {self.config.stereo_width}")
        
        # 3. Loudness normalization
        loudness = self._measure_loudness(processed, sample_rate)
        if abs(loudness - self.config.target_loudness_lufs) > 0.1:
            processed = self._normalize_loudness(processed, sample_rate,
                                                 self.config.target_loudness_lufs)
            logger.debug(f"Normalized loudness to {self.config.target_loudness_lufs} LUFS")
        
        # 4. True peak limiting
        if self.config.true_peak_limit_db > -1.0:
            processed = self._apply_peak_limit(processed, sample_rate)
            logger.debug("Applied peak limiting")
        
        # 5. Smoothing
        if self.config.apply_smoothing:
            processed = self._apply_smoothing(processed, sample_rate)
            logger.debug("Applied smoothing")
        
        # Measure final metrics
        final_loudness = self._measure_loudness(processed, sample_rate)
        true_peak = self._measure_true_peak(processed)
        dynamic_range = self._measure_dynamic_range(processed, sample_rate)
        
        # 6. Export
        if output_path is None:
            output_path = self._generate_output_path()
        
        file_size = self._export(processed, sample_rate, output_path)
        
        processing_time = time.time() - start_time
        logger.info(f"✅ Mastering complete in {processing_time:.1f}s")
        
        return MasteringResult(
            audio=processed,
            sample_rate=sample_rate,
            format=self.config.output_format,
            bit_depth=self.config.output_bit_depth,
            file_path=output_path,
            file_size_bytes=file_size,
            loudness_lufs=final_loudness,
            true_peak_db=true_peak,
            dynamic_range_db=dynamic_range
        )
    
    def _apply_master_eq(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply master EQ adjustments."""
        from scipy import signal
        
        processed = audio.copy()
        
        nyquist = sample_rate / 2
        
        # Low shelf
        if self.config.master_eq_low_db != 0:
            low_freq = 200
            normalized = min(low_freq / nyquist, 0.99)
            # Simple approach: boost/cut based on frequency content
            b, a = signal.butter(2, normalized, btype='low')
            low_content = signal.filtfilt(b, a, processed)
            gain = db_to_gain(self.config.master_eq_low_db)
            processed = processed + (low_content * (gain - 1) * 0.5)
        
        # High shelf
        if self.config.master_eq_high_db != 0:
            high_freq = 8000
            normalized = min(high_freq / nyquist, 0.99)
            b, a = signal.butter(2, normalized, btype='high')
            high_content = signal.filtfilt(b, a, processed)
            gain = db_to_gain(self.config.master_eq_high_db)
            processed = processed + (high_content * (gain - 1) * 0.3)
        
        return processed
    
    def _adjust_stereo_width(self, audio: np.ndarray, width: float) -> np.ndarray:
        """
        Adjust stereo width.
        
        For mono input, this has no effect.
        For stereo input, applies width adjustment.
        """
        if audio.ndim == 1:
            return audio
        
        # Simple mid-side processing for width adjustment
        left = audio[:, 0]
        right = audio[:, 1]
        
        mid = (left + right) / 2
        side = (left - right) / 2
        
        # Adjust side component
        side = side * width
        
        # Recombine
        left_new = mid + side
        right_new = mid - side
        
        return np.column_stack([left_new, right_new])
    
    def _measure_loudness(self, audio: np.ndarray, sample_rate: int) -> float:
        """
        Measure integrated loudness in LUFS.
        
        Uses simple approximation (integrated LUFS calculation).
        """
        try:
            import pyloudnorm as pyln
            
            meter = pyln.Meter(sample_rate)
            return meter.integrated_loudness(audio)
            
        except ImportError:
            # Fallback: use RMS-based approximation
            rms = np.sqrt(np.mean(audio ** 2))
            # Convert RMS to approximate LUFS
            # This is a rough approximation - real LUFS uses filters
            lufs = 20 * np.log10(rms + 1e-10) - 0.691
            return float(lufs)
    
    def _normalize_loudness(self, audio: np.ndarray, sample_rate: int,
                           target_lufs: float) -> np.ndarray:
        """
        Normalize audio to target loudness.
        
        Args:
            audio: Input audio
            sample_rate: Sample rate
            target_lufs: Target loudness in LUFS
            
        Returns:
            Loudness-normalized audio
        """
        try:
            import pyloudnorm as pyln
            
            meter = pyln.Meter(sample_rate)
            current_lufs = meter.integrated_loudness(audio)
            
            # Calculate gain needed
            gain_db = target_lufs - current_lufs
            gain_linear = db_to_gain(gain_db)
            
            normalized = audio * gain_linear
            
            # Prevent clipping
            normalized = np.clip(normalized, -1.0, 1.0)
            
            return normalized
            
        except ImportError:
            # Fallback: RMS-based normalization
            rms = np.sqrt(np.mean(audio ** 2))
            target_rms = 10 ** ((target_lufs + 0.691) / 20)
            
            if rms > 0:
                gain = target_rms / rms
                normalized = audio * gain
                normalized = np.clip(normalized, -1.0, 1.0)
                return normalized
            
            return audio
    
    def _apply_peak_limit(self, audio: np.ndarray, 
                         sample_rate: int) -> np.ndarray:
        """
        Apply true peak limiting to prevent clipping.
        
        Uses look-ahead limiting with soft clipping.
        """
        from scipy import signal
        
        limit_db = self.config.true_peak_limit_db
        threshold_db = self.config.limiter_threshold_db
        release_ms = self.config.limiter_release_ms
        
        limit = db_to_gain(limit_db)
        threshold = db_to_gain(threshold_db)
        release_samples = int(release_ms * sample_rate / 1000)
        
        # Calculate envelope
        envelope = np.abs(audio)
        
        # Look-ahead peak detection
        lookahead = 100  # samples
        
        # Extend envelope for lookahead
        envelope_extended = np.concatenate([
            envelope[:lookahead] * 0 + np.max(envelope[:lookahead]),
            envelope
        ])
        
        # Envelope follower with fast attack, controlled release
        alpha_attack = 0.95  # Very fast attack
        alpha_release = 1 - np.exp(-1 / release_samples)
        
        envelope_followed = np.zeros_like(envelope_extended)
        for i in range(len(envelope_extended)):
            if i == 0:
                envelope_followed[i] = envelope_extended[i]
            else:
                if envelope_extended[i] > envelope_followed[i-1]:
                    envelope_followed[i] = (alpha_attack * envelope_extended[i] + 
                                           (1 - alpha_attack) * envelope_followed[i-1])
                else:
                    envelope_followed[i] = (alpha_release * envelope_extended[i] + 
                                           (1 - alpha_release) * envelope_followed[i-1])
        
        # Calculate gain reduction
        envelope_followed = envelope_followed[lookahead:]  # Remove lookahead
        
        gain_reduction = np.ones_like(audio)
        above_threshold = envelope_followed > threshold
        
        if np.any(above_threshold):
            overs = envelope_followed[above_threshold] - threshold
            target_gain = threshold / envelope_followed[above_threshold]
            # Soft limiting curve
            reduction = 1 - (1 - target_gain) * np.power(overs / threshold, 0.5)
            reduction = np.clip(reduction, 0, 1)
            gain_reduction[above_threshold] = reduction
        
        # Apply limiting
        limited = audio * gain_reduction
        
        # Final hard clip at limit
        limited = np.clip(limited, -limit, limit)
        
        # Makeup gain to recover lost loudness
        makeup_db = -limit_db + 1  # 1 dB makeup
        limited *= db_to_gain(makeup_db)
        
        # Final clip
        limited = np.clip(limited, -limit, limit)
        
        return limited
    
    def _apply_smoothing(self, audio: np.ndarray, 
                        sample_rate: int) -> np.ndarray:
        """
        Apply light smoothing to reduce harshness.
        
        Uses very gentle low-pass filtering.
        """
        from scipy import signal
        
        # Very gentle smoothing
        cutoff = 20000  # Hz
        nyquist = sample_rate / 2
        normalized = min(cutoff / nyquist, 0.99)
        
        b, a = signal.butter(2, normalized, btype='low')
        smoothed = signal.filtfilt(b, a, audio)
        
        # Blend: mostly original, little smoothing
        return audio * 0.8 + smoothed * 0.2
    
    def _measure_true_peak(self, audio: np.ndarray) -> float:
        """Measure true peak level in dB."""
        if audio.ndim == 1:
            peak = np.max(np.abs(audio))
        else:
            # Multi-channel: max across channels
            peak = np.max(np.abs(audio))
        
        return 20 * np.log10(peak + 1e-10)
    
    def _measure_dynamic_range(self, audio: np.ndarray, 
                              sample_rate: int) -> float:
        """Measure dynamic range in dB."""
        # Use ITU-R BS.1770 method (simplified)
        # RMS over sliding windows
        window_size = sample_rate  # 1 second windows
        
        rms_values = []
        for i in range(0, len(audio), window_size // 4):
            window = audio[i:i+window_size]
            if len(window) > 0:
                rms = np.sqrt(np.mean(window ** 2))
                if rms > 0:
                    rms_db = 20 * np.log10(rms + 1e-10)
                    rms_values.append(rms_db)
        
        if len(rms_values) >= 2:
            return max(rms_values) - min(rms_values)
        return 0
    
    def _generate_output_path(self) -> str:
        """Generate output file path."""
        import datetime
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = self.config.output_format.value
        
        return f"resonate_mastered_{timestamp}.{ext}"
    
    def _export(self, audio: np.ndarray, sample_rate: int, 
               output_path: str) -> int:
        """
        Export audio to file.
        
        Args:
            audio: Audio data
            sample_rate: Sample rate
            output_path: Output file path
            
        Returns:
            File size in bytes
        """
        output_path = Path(output_path)
        
        # Ensure correct file extension
        expected_ext = f".{self.config.output_format.value}"
        if output_path.suffix.lower() != expected_ext:
            output_path = output_path.with_suffix(expected_ext)
        
        # Determine subtype based on bit depth
        if self.config.output_format == OutputFormat.WAV:
            if self.config.output_bit_depth == 16:
                subtype = 'PCM_16'
            elif self.config.output_bit_depth == 24:
                subtype = 'PCM_24'
            else:  # 32
                subtype = 'FLOAT'
        elif self.config.output_format == OutputFormat.FLAC:
            subtype = 'PCM_24'
        else:
            subtype = 'MP3'
        
        # Export
        sf.write(str(output_path), audio, sample_rate, subtype=subtype)
        
        # Return file size
        return output_path.stat().st_size
    
    def get_info(self) -> Dict[str, Any]:
        """Get mastering configuration and info."""
        return {
            "config": {
                "target_loudness_lufs": self.config.target_loudness_lufs,
                "true_peak_limit_db": self.config.true_peak_limit_db,
                "master_eq_low_db": self.config.master_eq_low_db,
                "master_eq_mid_db": self.config.master_eq_mid_db,
                "master_eq_high_db": self.config.master_eq_high_db,
                "stereo_width": self.config.stereo_width,
                "output_format": self.config.output_format.value,
                "output_bit_depth": self.config.output_bit_depth
            }
        }
    
    def __repr__(self) -> str:
        return (f"AudioMaster(loudness={self.config.target_loudness_lufs} LUFS, "
                f"peak={self.config.true_peak_limit_db} dB, "
                f"format={self.config.output_format.value})")


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing AudioMaster...")
    
    # Create test audio
    sample_rate = 44100
    duration = 3.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create audio with varying levels
    audio = (
        0.5 * np.sin(2 * np.pi * 440 * t) +  # Normal level
        0.1 * np.sin(2 * np.pi * 880 * t) +  # Quiet
        0.8 * np.sin(2 * np.pi * 220 * t) * np.mod(t, 1)  # Louder sections
    )
    
    # Add some noise
    audio += 0.01 * np.random.randn(len(audio))
    
    print(f"Input audio: {audio.shape}, range=[{audio.min():.3f}, {audio.max():.3f}]")
    
    # Create master
    master = AudioMaster()
    print(f"Created: {master}")
    
    # Master audio
    result = master.master(audio, sample_rate, "test_master.wav")
    print(f"✅ Mastering complete: {result.to_dict()}")
    
    print(f"Saved to: {result.file_path}")
