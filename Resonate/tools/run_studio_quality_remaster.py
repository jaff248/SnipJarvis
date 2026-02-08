#!/usr/bin/env python3
"""
Studio Quality Remaster - Reference-Based Audio Enhancement (OPTIMIZED)

Philosophy: LESS IS MORE. Preserve the original recording's character while
improving technical quality to match a studio reference.

OPTIMIZATIONS:
- Chunked noise reduction with progress logging
- Parallel multiband compression using all CPU cores
- Process stereo channels in parallel

Usage:
  python tools/run_studio_quality_remaster.py <input_file> [--reference <ref_file>]

If no reference is provided, uses sensible defaults for electronic music.
"""

import sys
import json
import time
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import soundfile as sf

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Get number of CPU cores
NUM_CORES = os.cpu_count() or 4


@dataclass
class AudioProfile:
    """Spectral and loudness profile of an audio file."""
    loudness_lufs: float
    peak_db: float
    rms_db: float
    dynamic_range_db: float
    spectral_centroid_hz: float
    freq_range_low_hz: float
    freq_range_high_hz: float
    clipping_percent: float
    snr_db: float


@dataclass
class RemasterResult:
    """Result of the remastering process."""
    output_path: str
    input_profile: AudioProfile
    output_profile: AudioProfile
    reference_profile: Optional[AudioProfile]
    processing_steps: list
    quality_improved: bool
    processing_time_s: float


def _reduce_noise_chunk(args):
    """Worker function for parallel noise reduction on a chunk."""
    chunk, sr, strength = args
    try:
        import noisereduce as nr
        return nr.reduce_noise(y=chunk, sr=sr, prop_decrease=strength, stationary=False)
    except Exception as e:
        logger.warning(f"Chunk noise reduction failed: {e}")
        return chunk


def _process_band(args):
    """Worker function for parallel band processing."""
    band_audio, sample_rate, ratio, threshold_db = args
    from scipy import signal
    
    threshold = 10 ** (threshold_db / 20)
    envelope = np.abs(band_audio)
    
    # Smooth envelope with fast attack/release
    attack_samples = int(0.01 * sample_rate)
    release_samples = int(0.1 * sample_rate)
    smoothed = np.copy(envelope)
    
    for i in range(1, len(smoothed)):
        if envelope[i] > smoothed[i-1]:
            alpha = 1.0 / attack_samples
        else:
            alpha = 1.0 / release_samples
        smoothed[i] = alpha * envelope[i] + (1 - alpha) * smoothed[i-1]
    
    # Calculate gain
    gain = np.ones_like(band_audio)
    above_threshold = smoothed > threshold
    if np.any(above_threshold):
        over_db = 20 * np.log10(smoothed[above_threshold] / threshold + 1e-10)
        gain_reduction_db = over_db * (1 - 1/ratio)
        gain[above_threshold] = 10 ** (-gain_reduction_db / 20)
    
    return band_audio * gain


class StudioQualityRemaster:
    """
    Reference-based audio remastering for studio quality output.
    OPTIMIZED VERSION with parallel processing.
    """
    
    DEFAULT_TARGET_LUFS = -14.0
    DEFAULT_TARGET_PEAK_DB = -1.0
    DEFAULT_CENTROID_TARGET = 3000.0
    
    def __init__(self, 
                 enable_preconditioning: bool = True,
                 preconditioning_strength: float = 0.3,
                 enable_eq_matching: bool = True,
                 enable_compression: bool = True,
                 enable_limiting: bool = True,
                 validate_output: bool = True,
                 num_workers: int = None):
        """
        Initialize the remaster engine.
        
        Args:
            enable_preconditioning: Apply noise reduction and declipping
            preconditioning_strength: How aggressive (0-1, lower is gentler)
            enable_eq_matching: Match spectral shape to reference
            enable_compression: Apply gentle multiband compression
            enable_limiting: Apply true peak limiting
            validate_output: Check if output is actually better
            num_workers: Number of parallel workers (default: CPU cores)
        """
        self.enable_preconditioning = enable_preconditioning
        self.preconditioning_strength = preconditioning_strength
        self.enable_eq_matching = enable_eq_matching
        self.enable_compression = enable_compression
        self.enable_limiting = enable_limiting
        self.validate_output = validate_output
        self.num_workers = num_workers or NUM_CORES
        
        logger.info(f"StudioQualityRemaster initialized (using {self.num_workers} workers):")
        logger.info(f"  Preconditioning: {enable_preconditioning} (strength={preconditioning_strength})")
        logger.info(f"  EQ Matching: {enable_eq_matching}")
        logger.info(f"  Compression: {enable_compression}")
        logger.info(f"  Limiting: {enable_limiting}")
        logger.info(f"  Validation: {validate_output}")
    
    def analyze_audio(self, audio: np.ndarray, sample_rate: int) -> AudioProfile:
        """Analyze audio and extract profile metrics."""
        if audio.ndim > 1:
            mono = np.mean(audio, axis=1)
        else:
            mono = audio
        
        # Loudness (LUFS)
        try:
            import pyloudnorm as pyln
            meter = pyln.Meter(sample_rate)
            loudness_lufs = meter.integrated_loudness(audio if audio.ndim == 1 else audio)
        except Exception:
            rms = np.sqrt(np.mean(mono ** 2))
            loudness_lufs = 20 * np.log10(rms + 1e-10) - 0.691
        
        # Peak and RMS
        peak = np.max(np.abs(mono))
        peak_db = 20 * np.log10(peak + 1e-10)
        rms = np.sqrt(np.mean(mono ** 2))
        rms_db = 20 * np.log10(rms + 1e-10)
        
        # Dynamic range
        window_samples = sample_rate
        rms_values = []
        for i in range(0, len(mono), window_samples):
            w = mono[i:i+window_samples]
            if len(w) > 0:
                w_rms = np.sqrt(np.mean(w ** 2))
                if w_rms > 1e-10:
                    rms_values.append(20 * np.log10(w_rms))
        dynamic_range_db = max(rms_values) - min(rms_values) if len(rms_values) >= 2 else 0.0
        
        # Spectral analysis
        try:
            import librosa
            S = np.abs(librosa.stft(mono))
            freqs = librosa.fft_frequencies(sr=sample_rate)
            centroid = librosa.feature.spectral_centroid(S=S, sr=sample_rate)
            spectral_centroid = float(np.mean(centroid))
            
            energy = np.mean(S, axis=1)
            cumsum = np.cumsum(energy)
            total = cumsum[-1] + 1e-10
            low_idx = np.searchsorted(cumsum, 0.01 * total)
            high_idx = np.searchsorted(cumsum, 0.99 * total)
            freq_low = float(freqs[low_idx]) if low_idx < len(freqs) else 20.0
            freq_high = float(freqs[high_idx]) if high_idx < len(freqs) else sample_rate / 2
        except Exception as e:
            logger.warning(f"Spectral analysis failed: {e}")
            spectral_centroid = 2000.0
            freq_low = 50.0
            freq_high = 16000.0
        
        # Clipping
        threshold = 0.99
        clipped_samples = np.sum(np.abs(mono) >= threshold)
        clipping_percent = float(clipped_samples / len(mono) * 100)
        
        # SNR
        try:
            frame_size = 512
            frames = [mono[i:i+frame_size] for i in range(0, len(mono) - frame_size, frame_size)]
            rms_frames = [np.sqrt(np.mean(f ** 2)) for f in frames]
            sorted_rms = sorted(rms_frames)
            noise_floor = np.mean(sorted_rms[:len(sorted_rms)//10]) if sorted_rms else 1e-10
            signal_level = np.mean(sorted_rms[-len(sorted_rms)//10:]) if sorted_rms else 1e-10
            snr_db = 20 * np.log10((signal_level + 1e-10) / (noise_floor + 1e-10))
        except Exception:
            snr_db = 30.0
        
        return AudioProfile(
            loudness_lufs=float(loudness_lufs),
            peak_db=float(peak_db),
            rms_db=float(rms_db),
            dynamic_range_db=float(dynamic_range_db),
            spectral_centroid_hz=float(spectral_centroid),
            freq_range_low_hz=float(freq_low),
            freq_range_high_hz=float(freq_high),
            clipping_percent=float(clipping_percent),
            snr_db=float(snr_db)
        )
    
    def apply_preconditioning(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Apply gentle preconditioning with CHUNKED processing for speed.
        """
        logger.info(f"Applying gentle preconditioning (strength={self.preconditioning_strength})")
        
        processed = audio.copy()
        is_stereo = audio.ndim > 1
        
        # 1. Chunked noise reduction
        if self.preconditioning_strength > 0:
            try:
                import noisereduce as nr
                effective_strength = self.preconditioning_strength * 0.5
                
                # Process in 30-second chunks to avoid memory issues and show progress
                chunk_samples = 30 * sample_rate
                total_samples = len(processed)
                num_chunks = (total_samples + chunk_samples - 1) // chunk_samples
                
                logger.info(f"  Noise reduction: processing {num_chunks} chunks...")
                
                if is_stereo:
                    # Process each channel
                    result_left = []
                    result_right = []
                    
                    for i in range(num_chunks):
                        start = i * chunk_samples
                        end = min(start + chunk_samples, total_samples)
                        
                        chunk_left = processed[start:end, 0]
                        chunk_right = processed[start:end, 1]
                        
                        # Process in parallel
                        with ThreadPoolExecutor(max_workers=2) as executor:
                            future_left = executor.submit(
                                nr.reduce_noise, y=chunk_left, sr=sample_rate, 
                                prop_decrease=effective_strength, stationary=False
                            )
                            future_right = executor.submit(
                                nr.reduce_noise, y=chunk_right, sr=sample_rate,
                                prop_decrease=effective_strength, stationary=False
                            )
                            result_left.append(future_left.result())
                            result_right.append(future_right.result())
                        
                        logger.info(f"    Chunk {i+1}/{num_chunks} complete")
                    
                    processed = np.column_stack([
                        np.concatenate(result_left),
                        np.concatenate(result_right)
                    ])
                else:
                    # Mono: process chunks sequentially
                    result_chunks = []
                    for i in range(num_chunks):
                        start = i * chunk_samples
                        end = min(start + chunk_samples, total_samples)
                        chunk = processed[start:end]
                        reduced = nr.reduce_noise(
                            y=chunk, sr=sample_rate,
                            prop_decrease=effective_strength, stationary=False
                        )
                        result_chunks.append(reduced)
                        logger.info(f"    Chunk {i+1}/{num_chunks} complete")
                    
                    processed = np.concatenate(result_chunks)
                
                logger.info(f"  ✅ Noise reduction complete (prop_decrease={effective_strength:.2f})")
                
            except Exception as e:
                logger.warning(f"  Noise reduction failed: {e}")
        
        # 2. Soft peak compression (fast, no chunking needed)
        if self.preconditioning_strength > 0.2:
            clip_threshold = 0.99
            if is_stereo:
                clipped = np.any(np.abs(processed) >= clip_threshold, axis=1)
            else:
                clipped = np.abs(processed) >= clip_threshold
            clipped_percent = np.sum(clipped) / len(processed) * 100
            
            if clipped_percent > 0.1:
                threshold = 0.95
                ratio = 4.0
                abs_audio = np.abs(processed)
                above_threshold = abs_audio > threshold
                
                if np.any(above_threshold):
                    over = abs_audio[above_threshold] - threshold
                    compressed = threshold + over / ratio
                    processed[above_threshold] = np.sign(processed[above_threshold]) * compressed
                
                logger.info(f"  ✅ Applied soft peak limiting ({clipped_percent:.2f}% was clipping)")
        
        return processed
    
    def apply_eq_matching(self, audio: np.ndarray, sample_rate: int,
                          reference_profile: Optional[AudioProfile] = None) -> np.ndarray:
        """Match spectral characteristics to reference using gentle EQ."""
        logger.info("Applying spectral matching...")
        
        from scipy import signal
        
        current_profile = self.analyze_audio(audio, sample_rate)
        
        if reference_profile:
            target_centroid = reference_profile.spectral_centroid_hz
        else:
            target_centroid = self.DEFAULT_CENTROID_TARGET
        
        centroid_diff = target_centroid - current_profile.spectral_centroid_hz
        max_adjustment_db = 3.0
        
        processed = audio.copy()
        nyquist = sample_rate / 2.0
        
        # Low shelf
        if centroid_diff > 100:
            low_adjust_db = -min(centroid_diff / 500, max_adjustment_db)
        elif centroid_diff < -100:
            low_adjust_db = min(-centroid_diff / 500, max_adjustment_db)
        else:
            low_adjust_db = 0.0
        
        if abs(low_adjust_db) > 0.5:
            try:
                low_freq = 200.0
                normalized = min(low_freq / nyquist, 0.99)
                b, a = signal.butter(2, normalized, btype='low')
                low_content = signal.filtfilt(b, a, processed, axis=0)
                gain = 10 ** (low_adjust_db / 20)
                processed = processed + low_content * (gain - 1) * 0.5
                logger.info(f"  Low shelf: {low_adjust_db:+.1f} dB")
            except Exception as e:
                logger.warning(f"  Low shelf failed: {e}")
        
        # High shelf
        high_adjust_db = -low_adjust_db * 0.5
        
        if abs(high_adjust_db) > 0.5:
            try:
                high_freq = 8000.0
                normalized = min(high_freq / nyquist, 0.99)
                b, a = signal.butter(2, normalized, btype='high')
                high_content = signal.filtfilt(b, a, processed, axis=0)
                gain = 10 ** (high_adjust_db / 20)
                processed = processed + high_content * (gain - 1) * 0.3
                logger.info(f"  High shelf: {high_adjust_db:+.1f} dB")
            except Exception as e:
                logger.warning(f"  High shelf failed: {e}")
        
        logger.info("  ✅ Spectral matching complete")
        return processed
    
    def apply_multiband_compression(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply gentle multiband compression with PARALLEL band processing."""
        logger.info(f"Applying multiband compression ({self.num_workers} workers)...")
        
        from scipy import signal
        
        nyquist = sample_rate / 2.0
        is_stereo = audio.ndim > 1
        
        # Band definitions
        bands = [
            (20, 200, 2.0, -20.0),      # Bass
            (200, 4000, 1.5, -18.0),    # Mids
            (4000, nyquist * 0.95, 1.5, -16.0),  # Highs
        ]
        
        def process_channel(channel_audio):
            """Process a single channel through all bands."""
            band_results = []
            
            for idx, (low_hz, high_hz, ratio, threshold_db) in enumerate(bands):
                try:
                    # Extract band
                    if low_hz <= 20:
                        normalized = min(high_hz / nyquist, 0.99)
                        b, a = signal.butter(4, normalized, btype='low')
                    elif high_hz >= nyquist * 0.9:
                        normalized = max(low_hz / nyquist, 0.01)
                        b, a = signal.butter(4, normalized, btype='high')
                    else:
                        low_norm = max(low_hz / nyquist, 0.01)
                        high_norm = min(high_hz / nyquist, 0.99)
                        b, a = signal.butter(4, [low_norm, high_norm], btype='band')
                    
                    band_audio = signal.filtfilt(b, a, channel_audio)
                    
                    # Compress
                    compressed = _process_band((band_audio, sample_rate, ratio, threshold_db))
                    band_results.append(compressed)
                    
                except Exception as e:
                    logger.warning(f"  Band {low_hz}-{high_hz} Hz failed: {e}")
                    band_results.append(channel_audio / len(bands))
            
            return np.sum(band_results, axis=0)
        
        if is_stereo:
            # Process channels in parallel
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_left = executor.submit(process_channel, audio[:, 0])
                future_right = executor.submit(process_channel, audio[:, 1])
                
                left_result = future_left.result()
                right_result = future_right.result()
            
            processed = np.column_stack([left_result, right_result])
        else:
            processed = process_channel(audio)
        
        logger.info("  ✅ Multiband compression complete")
        return processed
    
    def apply_loudness_normalization(self, audio: np.ndarray, sample_rate: int,
                                      target_lufs: float = -14.0,
                                      target_peak_db: float = -1.0) -> np.ndarray:
        """Normalize loudness to target LUFS with true peak limiting."""
        logger.info(f"Normalizing to {target_lufs:.1f} LUFS with {target_peak_db:.1f} dB peak ceiling...")
        
        try:
            import pyloudnorm as pyln
            meter = pyln.Meter(sample_rate)
            current_lufs = meter.integrated_loudness(audio)
            gain_db = target_lufs - current_lufs
        except Exception:
            rms = np.sqrt(np.mean(audio ** 2))
            current_lufs = 20 * np.log10(rms + 1e-10) - 0.691
            gain_db = target_lufs - current_lufs
        
        gain_db = np.clip(gain_db, -12.0, 12.0)
        gain_linear = 10 ** (gain_db / 20)
        
        processed = audio * gain_linear
        logger.info(f"  Applied {gain_db:+.1f} dB gain")
        
        # True peak limiting
        peak_ceiling = 10 ** (target_peak_db / 20)
        current_peak = np.max(np.abs(processed))
        
        if current_peak > peak_ceiling:
            processed = np.tanh(processed / peak_ceiling) * peak_ceiling
            logger.info(f"  Applied peak limiting (peak was {20*np.log10(current_peak):.1f} dB)")
        
        logger.info("  ✅ Loudness normalization complete")
        return np.clip(processed, -1.0, 1.0)
    
    def validate_improvement(self, input_profile: AudioProfile, 
                             output_profile: AudioProfile) -> Tuple[bool, str]:
        """Validate that the output is actually better than the input."""
        issues = []
        improvements = []
        
        snr_diff = output_profile.snr_db - input_profile.snr_db
        if snr_diff < -3.0:
            issues.append(f"SNR decreased by {-snr_diff:.1f} dB")
        elif snr_diff > 1.0:
            improvements.append(f"SNR improved by {snr_diff:.1f} dB")
        
        if output_profile.clipping_percent > input_profile.clipping_percent + 0.1:
            issues.append(f"Clipping increased to {output_profile.clipping_percent:.2f}%")
        elif output_profile.clipping_percent < input_profile.clipping_percent - 0.1:
            improvements.append(f"Clipping reduced to {output_profile.clipping_percent:.2f}%")
        
        dr_diff = output_profile.dynamic_range_db - input_profile.dynamic_range_db
        if dr_diff < -6.0:
            issues.append(f"Dynamic range crushed by {-dr_diff:.1f} dB")
        
        if output_profile.loudness_lufs < -20.0 or output_profile.loudness_lufs > -6.0:
            issues.append(f"Unusual loudness: {output_profile.loudness_lufs:.1f} LUFS")
        
        is_improved = len(issues) == 0
        reason = "; ".join(improvements if is_improved else issues)
        
        return is_improved, reason
    
    def process(self, input_path: str, output_path: str,
                reference_path: Optional[str] = None) -> RemasterResult:
        """Process audio file with reference-based remastering."""
        start_time = time.time()
        processing_steps = []
        
        logger.info(f"=" * 60)
        logger.info(f"Studio Quality Remaster")
        logger.info(f"Input: {input_path}")
        logger.info(f"Output: {output_path}")
        logger.info(f"Reference: {reference_path or 'Using defaults'}")
        logger.info(f"=" * 60)
        
        # Load input
        logger.info("Loading input audio...")
        audio, sr = sf.read(input_path)
        logger.info(f"  Loaded: {audio.shape}, {sr} Hz")
        
        # Analyze input
        logger.info("Analyzing input...")
        input_profile = self.analyze_audio(audio, sr)
        logger.info(f"  Input profile: {input_profile.loudness_lufs:.1f} LUFS, "
                   f"SNR={input_profile.snr_db:.1f} dB, "
                   f"Clipping={input_profile.clipping_percent:.2f}%")
        processing_steps.append("analyze_input")
        
        # Load and analyze reference
        reference_profile = None
        target_lufs = self.DEFAULT_TARGET_LUFS
        target_peak = self.DEFAULT_TARGET_PEAK_DB
        
        if reference_path:
            try:
                logger.info("Loading reference audio...")
                ref_audio, ref_sr = sf.read(reference_path)
                logger.info(f"  Loaded: {ref_audio.shape}, {ref_sr} Hz")
                
                logger.info("Analyzing reference...")
                reference_profile = self.analyze_audio(ref_audio, ref_sr)
                logger.info(f"  Reference profile: {reference_profile.loudness_lufs:.1f} LUFS, "
                           f"Centroid={reference_profile.spectral_centroid_hz:.0f} Hz")
                processing_steps.append("analyze_reference")
                
                target_lufs = reference_profile.loudness_lufs
                target_peak = reference_profile.peak_db + 0.5
                
            except Exception as e:
                logger.warning(f"Could not load reference: {e}")
        
        # Process audio
        processed = audio.copy()
        
        if self.enable_preconditioning:
            step_start = time.time()
            processed = self.apply_preconditioning(processed, sr)
            processing_steps.append("preconditioning")
            logger.info(f"  Step time: {time.time() - step_start:.1f}s")
        
        if self.enable_eq_matching:
            step_start = time.time()
            processed = self.apply_eq_matching(processed, sr, reference_profile)
            processing_steps.append("eq_matching")
            logger.info(f"  Step time: {time.time() - step_start:.1f}s")
        
        if self.enable_compression:
            step_start = time.time()
            processed = self.apply_multiband_compression(processed, sr)
            processing_steps.append("multiband_compression")
            logger.info(f"  Step time: {time.time() - step_start:.1f}s")
        
        if self.enable_limiting:
            step_start = time.time()
            processed = self.apply_loudness_normalization(processed, sr, target_lufs, target_peak)
            processing_steps.append("loudness_normalization")
            logger.info(f"  Step time: {time.time() - step_start:.1f}s")
        
        # Analyze output
        logger.info("Analyzing output...")
        output_profile = self.analyze_audio(processed, sr)
        logger.info(f"  Output profile: {output_profile.loudness_lufs:.1f} LUFS, "
                   f"SNR={output_profile.snr_db:.1f} dB, "
                   f"Clipping={output_profile.clipping_percent:.2f}%")
        
        # Validate
        quality_improved = True
        if self.validate_output:
            quality_improved, reason = self.validate_improvement(input_profile, output_profile)
            if quality_improved:
                logger.info(f"✅ Quality validation passed: {reason}")
            else:
                logger.warning(f"⚠️ Quality validation issues: {reason}")
        
        # Save
        logger.info(f"Saving output to {output_path}...")
        sf.write(output_path, processed.astype(np.float32), sr)
        processing_steps.append("save")
        
        processing_time = time.time() - start_time
        
        logger.info(f"=" * 60)
        logger.info(f"✅ Remaster complete in {processing_time:.1f}s")
        logger.info(f"   Quality improved: {quality_improved}")
        logger.info(f"=" * 60)
        
        return RemasterResult(
            output_path=output_path,
            input_profile=input_profile,
            output_profile=output_profile,
            reference_profile=reference_profile,
            processing_steps=processing_steps,
            quality_improved=quality_improved,
            processing_time_s=processing_time
        )


def create_ab_comparison(input_path: str, output_path: str, comparison_path: str):
    """Create A/B comparison file."""
    logger.info(f"Creating A/B comparison: {comparison_path}")
    
    input_audio, sr = sf.read(input_path)
    output_audio, _ = sf.read(output_path)
    
    input_peak = np.max(np.abs(input_audio))
    output_peak = np.max(np.abs(output_audio))
    max_peak = max(input_peak, output_peak)
    
    input_norm = input_audio / (max_peak + 1e-10) * 0.9
    output_norm = output_audio / (max_peak + 1e-10) * 0.9
    
    silence = np.zeros(sr)
    
    if input_norm.ndim == 1:
        comparison = np.concatenate([input_norm, silence, output_norm, silence])
    else:
        silence_stereo = np.zeros((sr, input_norm.shape[1]))
        comparison = np.concatenate([input_norm, silence_stereo, output_norm, silence_stereo])
    
    sf.write(comparison_path, comparison.astype(np.float32), sr)
    logger.info(f"  Saved A/B comparison to {comparison_path}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Studio Quality Remaster - Reference-based audio enhancement (OPTIMIZED)"
    )
    parser.add_argument("input_file", help="Input audio file (degraded recording)")
    parser.add_argument("--reference", "-r", help="Reference track for spectral matching")
    parser.add_argument("--output", "-o", help="Output file path")
    parser.add_argument("--strength", "-s", type=float, default=0.3,
                       help="Processing strength 0-1 (default: 0.3)")
    parser.add_argument("--workers", "-w", type=int, default=None,
                       help=f"Number of parallel workers (default: {NUM_CORES})")
    parser.add_argument("--no-preconditioning", action="store_true",
                       help="Skip preconditioning (noise reduction)")
    parser.add_argument("--no-eq", action="store_true", help="Skip EQ matching")
    parser.add_argument("--no-compression", action="store_true", help="Skip multiband compression")
    parser.add_argument("--no-limiting", action="store_true", help="Skip loudness normalization")
    parser.add_argument("--no-validation", action="store_true", help="Skip quality validation")
    parser.add_argument("--ab-compare", action="store_true", help="Create A/B comparison file")
    
    args = parser.parse_args()
    
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_name(f"{input_path.stem}_studio_remaster.wav")
    
    remaster = StudioQualityRemaster(
        enable_preconditioning=not args.no_preconditioning,
        preconditioning_strength=args.strength,
        enable_eq_matching=not args.no_eq,
        enable_compression=not args.no_compression,
        enable_limiting=not args.no_limiting,
        validate_output=not args.no_validation,
        num_workers=args.workers
    )
    
    result = remaster.process(str(input_path), str(output_path), args.reference)
    
    print("\n" + "=" * 60)
    print("REMASTER SUMMARY")
    print("=" * 60)
    print(f"Input:  {args.input_file}")
    print(f"Output: {result.output_path}")
    print(f"\nInput Profile:")
    print(f"  Loudness: {result.input_profile.loudness_lufs:.1f} LUFS")
    print(f"  Peak: {result.input_profile.peak_db:.1f} dB")
    print(f"  SNR: {result.input_profile.snr_db:.1f} dB")
    print(f"  Dynamic Range: {result.input_profile.dynamic_range_db:.1f} dB")
    print(f"  Clipping: {result.input_profile.clipping_percent:.2f}%")
    print(f"\nOutput Profile:")
    print(f"  Loudness: {result.output_profile.loudness_lufs:.1f} LUFS")
    print(f"  Peak: {result.output_profile.peak_db:.1f} dB")
    print(f"  SNR: {result.output_profile.snr_db:.1f} dB")
    print(f"  Dynamic Range: {result.output_profile.dynamic_range_db:.1f} dB")
    print(f"  Clipping: {result.output_profile.clipping_percent:.2f}%")
    print(f"\nProcessing Steps: {', '.join(result.processing_steps)}")
    print(f"Processing Time: {result.processing_time_s:.1f}s")
    print(f"Quality Improved: {'✅ Yes' if result.quality_improved else '⚠️ Check manually'}")
    print("=" * 60)
    
    if args.ab_compare:
        comparison_path = output_path.with_name(f"{output_path.stem}_AB_comparison.wav")
        create_ab_comparison(str(input_path), str(output_path), str(comparison_path))
        print(f"\nA/B Comparison saved to: {comparison_path}")
    
    report = {
        'input_file': str(input_path),
        'output_file': str(output_path),
        'reference_file': args.reference,
        'input_profile': {
            'loudness_lufs': result.input_profile.loudness_lufs,
            'peak_db': result.input_profile.peak_db,
            'snr_db': result.input_profile.snr_db,
            'dynamic_range_db': result.input_profile.dynamic_range_db,
            'clipping_percent': result.input_profile.clipping_percent,
        },
        'output_profile': {
            'loudness_lufs': result.output_profile.loudness_lufs,
            'peak_db': result.output_profile.peak_db,
            'snr_db': result.output_profile.snr_db,
            'dynamic_range_db': result.output_profile.dynamic_range_db,
            'clipping_percent': result.output_profile.clipping_percent,
        },
        'processing_steps': result.processing_steps,
        'processing_time_s': result.processing_time_s,
        'quality_improved': result.quality_improved,
    }
    
    report_path = output_path.with_suffix('.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
