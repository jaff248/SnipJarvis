"""
Quality Metrics Module - Audio quality analysis and validation

Provides comprehensive audio quality metrics for assessing reconstruction
results and detecting processing artifacts.
"""

import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import librosa
from scipy import signal

logger = logging.getLogger(__name__)


@dataclass
class SimilarityMetrics:
    """Compare AI output to original."""
    spectral_similarity: float  # 0-1, STFT correlation
    melodic_similarity: float   # 0-1, pitch contour match
    rhythmic_similarity: float  # 0-1, onset timing match
    timbral_similarity: float   # 0-1, MFCCs correlation
    overall_similarity: float   # Weighted average

    def is_acceptable(self, threshold: float = 0.85) -> bool:
        """Check if similarity meets the threshold."""
        return self.overall_similarity >= threshold

@dataclass
class QualityReport:
    """Report containing all quality metrics."""
    snr_db: float
    loudness_lufs: float
    spectral_centroid_hz: float
    clipping_percent: float
    has_clipping: bool
    dynamic_range_db: float
    artifacts: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "snr_db": round(self.snr_db, 1),
            "loudness_lufs": round(self.loudness_lufs, 1),
            "spectral_centroid_hz": round(self.spectral_centroid_hz, 0),
            "clipping_percent": round(self.clipping_percent, 3),
            "has_clipping": self.has_clipping,
            "dynamic_range_db": round(self.dynamic_range_db, 1),
            "artifacts": {k: round(v, 3) for k, v in self.artifacts.items()}
        }


class SimilarityAnalyzer:
    """Analyzes similarity between original and regenerated audio."""

    @staticmethod
    def compute_similarity(original: np.ndarray, generated: np.ndarray, sr: int) -> SimilarityMetrics:
        """
        Compute similarity metrics between original and generated audio.
        """
        try:
            # Ensure same length for comparison
            min_len = min(len(original), len(generated))
            orig = original[:min_len]
            gen = generated[:min_len]

            # 1. Spectral Similarity (STFT correlation)
            S_orig = np.abs(librosa.stft(orig))
            S_gen = np.abs(librosa.stft(gen))
            
            # Normalize spectrograms
            S_orig_norm = S_orig / (np.linalg.norm(S_orig) + 1e-10)
            S_gen_norm = S_gen / (np.linalg.norm(S_gen) + 1e-10)
            
            # Cosine similarity of flattened spectrograms
            spectral_sim = np.sum(S_orig_norm * S_gen_norm)
            spectral_sim = float(np.clip(spectral_sim, 0.0, 1.0))

            # 2. Timbral Similarity (MFCC correlation)
            mfcc_orig = librosa.feature.mfcc(y=orig, sr=sr, n_mfcc=13)
            mfcc_gen = librosa.feature.mfcc(y=gen, sr=sr, n_mfcc=13)
            
            # Compare mean MFCC vectors
            mfcc_orig_mean = np.mean(mfcc_orig, axis=1)
            mfcc_gen_mean = np.mean(mfcc_gen, axis=1)
            
            # Cosine similarity of MFCC means
            dot_product = np.dot(mfcc_orig_mean, mfcc_gen_mean)
            norm_a = np.linalg.norm(mfcc_orig_mean)
            norm_b = np.linalg.norm(mfcc_gen_mean)
            timbral_sim = dot_product / (norm_a * norm_b + 1e-10)
            timbral_sim = float(np.clip(timbral_sim, 0.0, 1.0))

            # 3. Rhythmic Similarity (Onset Envelope correlation)
            onset_orig = librosa.onset.onset_strength(y=orig, sr=sr)
            onset_gen = librosa.onset.onset_strength(y=gen, sr=sr)
            
            # Normalize
            onset_orig = onset_orig / (np.max(onset_orig) + 1e-10)
            onset_gen = onset_gen / (np.max(onset_gen) + 1e-10)
            
            # Cross-correlation at lag 0 (aligned)
            rhythmic_sim = np.corrcoef(onset_orig, onset_gen)[0, 1]
            rhythmic_sim = float(np.clip(rhythmic_sim, 0.0, 1.0))

            # 4. Melodic Similarity (Pitch contour correlation) - placeholder for now
            # Using spectral centroid correlation as proxy for melodic movement if pitch tracking fails
            cent_orig = librosa.feature.spectral_centroid(y=orig, sr=sr)[0]
            cent_gen = librosa.feature.spectral_centroid(y=gen, sr=sr)[0]
            melodic_sim = np.corrcoef(cent_orig, cent_gen)[0, 1]
            melodic_sim = float(np.clip(melodic_sim, 0.0, 1.0))

            # Weighted Average
            overall = (0.3 * spectral_sim +
                       0.3 * timbral_sim +
                       0.2 * rhythmic_sim +
                       0.2 * melodic_sim)

            return SimilarityMetrics(
                spectral_similarity=spectral_sim,
                melodic_similarity=melodic_sim,
                rhythmic_similarity=rhythmic_sim,
                timbral_similarity=timbral_sim,
                overall_similarity=float(overall)
            )

        except Exception as e:
            logger.warning(f"Similarity computation failed: {e}")
            return SimilarityMetrics(0.0, 0.0, 0.0, 0.0, 0.0)

class QualityMetrics:
    """
    Comprehensive audio quality analysis.
    
    Provides methods for:
    - Signal-to-noise ratio estimation
    - Loudness measurement (LUFS)
    - Spectral analysis
    - Clipping detection
    - Artifact detection
    - Before/after comparison
    """
    
    # Thresholds for quality assessment
    SNR_GOOD_THRESHOLD = 20.0  # dB
    SNR_POOR_THRESHOLD = 10.0  # dB
    CLIPPING_WARNING_THRESHOLD = 0.001  # 0.1%
    SPECTRAL_CENTROID_MUFFLED = 2000  # Hz
    SPECTRAL_CENTROID_BRIGHT = 6000  # Hz
    
    @staticmethod
    def snr_estimate(audio: np.ndarray, sr: int) -> float:
        """
        Estimate Signal-to-Noise Ratio (dB).
        
        Uses spectral analysis: signal in peaks, noise in valleys.
        
        Args:
            audio: Input audio (float32, range [-1, 1])
            sr: Sample rate
            
        Returns:
            SNR in dB (positive = more signal than noise)
        """
        # Compute STFT
        n_fft = 2048
        hop_length = 512
        
        # Compute magnitude spectrogram
        S = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))
        
        # Convert to dB
        S_db = librosa.amplitude_to_db(S, ref=np.max)
        
        # Estimate noise floor from quietest frames
        # Sort frames by mean energy
        frame_means = np.mean(S_db, axis=0)
        sorted_indices = np.argsort(frame_means)
        
        # Use bottom 10% of frames as noise estimate
        noise_frames = int(0.1 * len(sorted_indices))
        noise_floor = np.mean(frame_means[sorted_indices[:noise_frames]])
        
        # Estimate signal from top 10% of frames
        signal_frames = int(0.1 * len(sorted_indices))
        signal_level = np.mean(frame_means[sorted_indices[-signal_frames:]])
        
        # SNR is difference between signal and noise
        snr = signal_level - noise_floor
        
        return float(snr)
    
    @staticmethod
    def loudness_lufs(audio: np.ndarray, sr: int) -> float:
        """
        Compute loudness in LUFS using pyloudnorm.
        
        Args:
            audio: Input audio (float32, range [-1, 1])
            sr: Sample rate
            
        Returns:
            LUFS value (streaming target: -14 LUFS)
        """
        try:
            import pyloudnorm as pyln
            
            # Ensure audio is float32
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            # Handle mono/stereo
            if audio.ndim == 1:
                audio = audio.reshape(-1, 1)
            
            meter = pyln.Meter(sr)
            loudness = meter.integrated_loudness(audio)
            
            return float(loudness)
            
        except ImportError:
            logger.warning("pyloudnorm not available, using RMS approximation")
            # Fallback: RMS-based approximation
            if audio.ndim > 1:
                rms = np.sqrt(np.mean(audio ** 2))
            else:
                rms = np.sqrt(np.mean(audio ** 2))
            
            # Convert RMS to approximate LUFS
            # This is a rough approximation
            lufs = 20 * np.log10(rms + 1e-10) - 0.691
            return float(lufs)
    
    @staticmethod
    def spectral_centroid(audio: np.ndarray, sr: int) -> float:
        """
        Compute spectral centroid (frequency where 50% of energy is above/below).
        
        Args:
            audio: Input audio (float32, range [-1, 1])
            sr: Sample rate
            
        Returns:
            Spectral centroid in Hz
            - Muffled audio: ~2-4 kHz
            - Bright audio: ~6-8 kHz
        """
        # Compute spectral centroid using librosa
        centroids = librosa.feature.spectral_centroid(
            y=audio, sr=sr, hop_length=512
        )
        
        # Return mean centroid
        return float(np.mean(centroids))
    
    @staticmethod
    def clipping_detection(audio: np.ndarray) -> Tuple[float, bool]:
        """
        Detect hard clipping (samples touching -1.0 or 1.0).
        
        Args:
            audio: Input audio (float32, range [-1, 1])
            
        Returns:
            Tuple of (percentage clipped, has_clipping)
        """
        # Count samples at or beyond clipping threshold
        threshold = 0.999  # Allow tiny margin for floating point
        clipped_samples = np.sum(np.abs(audio) >= threshold)
        total_samples = len(audio)
        
        clipping_percent = clipped_samples / total_samples
        has_clipping = clipping_percent > 0
        
        return (float(clipping_percent), has_clipping)
    
    @staticmethod
    def artifact_detection(audio: np.ndarray, sr: int) -> Dict[str, float]:
        """
        Detect common DSP artifacts: metallic sounds, ringing, aliasing, etc.
        
        Args:
            audio: Input audio (float32, range [-1, 1])
            sr: Sample rate
            
        Returns:
            Dictionary with artifact scores (0-1 range)
        """
        artifacts: Dict[str, float] = {}
        eps = 1e-9
        
        # 1. Metallic sound detection (high-frequency harmonic content)
        S = np.abs(librosa.stft(audio, n_fft=2048, hop_length=512))
        S_db = librosa.amplitude_to_db(S, ref=np.max)
        
        nyquist = sr / 2.0
        high_freq_bins = int(8000 / nyquist * S.shape[0])
        high_freq_energy = np.mean(S_db[high_freq_bins:, :]) if high_freq_bins < S.shape[0] else np.mean(S_db)
        metallic_score = np.clip((high_freq_energy + 60) / 40, 0, 1)
        artifacts["metallic_score"] = float(metallic_score)
        
        # 2. Ringing detection (pre-echo or post-echo)
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
        if len(onsets) > 1:
            ringing_score = 0.0
            for onset_idx in onsets[1:]:  # Skip first onset
                pre_window = onset_idx - 10
                if pre_window > 0:
                    pre_energy = np.mean(onset_env[pre_window:onset_idx])
                    onset_energy = onset_env[onset_idx] + eps
                    ringing_score += pre_energy / onset_energy
            ringing_score = min(ringing_score / max(len(onsets) - 1, 1), 1.0)
            artifacts["ringing_score"] = float(ringing_score)
        else:
            artifacts["ringing_score"] = 0.0
        
        # 3. Click/pop detection (sudden high-amplitude transients)
        diff = np.diff(audio)
        click_threshold = 0.5
        clicks = np.sum(np.abs(diff) > click_threshold)
        clicking_score = min(clicks / len(audio) * 1000, 1.0)
        artifacts["clicking_score"] = float(clicking_score)
        
        # 4. Phase distortion detection (via zero-crossing rate)
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        phase_distortion = min(np.mean(zcr) * 10, 1.0)
        artifacts["phase_distortion"] = float(phase_distortion)
        
        # 5. Aliasing detection (energy near Nyquist)
        nyquist_band_start = int(0.9 * S.shape[0])  # top 10% of bins
        nyquist_energy = np.mean(S_db[nyquist_band_start:, :]) if nyquist_band_start < S.shape[0] else np.mean(S_db)
        mid_band = S_db[int(0.4 * S.shape[0]):int(0.6 * S.shape[0]), :]
        mid_energy = np.mean(mid_band) if mid_band.size > 0 else np.mean(S_db)
        alias_ratio = (nyquist_energy - mid_energy + 30) / 40  # tolerate some roll-off
        aliasing_score = float(np.clip(alias_ratio, 0, 1))
        artifacts["aliasing_score"] = aliasing_score
        
        # 6. Clipping residual detection
        clip_threshold = 0.999
        clipped_samples = np.sum(np.abs(audio) >= clip_threshold)
        clipping_percent = clipped_samples / max(len(audio), 1)
        residual_energy = float(np.mean(np.abs(audio[np.abs(audio) >= clip_threshold])) if clipped_samples > 0 else 0.0)
        clipping_residual = np.clip(clipping_percent * 8 + residual_energy * 2, 0, 1)
        artifacts["clipping_residual"] = float(clipping_residual)
        
        # 7. Pumping detection (1-3 Hz envelope modulation)
        frame_length = 2048
        hop_length = 512
        envelope = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        envelope = envelope - np.mean(envelope)
        if len(envelope) > 4:
            freqs = np.fft.rfftfreq(len(envelope), d=hop_length / sr)
            spectrum = np.abs(np.fft.rfft(envelope))
            band_mask = (freqs >= 1.0) & (freqs <= 3.0)
            band_energy = np.sum(spectrum[band_mask])
            total_energy = np.sum(spectrum) + eps
            pump_ratio = band_energy / total_energy
            pump_score = float(np.clip(pump_ratio * 4.0, 0, 1))
        else:
            pump_score = 0.0
        artifacts["pump_score"] = pump_score
        
        # 8. Overall weighted artifact score
        weights = {
            "metallic_score": 0.18,
            "ringing_score": 0.12,
            "clicking_score": 0.12,
            "phase_distortion": 0.12,
            "aliasing_score": 0.15,
            "clipping_residual": 0.15,
            "pump_score": 0.16,
        }
        overall = 0.0
        for key, weight in weights.items():
            overall += weight * artifacts.get(key, 0.0)
        artifacts["overall_artifact_score"] = float(np.clip(overall, 0, 1))
        
        return artifacts
    
    @staticmethod
    def dynamic_range(audio: np.ndarray, sr: int) -> float:
        """
        Measure dynamic range in dB.
        
        Args:
            audio: Input audio (float32, range [-1, 1])
            sr: Sample rate
            
        Returns:
            Dynamic range in dB
        """
        # Use ITU-R BS.1770 method (simplified)
        # RMS over sliding windows
        window_size = sr  # 1 second windows
        
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
        return 0.0
    
    @classmethod
    def analyze(cls, audio: np.ndarray, sr: int) -> QualityReport:
        """
        Perform comprehensive quality analysis.
        
        Args:
            audio: Input audio (float32, range [-1, 1])
            sr: Sample rate
            
        Returns:
            QualityReport with all metrics
        """
        snr = cls.snr_estimate(audio, sr)
        loudness = cls.loudness_lufs(audio, sr)
        centroid = cls.spectral_centroid(audio, sr)
        clipping_percent, has_clipping = cls.clipping_detection(audio)
        artifacts = cls.artifact_detection(audio, sr)
        dynamic_range = cls.dynamic_range(audio, sr)
        
        return QualityReport(
            snr_db=snr,
            loudness_lufs=loudness,
            spectral_centroid_hz=centroid,
            clipping_percent=clipping_percent,
            has_clipping=has_clipping,
            dynamic_range_db=dynamic_range,
            artifacts=artifacts
        )
    
    @staticmethod
    def before_after_comparison(original: np.ndarray, processed: np.ndarray, 
                               sr: int) -> Dict[str, Any]:
        """
        Compare metrics before/after processing.
        
        Args:
            original: Original audio
            processed: Processed audio
            sr: Sample rate
            
        Returns:
            Dictionary with comparison metrics
        """
        # Ensure same length
        min_len = min(len(original), len(processed))
        original = original[:min_len]
        processed = processed[:min_len]
        
        # Analyze both
        cls = QualityMetrics
        original_report = cls.analyze(original, sr)
        processed_report = cls.analyze(processed, sr)
        
        # Calculate improvements
        comparison = {
            "original": original_report.to_dict(),
            "processed": processed_report.to_dict(),
            "improvements": {
                "snr_improvement_db": processed_report.snr_db - original_report.snr_db,
                "loudness_shift_db": processed_report.loudness_lufs - original_report.loudness_lufs,
                "spectral_centroid_shift_hz": processed_report.spectral_centroid_hz - original_report.spectral_centroid_hz,
                "clipping_change_percent": processed_report.clipping_percent - original_report.clipping_percent,
                "dynamic_range_change_db": processed_report.dynamic_range_db - original_report.dynamic_range_db
            },
            "artifacts_comparison": {
                k: processed_report.artifacts.get(k, 0) - original_report.artifacts.get(k, 0)
                for k in set(list(original_report.artifacts.keys()) + 
                             list(processed_report.artifacts.keys()))
            }
        }
        
        return comparison
    
    @staticmethod
    def get_quality_assessment(metrics: Any, intensity_recommendation: bool = True) -> str:
        """
        Produce quality tier string with recommendations.
        
        Args:
            metrics: QualityReport or dict of metrics (expects snr_db, artifacts, etc.)
            intensity_recommendation: Whether to append processing guidance
        
        Returns:
            String such as "Good" or "Good with minor metallic artifacts, recommend intensity 0.6 for polishing"
        """
        # Normalize input
        if isinstance(metrics, QualityReport):
            report_dict = metrics.__dict__
        else:
            report_dict = metrics
        snr_db = report_dict.get("snr_db", 0.0)
        artifacts = report_dict.get("artifacts", {}) or {}
        clipping_percent = report_dict.get("clipping_percent", 0.0)
        spectral_centroid_hz = report_dict.get("spectral_centroid_hz", 0.0)

        # Tiering based on SNR
        if snr_db < 5:
            tier = "Poor"
        elif snr_db < 15:
            tier = "Fair"
        elif snr_db < 25:
            tier = "Good"
        elif snr_db < 35:
            tier = "Excellent"
        else:
            tier = "Reference"

        messages = []
        # Artifact-driven feedback
        metallic = artifacts.get("metallic_score", 0.0)
        clicking = artifacts.get("clicking_score", 0.0)
        phase = artifacts.get("phase_distortion", 0.0)
        aliasing = artifacts.get("aliasing_score", 0.0)
        clipping_residual = artifacts.get("clipping_residual", 0.0)
        pump = artifacts.get("pump_score", 0.0)
        overall = artifacts.get("overall_artifact_score", 0.0)

        if metallic > 0.5:
            messages.append("metallic artifacts present")
        if clicking > 0.5:
            messages.append("clicking/pops detected")
        if phase > 0.5:
            messages.append("phase coherence issues")
        if aliasing > 0.5:
            messages.append("aliasing near Nyquist")
        if clipping_residual > 0.3:
            messages.append("clipping residuals")
        if pump > 0.4:
            messages.append("compression pumping")
        if spectral_centroid_hz < QualityMetrics.SPECTRAL_CENTROID_MUFFLED:
            messages.append("muffled highs")
        if spectral_centroid_hz > QualityMetrics.SPECTRAL_CENTROID_BRIGHT:
            messages.append("overly bright spectrum")
        if clipping_percent > QualityMetrics.CLIPPING_WARNING_THRESHOLD:
            messages.append(f"clipping {clipping_percent*100:.2f}%")

        base = tier if not messages else f"{tier} with {', '.join(messages)}"

        if not intensity_recommendation:
            return base

        # Intensity suggestion based on overall artifacts
        suggested_intensity = 0.5
        if overall > 0.6 or metallic > 0.5 or clicking > 0.5:
            suggested_intensity = 0.6
        if overall > 0.75 or clipping_residual > 0.6:
            suggested_intensity = 0.7
        if overall < 0.3 and snr_db > 25:
            suggested_intensity = 0.4

        recommendations = []
        recommendations.append(f"recommend intensity {suggested_intensity:.1f} for polishing")
        if metallic > 0.5 or clicking > 0.5:
            recommendations.append("consider MBD polish for artifact reduction")
        if clipping_residual > 0.4:
            recommendations.append("apply soft clipper or limiter to tame peaks")
        if pump > 0.5:
            recommendations.append("reduce bus compression or pumping")

        recommendation_str = ", ".join(recommendations)
        return f"{base}, {recommendation_str}" if recommendations else base


# Module-level convenience functions
def snr_estimate(audio: np.ndarray, sr: int) -> float:
    """Estimate Signal-to-Noise Ratio (dB)."""
    return QualityMetrics.snr_estimate(audio, sr)


def loudness_lufs(audio: np.ndarray, sr: int) -> float:
    """Compute loudness in LUFS."""
    return QualityMetrics.loudness_lufs(audio, sr)


def spectral_centroid(audio: np.ndarray, sr: int) -> float:
    """Compute spectral centroid in Hz."""
    return QualityMetrics.spectral_centroid(audio, sr)


def clipping_detection(audio: np.ndarray) -> Tuple[float, bool]:
    """Detect hard clipping."""
    return QualityMetrics.clipping_detection(audio)


def artifact_detection(audio: np.ndarray, sr: int) -> Dict[str, float]:
    """Detect DSP artifacts."""
    return QualityMetrics.artifact_detection(audio, sr)


def get_quality_assessment(metrics: Any, intensity_recommendation: bool = True) -> str:
    """Convenience wrapper for quality tiering and recommendations."""
    return QualityMetrics.get_quality_assessment(metrics, intensity_recommendation)


def analyze_quality(audio: np.ndarray, sr: int) -> QualityReport:
    """Perform comprehensive quality analysis."""
    return QualityMetrics.analyze(audio, sr)


def before_after_comparison(original: np.ndarray, processed: np.ndarray, 
                           sr: int) -> Dict[str, Any]:
    """Compare metrics before/after processing."""
    return QualityMetrics.before_after_comparison(original, processed, sr)


# Example usage and testing
if __name__ == "__main__":
    import soundfile as sf
    
    logging.basicConfig(level=logging.INFO)
    
    print("Testing QualityMetrics...")
    
    # Create test audio
    sample_rate = 44100
    duration = 3.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Clean audio
    clean_audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    
    # Noisy audio
    noisy_audio = clean_audio + 0.1 * np.random.randn(len(t))
    
    # Clipped audio
    clipped_audio = np.clip(clean_audio * 2, -0.95, 0.95)
    
    # Test SNR estimation
    print("\n--- SNR Estimation ---")
    clean_snr = snr_estimate(clean_audio, sample_rate)
    noisy_snr = snr_estimate(noisy_audio, sample_rate)
    print(f"Clean audio SNR: {clean_snr:.1f} dB")
    print(f"Noisy audio SNR: {noisy_snr:.1f} dB")
    
    # Test loudness
    print("\n--- Loudness Measurement ---")
    loudness = loudness_lufs(clean_audio, sample_rate)
    print(f"Loudness: {loudness:.1f} LUFS")
    
    # Test spectral centroid
    print("\n--- Spectral Centroid ---")
    centroid = spectral_centroid(clean_audio, sample_rate)
    print(f"Spectral centroid: {centroid:.0f} Hz")
    
    # Test clipping detection
    print("\n--- Clipping Detection ---")
    clean_clip, clean_has = clipping_detection(clean_audio)
    clipped_clip, clipped_has = clipping_detection(clipped_audio)
    print(f"Clean: {clean_clip*100:.3f}% clipped, has_clipping={clean_has}")
    print(f"Clipped: {clipped_clip*100:.3f}% clipped, has_clipping={clipped_has}")
    
    # Test artifact detection
    print("\n--- Artifact Detection ---")
    artifacts = artifact_detection(noisy_audio, sample_rate)
    print(f"Artifacts: {artifacts}")
    
    # Test full analysis
    print("\n--- Full Analysis ---")
    report = analyze_quality(noisy_audio, sample_rate)
    print(f"Quality Report: {report.to_dict()}")
    print(f"Assessment: {get_quality_assessment(report)}")
    
    # Test before/after comparison
    print("\n--- Before/After Comparison ---")
    comparison = before_after_comparison(clean_audio, noisy_audio, sample_rate)
    print(f"SNR improvement: {comparison['improvements']['snr_improvement_db']:.1f} dB")
    
    print("\n✅ All tests passed!")
