"""
Quality Detector - Automatic stem damage detection based on artifact metrics.

Detects heavily damaged/cooked stems that need regeneration vs. stems that can be preserved.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class DamageLevel(Enum):
    """Classification of stem damage severity."""
    GOOD = "good"           # No significant damage, preserve as-is
    MINOR = "minor"         # Minor issues, DSP enhancement should suffice
    MODERATE = "moderate"   # Noticeable damage, may need selective regeneration
    SEVERE = "severe"       # Significant damage, regeneration recommended
    CRITICAL = "critical"   # Heavily cooked, regeneration required


@dataclass
class StemQualityReport:
    """Comprehensive quality report for a stem."""
    damage_level: DamageLevel
    confidence: float  # 0-1 confidence in damage assessment
    
    # Component scores (0-1, higher = more damaged)
    clipping_score: float = 0.0
    distortion_score: float = 0.0
    noise_score: float = 0.0
    artifact_score: float = 0.0
    spectral_score: float = 0.0
    
    # Recommendations
    needs_regeneration: bool = False
    regenerate_regions: List[Tuple[float, float]] = field(default_factory=list)
    preserve_regions: List[Tuple[float, float]] = field(default_factory=list)
    
    # Raw metrics
    snr_db: Optional[float] = None
    clipping_percent: Optional[float] = None
    overall_artifact_score: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'damage_level': self.damage_level.value,
            'confidence': self.confidence,
            'clipping_score': self.clipping_score,
            'distortion_score': self.distortion_score,
            'noise_score': self.noise_score,
            'artifact_score': self.artifact_score,
            'spectral_score': self.spectral_score,
            'needs_regeneration': self.needs_regeneration,
            'regenerate_regions': self.regenerate_regions,
            'preserve_regions': self.preserve_regions,
            'snr_db': self.snr_db,
            'clipping_percent': self.clipping_percent,
            'overall_artifact_score': self.overall_artifact_score,
        }


class QualityDetector:
    """
    Detects damage in audio stems and determines if regeneration is needed.
    
    Uses multiple signal analysis techniques to identify:
    - Clipping/hard limiting
    - Distortion artifacts
    - Noise floor issues
    - Spectral damage
    - Overall artifact presence
    """
    
    # Thresholds for damage classification
    THRESHOLDS = {
        'snr_good_db': 30.0,        # SNR above this = good
        'snr_poor_db': 15.0,        # SNR below this = poor
        
        'clipping_good_percent': 0.1,    # % clipped samples = good
        'clipping_poor_percent': 1.0,    # % clipped samples = poor
        
        'artifact_good': 0.2,       # Overall artifact score = good
        'artifact_poor': 0.5,       # Overall artifact score = poor
        
        'distortion_threshold': 0.3,  # Above this = distorted
        'noise_threshold': 0.4,       # Above this = noisy
        'spectral_threshold': 0.35,   # Above this = spectrally damaged
    }
    
    def __init__(self):
        """Initialize quality detector with default thresholds."""
        pass
    
    def analyze(self, audio: np.ndarray, sample_rate: int, 
                artifact_metrics: Optional[Dict] = None,
                snr_db: Optional[float] = None,
                clipping_percent: Optional[float] = None) -> StemQualityReport:
        """
        Analyze audio stem for damage.
        
        Args:
            audio: Audio samples (float32, range [-1, 1])
            sample_rate: Sample rate in Hz
            artifact_metrics: Optional dict of artifact scores from QualityMetrics
            snr_db: Optional pre-computed SNR in dB
            clipping_percent: Optional pre-computed clipping percentage
            
        Returns:
            StemQualityReport with damage assessment
        """
        # Compute basic metrics if not provided
        if snr_db is None:
            snr_db = self._compute_snr(audio, sample_rate)
        
        if clipping_percent is None:
            clipping_percent = self._compute_clipping(audio)
        
        # Compute component scores
        clipping_score = self._score_clipping(clipping_percent)
        distortion_score = self._score_distortion(audio, sample_rate)
        noise_score = self._score_noise(audio, sample_rate, snr_db)
        artifact_score = self._score_artifacts(artifact_metrics)
        spectral_score = self._score_spectral(audio, sample_rate)
        
        # Determine damage level
        damage_level, confidence = self._classify_damage(
            snr_db, clipping_percent, clipping_score, distortion_score,
            noise_score, artifact_score, spectral_score
        )
        
        # Determine regeneration needs
        needs_regenerate, regenerate_regions = self._detect_regeneration_regions(
            audio, sample_rate, damage_level, clipping_score, artifact_score
        )
        
        # Compute preserve regions (inverse of regenerate regions)
        preserve_regions = self._compute_preserve_regions(
            audio, sample_rate, regenerate_regions
        )
        
        return StemQualityReport(
            damage_level=damage_level,
            confidence=confidence,
            clipping_score=clipping_score,
            distortion_score=distortion_score,
            noise_score=noise_score,
            artifact_score=artifact_score,
            spectral_score=spectral_score,
            needs_regeneration=needs_regenerate,
            regenerate_regions=regenerate_regions,
            preserve_regions=preserve_regions,
            snr_db=snr_db,
            clipping_percent=clipping_percent,
            overall_artifact_score=artifact_metrics.get('overall_artifact_score', 0.0) if artifact_metrics else None
        )
    
    def _compute_snr(self, audio: np.ndarray, sample_rate: int) -> float:
        """Estimate SNR using spectral analysis."""
        try:
            import librosa
            
            # Compute magnitude spectrogram
            n_fft = 2048
            hop_length = 512
            S = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))
            S_db = librosa.amplitude_to_db(S, ref=np.max)
            
            # Estimate noise floor from quietest frames
            frame_means = np.mean(S_db, axis=0)
            sorted_indices = np.argsort(frame_means)
            noise_frames = int(0.1 * len(sorted_indices))
            noise_floor = np.mean(frame_means[sorted_indices[:noise_frames]])
            
            # Estimate signal from loudest frames
            signal_frames = int(0.1 * len(sorted_indices))
            signal_level = np.mean(frame_means[sorted_indices[-signal_frames:]])
            
            # SNR is difference
            snr = signal_level - noise_floor
            return float(max(snr, 0))  # Ensure non-negative
            
        except Exception as e:
            logger.warning(f"SNR computation failed: {e}")
            return 30.0  # Default to good
    
    def _compute_clipping(self, audio: np.ndarray) -> float:
        """Compute percentage of clipped samples."""
        threshold = 0.999
        clipped = np.sum(np.abs(audio) >= threshold)
        return float(clipped / len(audio) * 100)
    
    def _score_clipping(self, clipping_percent: float) -> float:
        """Score clipping damage (0-1, higher = more damage)."""
        threshold_good = self.THRESHOLDS['clipping_good_percent']
        threshold_poor = self.THRESHOLDS['clipping_poor_percent']
        
        if clipping_percent <= threshold_good:
            return 0.0
        elif clipping_percent >= threshold_poor:
            return 1.0
        else:
            return (clipping_percent - threshold_good) / (threshold_poor - threshold_good)
    
    def _score_distortion(self, audio: np.ndarray, sample_rate: int) -> float:
        """Score distortion artifacts using harmonic analysis."""
        try:
            import librosa
            
            # Compute harmonic-percussive separation
            harmonic, percussive = librosa.effects.hpss(audio)
            
            # High harmonic content relative to percussive can indicate distortion
            harmonic_energy = np.mean(harmonic ** 2)
            percussive_energy = np.mean(percussive ** 2)

            # If both energies are near-zero (silent), consider distortion low.
            if harmonic_energy < 1e-12 and percussive_energy < 1e-12:
                return 0.0

            # Compute spectral flatness to distinguish tonal (pure harmonic) from noisy/distorted
            flatness = np.mean(librosa.feature.spectral_flatness(y=audio))

            # Avoid division-by-zero
            ratio = harmonic_energy / (percussive_energy + 1e-12)

            # Combine ratio with spectral flatness: pure tones (low flatness) should not be scored as high distortion
            score = min((ratio / 3.0) * flatness, 1.0)
            return float(score)
            
        except Exception as e:
            logger.warning(f"Distortion scoring failed: {e}")
            return 0.0
    
    def _score_noise(self, audio: np.ndarray, sample_rate: int, snr_db: float) -> float:
        """Score noise floor issues."""
        threshold_good = self.THRESHOLDS['snr_good_db']
        threshold_poor = self.THRESHOLDS['snr_poor_db']
        
        if snr_db >= threshold_good:
            return 0.0
        elif snr_db <= threshold_poor:
            return 1.0
        else:
            return 1.0 - (snr_db - threshold_poor) / (threshold_good - threshold_poor)
    
    def _score_artifacts(self, artifact_metrics: Optional[Dict]) -> float:
        """Score overall artifact presence."""
        if artifact_metrics is None:
            return 0.0
        
        # Use overall artifact score if available
        if 'overall_artifact_score' in artifact_metrics:
            return artifact_metrics['overall_artifact_score']
        
        # Otherwise compute from components
        scores = []
        if 'metallic_score' in artifact_metrics:
            scores.append(artifact_metrics['metallic_score'])
        if 'ringing_score' in artifact_metrics:
            scores.append(artifact_metrics['ringing_score'])
        if 'clicking_score' in artifact_metrics:
            scores.append(artifact_metrics['clicking_score'])
        if 'phase_distortion' in artifact_metrics:
            scores.append(artifact_metrics['phase_distortion'])
        
        if scores:
            return min(np.mean(scores), 1.0)
        return 0.0
    
    def _score_spectral(self, audio: np.ndarray, sample_rate: int) -> float:
        """Score spectral damage (holes, discontinuities)."""
        try:
            import librosa
            
            # Compute mel spectrogram
            n_mels = 128
            S = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=n_mels)
            S_db = librosa.power_to_db(S, ref=np.max)
            
            # Check for spectral holes (very low energy regions)
            mean_energy = np.mean(S_db)
            hole_regions = np.sum(S_db < mean_energy - 30)  # 30 dB below mean
            
            # Score based on hole percentage
            hole_percent = hole_regions / S_db.size
            score = min(hole_percent * 10, 1.0)  # Scale and clamp
            
            return score
            
        except Exception as e:
            logger.warning(f"Spectral scoring failed: {e}")
            return 0.0
    
    def _classify_damage(self, snr_db: float, clipping_percent: float,
                         clipping_score: float, distortion_score: float,
                         noise_score: float, artifact_score: float,
                         spectral_score: float) -> Tuple[DamageLevel, float]:
        """Classify overall damage level from component scores."""
        
        # Compute weighted damage score
        weights = {
            'clipping': 0.25,
            'distortion': 0.20,
            'noise': 0.15,
            'artifact': 0.25,
            'spectral': 0.15,
        }
        
        damage_score = (
            weights['clipping'] * clipping_score +
            weights['distortion'] * distortion_score +
            weights['noise'] * noise_score +
            weights['artifact'] * artifact_score +
            weights['spectral'] * spectral_score
        )
        
        # Classify based on damage score
        if damage_score < 0.15:
            return DamageLevel.GOOD, 0.9
        elif damage_score < 0.30:
            return DamageLevel.MINOR, 0.85
        elif damage_score < 0.50:
            return DamageLevel.MODERATE, 0.80
        elif damage_score < 0.70:
            return DamageLevel.SEVERE, 0.75
        else:
            return DamageLevel.CRITICAL, 0.85
    
    def _detect_regeneration_regions(self, audio: np.ndarray, sample_rate: int,
                                      damage_level: DamageLevel,
                                      clipping_score: float,
                                      artifact_score: float,
                                      force_regenerate: bool = False) -> Tuple[bool, List[Tuple[float, float]]]:
        """Detect time regions that need regeneration."""
        
        if not force_regenerate and damage_level in [DamageLevel.GOOD, DamageLevel.MINOR]:
            return False, []
        
        # For moderate+ damage (or forced), identify problematic regions
        try:
            import librosa
            
            # Compute frame-level energy
            frame_length = 2048
            hop_length = 512
            energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
            
            # Detect low-energy regions (potential dropouts)
            energy_normalized = (energy - np.min(energy)) / (np.max(energy) - np.min(energy) + 1e-10)
            
            # Find regions with very low energy AND clipping
            regenerate_regions = []
            threshold = 0.1  # Low energy threshold
            
            in_region = False
            region_start = 0
            
            for i, e in enumerate(energy_normalized):
                time_pos = i * hop_length / sample_rate
                
                if e < threshold and not in_region:
                    in_region = True
                    region_start = time_pos
                elif e >= threshold and in_region:
                    in_region = False
                    # Only record if region is significant (>0.5 seconds)
                    if time_pos - region_start > 0.5:
                        regenerate_regions.append((region_start, time_pos))
            
            # Handle region extending to end
            if in_region and len(energy) > 0:
                end_time = len(energy) * hop_length / sample_rate
                if end_time - region_start > 0.5:
                    regenerate_regions.append((region_start, end_time))
            
            # For critical damage, recommend whole-stem regeneration
            if damage_level == DamageLevel.CRITICAL:
                regenerate_regions = [(0.0, len(audio) / sample_rate)]
            
            needs_regen = len(regenerate_regions) > 0 or damage_level == DamageLevel.CRITICAL
            
            return needs_regen, regenerate_regions
            
        except Exception as e:
            logger.warning(f"Region detection failed: {e}")
            # Fall back to whole-stem for severe/critical damage
            if damage_level in [DamageLevel.SEVERE, DamageLevel.CRITICAL]:
                return True, [(0.0, len(audio) / sample_rate)]
            return False, []
    
    def _compute_preserve_regions(self, audio: np.ndarray, sample_rate: int,
                                   regenerate_regions: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Compute inverse of regenerate regions (portions to preserve)."""
        if not regenerate_regions:
            return [(0.0, len(audio) / sample_rate)]
        
        audio_duration = len(audio) / sample_rate
        preserve_regions = []
        
        current_time = 0.0
        
        for reg_start, reg_end in regenerate_regions:
            if current_time < reg_start:
                preserve_regions.append((current_time, reg_start))
            current_time = reg_end
        
        if current_time < audio_duration:
            preserve_regions.append((current_time, audio_duration))
        
        return preserve_regions
    
    def should_regenerate(self, report: StemQualityReport) -> bool:
        """Quick check if stem needs regeneration."""
        return report.needs_regeneration or report.damage_level in [
            DamageLevel.SEVERE, DamageLevel.CRITICAL
        ]
    
    def get_stem_priority(self, reports: Dict[str, StemQualityReport]) -> List[str]:
        """Get list of stem names sorted by regeneration priority."""
        # Sort by damage level (worst first)
        priority_order = [
            DamageLevel.CRITICAL,
            DamageLevel.SEVERE,
            DamageLevel.MODERATE,
            DamageLevel.MINOR,
            DamageLevel.GOOD,
        ]
        
        sorted_stems = []
        for damage_level in priority_order:
            for stem_name, report in reports.items():
                if report.damage_level == damage_level:
                    sorted_stems.append(stem_name)
        
        # Add any missing stems
        for stem_name in reports:
            if stem_name not in sorted_stems:
                sorted_stems.append(stem_name)
        
        return sorted_stems


# Convenience function
def detect_stem_quality(audio: np.ndarray, sample_rate: int,
                        artifact_metrics: Optional[Dict] = None,
                        snr_db: Optional[float] = None) -> StemQualityReport:
    """
    Quick function to detect stem quality.
    
    Args:
        audio: Audio samples
        sample_rate: Sample rate
        artifact_metrics: Optional artifact scores
        snr_db: Optional pre-computed SNR
        
    Returns:
        StemQualityReport
    """
    detector = QualityDetector()
    return detector.analyze(audio, sample_rate, artifact_metrics, snr_db)
