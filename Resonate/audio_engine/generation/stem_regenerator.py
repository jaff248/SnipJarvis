"""
Stem Regenerator - Selective regeneration for damaged audio stems.

This module provides the StemRegenerator class that orchestrates selective
regeneration of damaged stem regions using JASCO while preserving
high-quality portions of the original audio.

Features:
- Identify damaged time regions using quality detection
- Selective regeneration of only damaged portions
- Integration with JASCOGenerator for AI-powered regeneration
- Fallback to whole-stem regeneration when needed
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..profiling.quality_detector import (
    DamageLevel,
    QualityDetector,
    StemQualityReport,
)
from .jasco_generator import (
    GenerationConfig,
    GenerationResult,
    JASCOGenerator,
)
# BSSR Imports (lazy loaded where possible to avoid circular imports,
# but type checking needs them)
from .bssr import (
    MusicalStructureAnalyzer,
    BarAlignedChunker,
    AutoregressiveGenerator,
    StemSequentialOrchestrator,
)

logger = logging.getLogger(__name__)


@dataclass
class RegenerationRegion:
    """
    Represents a time region of a stem that needs regeneration.
    
    Attributes:
        start_time: Start time of region in seconds
        end_time: End time of region in seconds
        stem_name: Name of the stem (e.g., "vocals", "drums")
        damage_level: Severity of damage (from QualityDetector.DamageLevel)
        quality_report: Reference to the quality report for this stem
        confidence: Confidence score for this region (0-1)
    """
    start_time: float
    end_time: float
    stem_name: str
    damage_level: DamageLevel
    quality_report: Optional[StemQualityReport] = None
    confidence: float = 0.8
    
    @property
    def duration(self) -> float:
        """Get duration of the region in seconds."""
        return self.end_time - self.start_time
    
    @property
    def damage_score(self) -> float:
        """Get numeric damage score for sorting."""
        level_scores = {
            DamageLevel.GOOD: 0.0,
            DamageLevel.MINOR: 0.25,
            DamageLevel.MODERATE: 0.5,
            DamageLevel.SEVERE: 0.75,
            DamageLevel.CRITICAL: 1.0,
        }
        return level_scores.get(self.damage_level, 0.5)


@dataclass
class RegenerationPlan:
    """
    Complete plan for regenerating a stem.
    
    Attributes:
        stem_name: Name of the stem to regenerate
        original_audio: Original audio samples
        sample_rate: Sample rate of the audio
        regions: List of regions to regenerate
        use_whole_stem: Whether to regenerate entire stem
        quality_report: Quality report for the stem
        musical_profile: Musical profile for conditioning
    """
    stem_name: str
    original_audio: np.ndarray
    sample_rate: int
    regions: List[RegenerationRegion]
    use_whole_stem: bool = False
    quality_report: Optional[StemQualityReport] = None
    musical_profile: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RegenerationSummary:
    """
    Summary of a regeneration operation.
    
    Attributes:
        stem_name: Name of the regenerated stem
        original_duration: Duration of original audio in seconds
        regenerated_duration: Duration of regenerated audio in seconds
        regions_regenerated: Number of regions regenerated
        total_regenerated_time: Total time regenerated in seconds
        percent_regenerated: Percentage of audio that was regenerated
        use_whole_stem: Whether whole stem was regenerated
        regions: Details of each regenerated region
        success: Whether regeneration succeeded
        error: Error message if failed
    """
    stem_name: str
    original_duration: float
    regenerated_duration: float
    regions_regenerated: int
    total_regenerated_time: float
    percent_regenerated: float
    use_whole_stem: bool
    regions: List[Dict[str, Any]]
    success: bool = True
    error: Optional[str] = None


class StemRegenerator:
    """
    Orchestrates selective regeneration of damaged audio stems.
    
    This class identifies damaged regions in stems using quality detection,
    then uses JASCOGenerator to regenerate only the damaged portions while
    preserving high-quality regions.
    
    Features:
    - Automatic damage region detection
    - Region-based selective regeneration
    - Integration with musical profile extraction
    - Fallback to whole-stem regeneration
    - Comprehensive regeneration summaries
    
    Example:
        >>> regenerator = StemRegenerator(config, quality_detector)
        >>> regions = regenerator.identify_regions(audio, "drums", 44100)
        >>> if regions:
        ...     regenerated = regenerator.regenerate_regions(
        ...         audio, regions, profile, "drums"
        ...     )
    """
    
    # Thresholds for regeneration decisions
    REGION_COVERAGE_THRESHOLD = 0.5  # Regenerate whole stem if >50% damaged
    MIN_REGION_DURATION = 0.5  # Minimum region duration in seconds
    MAX_REGION_DURATION = 30.0  # Maximum region duration for JASCO
    
    def __init__(
        self,
        config: Optional[GenerationConfig] = None,
        quality_detector: Optional[QualityDetector] = None,
        use_bssr: bool = True,
    ):
        """
        Initialize the stem regenerator.
        
        Args:
            config: GenerationConfig for JASCOGenerator. If None, uses defaults.
            quality_detector: QualityDetector instance. If None, creates new one.
        """
        self.config = config or GenerationConfig()
        self.quality_detector = quality_detector or QualityDetector()
        self._generator: Optional[JASCOGenerator] = None
        self.use_bssr = use_bssr
        
        logger.info(f"Initialized StemRegenerator (BSSR enabled: {self.use_bssr})")
    
    @property
    def generator(self) -> JASCOGenerator:
        """Get or create the JASCO generator."""
        if self._generator is None:
            self._generator = JASCOGenerator(self.config)
        return self._generator
    
    def identify_regions(
        self,
        audio: np.ndarray,
        stem_name: str,
        sample_rate: int,
        artifact_metrics: Optional[Dict] = None,
    ) -> List[RegenerationRegion]:
        """
        Identify damaged time regions in a stem using quality detection.
        
        Uses the quality detector to analyze the audio and identify
        time regions that exhibit damage such as clipping, distortion,
        noise, or spectral issues.
        
        Args:
            audio: Audio samples (float32, range [-1, 1])
            stem_name: Name of the stem (e.g., "vocals", "drums")
            sample_rate: Sample rate in Hz
            artifact_metrics: Optional artifact metrics from QualityMetrics
            
        Returns:
            List of RegenerationRegion sorted by damage severity
        """
        logger.info(f"Identifying regions for stem '{stem_name}'")
        
        # Analyze quality of the stem
        quality_report = self.quality_detector.analyze(
            audio=audio,
            sample_rate=sample_rate,
            artifact_metrics=artifact_metrics,
        )
        
        regions: List[RegenerationRegion] = []
        
        # Use regions from quality report if available
        if quality_report.regenerate_regions:
            for start_time, end_time in quality_report.regenerate_regions:
                # Filter out very short regions
                if end_time - start_time >= self.MIN_REGION_DURATION:
                    region = RegenerationRegion(
                        start_time=start_time,
                        end_time=end_time,
                        stem_name=stem_name,
                        damage_level=quality_report.damage_level,
                        quality_report=quality_report,
                        confidence=quality_report.confidence,
                    )
                    regions.append(region)
        
        # For critical damage, add whole stem as one region
        if quality_report.damage_level == DamageLevel.CRITICAL:
            duration = len(audio) / sample_rate
            regions = [
                RegenerationRegion(
                    start_time=0.0,
                    end_time=duration,
                    stem_name=stem_name,
                    damage_level=DamageLevel.CRITICAL,
                    quality_report=quality_report,
                    confidence=quality_report.confidence,
                )
            ]
            logger.info(f"Critical damage detected - whole stem marked for regeneration")
        
        # Sort by damage severity (most damaged first)
        regions.sort(key=lambda r: r.damage_score, reverse=True)
        
        logger.info(f"Identified {len(regions)} regions for regeneration")
        return regions
    
    def regenerate_regions(
        self,
        audio: np.ndarray,
        regions: List[RegenerationRegion],
        musical_profile: Dict[str, Any],
        stem_type: str,
        sample_rate: int = 44100,
        callbacks: Optional[Any] = None,
    ) -> np.ndarray:
        """
        Regenerate damaged regions in a stem using JASCO.
        
        For each damaged region, extracts the appropriate musical conditioning
        from the profile and generates replacement audio using JASCO. The
        regenerated regions are then blended back into the original audio.
        
        Args:
            audio: Original audio samples
            regions: List of regions to regenerate
            musical_profile: Musical profile with chords, tempo, key, etc.
            stem_type: Type of stem ("vocals", "drums", "bass", "other")
            sample_rate: Sample rate in Hz
            callbacks: Optional progress callbacks
            
        Returns:
            Regenerated audio with same duration as input
        """
        if not regions:
            logger.warning("No regions to regenerate, returning original audio")
            return audio.copy()
        
        logger.info(f"Regenerating {len(regions)} regions for stem '{stem_type}'")
        
        # Check if we should regenerate the whole stem instead
        if self.should_regenerate_entire_stem(regions, audio, sample_rate):
            logger.info("Regions cover >50% of stem - regenerating entire stem")
            
            # Use BSSR for long audio if enabled
            duration = len(audio) / sample_rate
            if self.use_bssr and duration > 30.0:
                logger.info(f"Audio duration {duration:.1f}s > 30s - using BSSR pipeline")
                return self._regenerate_with_bssr(
                    audio, musical_profile, stem_type, sample_rate, callbacks
                )
            
            return self._regenerate_entire_stem(
                audio, musical_profile, stem_type, sample_rate, callbacks
            )
        
        # Prepare for blending
        from .blender import Blender
        
        blender = Blender()
        regenerated_segments: List[Tuple[np.ndarray, float, float]] = []
        
        # Process each region
        for i, region in enumerate(regions):
            try:
                logger.info(
                    f"Processing region {i+1}/{len(regions)}: "
                    f"{region.start_time:.2f}s - {region.end_time:.2f}s"
                )
                
                # Extract region from original audio
                start_sample = int(region.start_time * sample_rate)
                end_sample = int(region.end_time * sample_rate)
                region_audio = audio[start_sample:end_sample]
                
                # Regenerate this region
                regenerated = self._regenerate_region(
                    region=region,
                    original_region=region_audio,
                    musical_profile=musical_profile,
                    stem_type=stem_type,
                    sample_rate=sample_rate,
                    callbacks=callbacks,
                )
                
                if regenerated is not None and len(regenerated) > 0:
                    regenerated_segments.append((regenerated, region.start_time, region.end_time))
                    
            except Exception as e:
                logger.error(f"Failed to regenerate region {i}: {e}")
                # Continue with other regions
        
        # Blend regenerated segments into original
        if regenerated_segments:
            result = blender.blend_regions(audio, regenerated_segments, sample_rate)
            logger.info(f"Blended {len(regenerated_segments)} regenerated segments")
            return result
        else:
            logger.warning("No successful regenerations, returning original audio")
            return audio.copy()
    
    def _regenerate_region(
        self,
        region: RegenerationRegion,
        original_region: np.ndarray,
        musical_profile: Dict[str, Any],
        stem_type: str,
        sample_rate: int,
        callbacks: Optional[Any] = None,
    ) -> Optional[np.ndarray]:
        """
        Regenerate a single region using JASCO.
        
        Args:
            region: RegenerationRegion to process
            original_region: Audio samples for the region
            musical_profile: Musical conditioning
            stem_type: Type of stem
            sample_rate: Sample rate
            callbacks: Progress callbacks
            
        Returns:
            Regenerated audio or None if failed
        """
        duration = region.end_time - region.start_time
        
        # Check duration limits
        if duration > self.MAX_REGION_DURATION:
            logger.warning(f"Region too long ({duration:.1f}s), trimming to {self.MAX_REGION_DURATION}s")
            duration = self.MAX_REGION_DURATION
            region.end_time = region.start_time + duration
        
        # Extract conditioning from profile
        chords = musical_profile.get("chords", [])
        tempo = musical_profile.get("tempo", 120.0)
        key = musical_profile.get("key", "C")
        melody = musical_profile.get("melody")
        drums = musical_profile.get("drums")
        timbre = musical_profile.get("timbre")
        articulation = musical_profile.get("articulation")
        
        # Generate with conditioning
        result = self.generator.generate(
            chords_timeline=chords,
            tempo=tempo,
            key=key,
            stem_type=stem_type,
            duration=int(duration),
            callbacks=callbacks,
            melody=melody,
            drums=drums,
            timbre=timbre,
            articulation=articulation,
        )
        
        if result.success and result.audio is not None:
            logger.info(f"Successfully regenerated region ({result.duration:.1f}s)")
            # Resample if needed
            if result.sample_rate != sample_rate:
                regenerated = self._resample_audio(result.audio, result.sample_rate, sample_rate)
            else:
                regenerated = result.audio
            
            # Match length to original region
            target_length = len(original_region)
            if len(regenerated) > target_length:
                regenerated = regenerated[:target_length]
            elif len(regenerated) < target_length:
                # Pad with silence
                padding = np.zeros(target_length - len(regenerated), dtype=np.float32)
                regenerated = np.concatenate([regenerated, padding])
            
            return regenerated
        else:
            logger.warning(f"Region regeneration failed: {result.error}")
            return None
    
    def _regenerate_with_bssr(
        self,
        audio: np.ndarray,
        musical_profile: Dict[str, Any],
        stem_type: str,
        sample_rate: int,
        callbacks: Optional[Any] = None,
    ) -> np.ndarray:
        """
        Regenerate entire stem using BSSR pipeline for long-form audio.
        """
        try:
            # Initialize BSSR components
            analyzer = MusicalStructureAnalyzer()
            structure = analyzer.analyze(audio, sample_rate)
            
            # Create orchestrator
            # We wrap our existing generator
            autoregressive_gen = AutoregressiveGenerator(self.generator)
            chunker = BarAlignedChunker()
            
            orchestrator = StemSequentialOrchestrator(
                chunker=chunker,
                generator=autoregressive_gen
            )
            
            # BSSR expects a dictionary of stems, but we are regenerating one.
            # We pass empty dict for original stems if we don't have others,
            # but we force regeneration of this stem type.
            original_stems = {stem_type: audio}
            
            # Execute BSSR
            # This returns a dict {stem_type: audio}
            results = orchestrator.regenerate_all_stems(
                original_stems=original_stems,
                musical_profile=musical_profile,
                structure=structure,
                callbacks=callbacks
            )
            
            return results.get(stem_type, audio)
            
        except Exception as e:
            logger.error(f"BSSR regeneration failed: {e}")
            # Fallback to standard truncation method
            logger.warning("Falling back to standard regeneration (truncated)")
            return self._regenerate_entire_stem(
                audio, musical_profile, stem_type, sample_rate, callbacks
            )

    def _regenerate_entire_stem(
        self,
        audio: np.ndarray,
        musical_profile: Dict[str, Any],
        stem_type: str,
        sample_rate: int,
        callbacks: Optional[Any] = None,
    ) -> np.ndarray:
        """
        Regenerate an entire stem as a fallback.
        
        Args:
            audio: Original audio
            musical_profile: Musical conditioning
            stem_type: Type of stem
            sample_rate: Sample rate
            callbacks: Progress callbacks
            
        Returns:
            Regenerated audio
        """
        duration = len(audio) / sample_rate
        
        # Limit duration to JASCO maximum
        max_duration = 30
        if duration > max_duration:
            logger.warning(f"Stem too long ({duration:.1f}s), truncating to {max_duration}s")
            audio = audio[: int(max_duration * sample_rate)]
            duration = max_duration
        
        # Extract conditioning
        chords = musical_profile.get("chords", [])
        tempo = musical_profile.get("tempo", 120.0)
        key = musical_profile.get("key", "C")
        
        # Generate entire stem
        result = self.generator.generate(
            chords_timeline=chords,
            tempo=tempo,
            key=key,
            stem_type=stem_type,
            duration=int(duration),
            callbacks=callbacks,
        )
        
        if result.success and result.audio is not None:
            if result.sample_rate != sample_rate:
                regenerated = self._resample_audio(result.audio, result.sample_rate, sample_rate)
            else:
                regenerated = result.audio
            
            # Pad or trim to match original
            target_length = len(audio)
            if len(regenerated) > target_length:
                regenerated = regenerated[:target_length]
            elif len(regenerated) < target_length:
                padding = np.zeros(target_length - len(regenerated), dtype=np.float32)
                regenerated = np.concatenate([regenerated, padding])
            
            return regenerated
        else:
            logger.error(f"Whole-stem regeneration failed: {result.error}")
            return audio.copy()
    
    def _resample_audio(
        self,
        audio: np.ndarray,
        src_rate: int,
        dst_rate: int,
    ) -> np.ndarray:
        """Resample audio from src_rate to dst_rate."""
        try:
            import librosa
            
            if src_rate == dst_rate:
                return audio
            
            return librosa.resample(
                audio,
                orig_sr=src_rate,
                target_sr=dst_rate,
            )
        except Exception as e:
            logger.warning(f"Resampling failed: {e}, using original sample rate")
            return audio
    
    def should_regenerate_entire_stem(
        self,
        regions: List[RegenerationRegion],
        audio: np.ndarray,
        sample_rate: int,
    ) -> bool:
        """
        Determine if whole-stem regeneration is better than selective.
        
        Regenerates the entire stem if:
        - Regions cover more than 50% of the stem duration
        - Regions overlap significantly
        - Critical damage is detected
        
        Args:
            regions: List of regions to regenerate
            audio: Original audio
            sample_rate: Sample rate
            
        Returns:
            True if whole-stem regeneration is recommended
        """
        if not regions:
            return False
        
        audio_duration = len(audio) / sample_rate
        
        # Calculate total coverage
        covered_duration = 0.0
        for region in regions:
            covered_duration += region.duration
        
        coverage_ratio = covered_duration / audio_duration
        
        # Check for significant coverage
        if coverage_ratio > self.REGION_COVERAGE_THRESHOLD:
            logger.info(
                f"Coverage ratio {coverage_ratio:.1%} > threshold {self.REGION_COVERAGE_THRESHOLD:.0%} "
                f"- recommending whole-stem regeneration"
            )
            return True
        
        # Check for critical damage
        for region in regions:
            if region.damage_level == DamageLevel.CRITICAL:
                logger.info("Critical damage detected - recommending whole-stem regeneration")
                return True
        
        return False
    
    def get_regeneration_summary(
        self,
        regions: List[RegenerationRegion],
        audio: np.ndarray,
        sample_rate: int,
        use_whole_stem: bool = False,
    ) -> RegenerationSummary:
        """
        Generate a summary of regeneration plan.
        
        Args:
            regions: List of regions to regenerate
            audio: Original audio
            sample_rate: Sample rate
            use_whole_stem: Whether whole stem will be regenerated
            
        Returns:
            RegenerationSummary with details
        """
        audio_duration = len(audio) / sample_rate
        
        if use_whole_stem:
            total_regenerated = audio_duration
            percent = 100.0
        else:
            total_regenerated = sum(r.duration for r in regions)
            percent = (total_regenerated / audio_duration * 100) if audio_duration > 0 else 0
        
        region_details = []
        for region in regions:
            region_details.append({
                "start_time": region.start_time,
                "end_time": region.end_time,
                "duration": region.duration,
                "damage_level": region.damage_level.value,
                "confidence": region.confidence,
            })
        
        return RegenerationSummary(
            stem_name=regions[0].stem_name if regions else "unknown",
            original_duration=audio_duration,
            regenerated_duration=audio_duration,
            regions_regenerated=len(regions),
            total_regenerated_time=total_regenerated,
            percent_regenerated=percent,
            use_whole_stem=use_whole_stem,
            regions=region_details,
            success=True,
        )
    
    def create_regeneration_plan(
        self,
        audio: np.ndarray,
        stem_name: str,
        sample_rate: int,
        musical_profile: Dict[str, Any],
        artifact_metrics: Optional[Dict] = None,
    ) -> RegenerationPlan:
        """
        Create a complete regeneration plan for a stem.
        
        Args:
            audio: Audio samples
            stem_name: Name of the stem
            sample_rate: Sample rate
            musical_profile: Musical profile for conditioning
            artifact_metrics: Optional artifact metrics
            
        Returns:
            RegenerationPlan with regions and configuration
        """
        # Identify damaged regions
        regions = self.identify_regions(
            audio=audio,
            stem_name=stem_name,
            sample_rate=sample_rate,
            artifact_metrics=artifact_metrics,
        )
        
        # Analyze quality
        quality_report = self.quality_detector.analyze(
            audio=audio,
            sample_rate=sample_rate,
            artifact_metrics=artifact_metrics,
        )
        
        # Determine if whole-stem regeneration is needed
        use_whole_stem = self.should_regenerate_entire_stem(regions, audio, sample_rate)
        
        # If using whole stem, replace regions with single full-span region
        if use_whole_stem:
            duration = len(audio) / sample_rate
            regions = [
                RegenerationRegion(
                    start_time=0.0,
                    end_time=duration,
                    stem_name=stem_name,
                    damage_level=DamageLevel.CRITICAL,
                    quality_report=quality_report,
                    confidence=quality_report.confidence,
                )
            ]
        
        return RegenerationPlan(
            stem_name=stem_name,
            original_audio=audio,
            sample_rate=sample_rate,
            regions=regions,
            use_whole_stem=use_whole_stem,
            quality_report=quality_report,
            musical_profile=musical_profile,
        )
    
    def execute_regeneration_plan(
        self,
        plan: RegenerationPlan,
        callbacks: Optional[Any] = None,
    ) -> Tuple[np.ndarray, RegenerationSummary]:
        """
        Execute a regeneration plan.
        
        Args:
            plan: RegenerationPlan to execute
            callbacks: Optional progress callbacks
            
        Returns:
            Tuple of (regenerated audio, summary)
        """
        if not plan.regions:
            logger.warning("No regions in plan, returning original audio")
            return plan.original_audio.copy(), RegenerationSummary(
                stem_name=plan.stem_name,
                original_duration=len(plan.original_audio) / plan.sample_rate,
                regenerated_duration=len(plan.original_audio) / plan.sample_rate,
                regions_regenerated=0,
                total_regenerated_time=0.0,
                percent_regenerated=0.0,
                use_whole_stem=False,
                regions=[],
                success=True,
            )
        
        # Regenerate regions
        regenerated = self.regenerate_regions(
            audio=plan.original_audio,
            regions=plan.regions,
            musical_profile=plan.musical_profile,
            stem_type=plan.stem_name,
            sample_rate=plan.sample_rate,
            callbacks=callbacks,
        )
        
        # Generate summary
        summary = self.get_regeneration_summary(
            regions=plan.regions,
            audio=plan.original_audio,
            sample_rate=plan.sample_rate,
            use_whole_stem=plan.use_whole_stem,
        )
        
        return regenerated, summary


# =============================================================================
# Convenience Functions
# =============================================================================

def create_regenerator(
    device: str = "mps",
    stem_type: str = "drums",
    duration: int = 10,
) -> StemRegenerator:
    """
    Create a configured StemRegenerator.
    
    Args:
        device: Device for inference
        stem_type: Default stem type
        duration: Default duration for generation
        
    Returns:
        Configured StemRegenerator instance
    """
    config = GenerationConfig(
        device=device,
        stem_type=stem_type,
        duration=duration,
    )
    return StemRegenerator(config)


def regenerate_stem(
    audio: np.ndarray,
    sample_rate: int,
    stem_name: str,
    musical_profile: Dict[str, Any],
    quality_detector: Optional[QualityDetector] = None,
) -> Tuple[np.ndarray, RegenerationSummary]:
    """
    Convenience function to regenerate a stem.
    
    Args:
        audio: Audio samples
        sample_rate: Sample rate
        stem_name: Name of the stem
        musical_profile: Musical profile for conditioning
        quality_detector: Optional quality detector instance
        
    Returns:
        Tuple of (regenerated audio, summary)
    """
    detector = quality_detector or QualityDetector()
    regenerator = StemRegenerator(quality_detector=detector)
    
    plan = regenerator.create_regeneration_plan(
        audio=audio,
        stem_name=stem_name,
        sample_rate=sample_rate,
        musical_profile=musical_profile,
    )
    
    return regenerator.execute_regeneration_plan(plan)
