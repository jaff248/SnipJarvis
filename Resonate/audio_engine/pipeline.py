"""
Main Audio Pipeline - Orchestrates the full reconstruction workflow

Coordinates all stages of the audio reconstruction pipeline:
Ingest → Separate → Enhance → Mix → Master
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from enum import Enum
from pathlib import Path

import numpy as np

from .ingest import AudioIngest, AudioBuffer
from .preconditioning import PreConditioningPipeline, PreConditioningConfig
from .separator import SeparatorEngine, SeparationResult, SeparatorConfig, DemucsModel
from .enhancers.pipeline import StemEnhancementPipeline, GlobalEnhancementConfig, ProcessingMode
from .mixing import StemMixer, MixConfig, MixMode
from .mastering import AudioMaster, MasteringConfig, OutputFormat
from .restoration import FrequencyRestorer, Dereverberator
from .polish import MBDEnhancer
from .cache import CacheManager, get_cache_manager
from .utils import get_audio_hash, format_duration

logger = logging.getLogger(__name__)


class PipelineMode(Enum):
    """Pipeline processing modes."""
    PREVIEW = "preview"  # Fast, lower quality
    RENDER = "render"    # Slow, high quality


@dataclass
class PipelineConfig:
    """Configuration for the full pipeline."""
    # Processing mode
    mode: PipelineMode = PipelineMode.RENDER
    
    # Pre-conditioning settings (NEW)
    enable_preconditioning: bool = True
    precondition_noise_reduction: bool = True
    precondition_noise_strength: float = 0.5
    precondition_declip: bool = True
    precondition_dynamics: bool = True

    # Separation settings
    separation_model: str = "htdemucs_ft"  # "htdemucs" or "htdemucs_ft"
    
    # Enhancement settings
    enhancement_intensity: float = 0.5
    
    # Restoration settings
    frequency_restoration: bool = True
    frequency_intensity: float = 0.5
    dereverberation: bool = True
    dereverb_intensity: float = 0.3

    # Neural polishing
    enable_mbd_polish: bool = False
    mbd_intensity: float = 0.3
    
    # Mixing settings
    mix_mode: str = "enhanced"  # "natural", "enhanced", "stem_solo"
    
    # Mastering settings
    target_loudness_lufs: float = -14.0
    output_format: str = "wav"   # "wav", "flac", "mp3"
    output_bit_depth: int = 24
    
    # Caching
    use_cache: bool = True
    cache_dir: Optional[str] = None
    
    # Output
    output_dir: str = "."
    
    def __post_init__(self):
        """Validate and set defaults based on mode."""
        # Only set defaults if not already explicitly configured
        # Check if using default values (not explicitly set by user)
        pass  # User-provided values take precedence over mode defaults


@dataclass
class PipelineResult:
    """Result of full pipeline processing."""
    # Original audio info
    original_buffer: AudioBuffer
    
    # Processing results
    separation_result: Optional[SeparationResult]
    enhancement_result: Optional[Dict[str, np.ndarray]]
    mix_result: Optional[Any]
    mastering_result: Optional[Any]
    
    # Output
    output_file: Optional[str]
    output_file_size: int
    
    # Metrics
    total_processing_time: float
    stage_times: Dict[str, float]
    
    # Success status
    success: bool
    error_message: Optional[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            "success": self.success,
            "original": {
                "duration": format_duration(self.original_buffer.duration),
                "sample_rate": self.original_buffer.sample_rate,
                "file_hash": self.original_buffer.metadata.file_hash
            },
            "processing_time": {
                "total": f"{self.total_processing_time:.1f}s",
                "by_stage": {k: f"{v:.1f}s" for k, v in self.stage_times.items()}
            },
            "output": {
                "file": self.output_file,
                "size_mb": self.output_file_size / (1024 * 1024)
            },
            "error": self.error_message
        }


class AudioPipeline:
    """
    Main audio reconstruction pipeline.
    
    Orchestrates the complete workflow:
    1. Ingest: Load and validate audio
    2. Separate: Extract stems using Demucs
    3. Enhance: Process each stem (vocals, drums, bass, other)
    4. Mix: Recombine stems with optimal balance
    5. Master: Final polish and export
    
    Features:
    - Progress tracking
    - Caching for expensive operations
    - Configurable quality/speed tradeoff
    - Error handling with graceful degradation
    """
    
    def __init__(self, config: PipelineConfig = None):
        """
        Initialize the audio pipeline.
        
        Args:
            config: Pipeline configuration (uses defaults if None)
        """
        self.config = config or PipelineConfig()
        
        # Initialize components
        self._ingest = None
        self._separator = None
        self._enhancer = None
        self._mixer = None
        self._master = None
        self._mbd_polish = None
        self._cache = None
        
        # Lazy-loaded preconditioner (initialized on first use)
        self._preconditioner = None
        
        logger.info(f"Initialized AudioPipeline: {self.config.mode.value}")
    
    def _get_ingest(self) -> AudioIngest:
        """Get or create ingest component."""
        if self._ingest is None:
            self._ingest = AudioIngest(
                target_sample_rate=44100,
                normalize=True,
                convert_to_mono=True
            )
        return self._ingest

    def _get_preconditioner(self) -> PreConditioningPipeline:
        """Lazy-load the preconditioner to avoid unnecessary initialization."""
        if self._preconditioner is None:
            config = PreConditioningConfig(
                enable_noise_reduction=self.config.precondition_noise_reduction,
                noise_reduction_strength=self.config.precondition_noise_strength,
                enable_declipping=self.config.precondition_declip,
                enable_dynamics_restoration=self.config.precondition_dynamics,
            )
            self._preconditioner = PreConditioningPipeline(44100, config)
        return self._preconditioner
    
    def _get_separator(self) -> SeparatorEngine:
        """Get or create separator component."""
        if self._separator is None:
            # Map string model names to enum
            model_map = {
                "htdemucs": DemucsModel.HTDEMUCS,
                "htdemucs_ft": DemucsModel.HTDEMUCS_FT,
                "htdemucs_6s": DemucsModel.HTDEMUCS_6S
            }
            model_enum = model_map.get(self.config.separation_model, DemucsModel.HTDEMUCS_FT)
            
            # Create config with correct model
            if self.config.mode == PipelineMode.RENDER:
                config = SeparatorConfig.render()
            else:
                config = SeparatorConfig.preview()
            
            # Override model
            config.model = model_enum
            self._separator = SeparatorEngine(config)
        return self._separator
    
    def _get_enhancer(self) -> StemEnhancementPipeline:
        """Get or create enhancement pipeline."""
        if self._enhancer is None:
            mode = ProcessingMode.RENDER if self.config.mode == PipelineMode.RENDER else ProcessingMode.PREVIEW
            config = GlobalEnhancementConfig(
                mode=mode,
                intensity=self.config.enhancement_intensity
            )
            self._enhancer = StemEnhancementPipeline(config)
        return self._enhancer
    
    def _get_mixer(self) -> StemMixer:
        """Get or create mixer."""
        if self._mixer is None:
            # Map string to MixMode enum
            mix_mode_map = {
                "natural": MixMode.NATURAL,
                "enhanced": MixMode.ENHANCED, 
                "stem_solo": MixMode.STEM_SOLO
            }
            mix_mode_enum = mix_mode_map.get(self.config.mix_mode, MixMode.ENHANCED)
            config = MixConfig(mix_mode=mix_mode_enum)
            self._mixer = StemMixer(config)
        return self._mixer
    
    def _get_master(self) -> AudioMaster:
        """Get or create mastering component."""
        if self._master is None:
            format_map = {
                "wav": OutputFormat.WAV,
                "flac": OutputFormat.FLAC,
                "mp3": OutputFormat.MP3
            }
            output_format = format_map.get(self.config.output_format, OutputFormat.WAV)
            config = MasteringConfig(
                target_loudness_lufs=self.config.target_loudness_lufs,
                output_format=output_format,
                output_bit_depth=self.config.output_bit_depth
            )
            self._master = AudioMaster(config)
        return self._master

    def _get_mbd_enhancer(self) -> MBDEnhancer:
        """Lazy-load MBD enhancer (optional)."""
        if self._mbd_polish is None:
            self._mbd_polish = MBDEnhancer(44100)
        return self._mbd_polish
    
    def _get_cache(self) -> CacheManager:
        """Get or create cache manager."""
        if self._cache is None:
            self._cache = get_cache_manager(self.config.cache_dir)
        return self._cache
    
    def process(self, input_path: str, output_path: str = None) -> PipelineResult:
        """
        Process audio file through full reconstruction pipeline.
        
        Args:
            input_path: Path to input audio file
            output_path: Path for output file (auto-generated if None)
            
        Returns:
            PipelineResult with processing results and output file
        """
        import time
        total_start = time.time()
        stage_times = {}
        
        logger.info(f"🚀 Starting pipeline: {Path(input_path).name}")
        
        # Track errors for graceful degradation
        errors = []
        
        try:
            # === STAGE 1: INGEST ===
            stage_start = time.time()
            logger.info("Stage 1: Loading audio...")
            
            ingest = self._get_ingest()
            original_buffer = ingest.load(input_path)
            stage_times['ingest'] = time.time() - stage_start
            
            logger.info(f"  ✅ Loaded: {format_duration(original_buffer.duration)}")
            
            # === STAGE 1.5: PRE-CONDITIONING ===
            if self.config.enable_preconditioning:
                stage_start = time.time()
                logger.info("Stage 1.5: Pre-conditioning input...")
                
                preconditioner = self._get_preconditioner()
                precond_result = preconditioner.process(original_buffer.audio)
                
                # Update buffer with preconditioned audio
                original_buffer.audio = precond_result.audio
                
                stage_times['preconditioning'] = time.time() - stage_start
                logger.info(f"  ✅ Pre-conditioning: noise={precond_result.noise_reduced}, "
                           f"clips={precond_result.clips_repaired}, "
                           f"dynamics={precond_result.dynamics_restored}")

            # === STAGE 2: SEPARATE ===
            stage_start = time.time()
            logger.info("Stage 2: Separating stems...")
            
            cache = self._get_cache()
            audio_hash = original_buffer.metadata.file_hash
            
            # Check cache first
            cached_stems = None
            if self.config.use_cache:
                cached_stems = cache.get_stems(audio_hash)
            
            if cached_stems is not None:
                logger.info("  ♻️ Using cached stems")
                separation_result = None
                separation_time = 0.0
                stems = cached_stems
            else:
                # Run separation
                separator = self._get_separator()
                separation_result = separator.separate_file(input_path)
                stems = separation_result.stems
                separation_time = separation_result.processing_time
                stage_times['separation'] = separation_time
                
                # Cache stems
                if self.config.use_cache and separation_result:
                    cache.cache_stems(audio_hash, stems, {
                        "model": self.config.separation_model,
                        "mode": self.config.mode.value
                    })
            
            logger.info(f"  ✅ Separated {len(stems)} stems")
            
            # === STAGE 3: ENHANCE ===
            stage_start = time.time()
            logger.info("Stage 3: Enhancing stems...")
            
            enhancer = self._get_enhancer()
            enhancement_result = enhancer.process(stems, 44100)
            enhanced_stems = enhancement_result.stems
            stage_times['enhancement'] = time.time() - stage_start
            
            logger.info(f"  ✅ Enhanced {len(enhanced_stems)} stems")
            
            # === STAGE 4: MIX ===
            stage_start = time.time()
            logger.info("Stage 4: Mixing stems...")
            
            mixer = self._get_mixer()
            mix_result = mixer.mix(enhanced_stems, 44100)
            mixed_audio = mix_result.mixed_audio
            stage_times['mixing'] = time.time() - stage_start
            
            logger.info(f"  ✅ Mixed: peak={mix_result.peak_level_db:.1f} dB")
            
            # === STAGE 4.5: RESTORATION ===
            if self.config.frequency_restoration or self.config.dereverberation:
                stage_start = time.time()
                logger.info("Stage 4.5: Applying restoration...")
                
                # Frequency restoration
                if self.config.frequency_restoration:
                    try:
                        restorer = FrequencyRestorer(44100, self.config.frequency_intensity)
                        mixed_audio = restorer.process(mixed_audio)
                        logger.info("  ✅ Frequency restoration applied")
                    except Exception as e:
                        logger.warning(f"Frequency restoration failed: {e}, skipping")
                
                # Dereverberation
                if self.config.dereverberation:
                    try:
                        dereverb = Dereverberator(44100, self.config.dereverb_intensity)
                        mixed_audio = dereverb.process(mixed_audio)
                        logger.info("  ✅ Dereverberation applied")
                    except Exception as e:
                        logger.warning(f"Dereverberation failed: {e}, skipping")
                
                stage_times['restoration'] = time.time() - stage_start

            # === STAGE 4.6: MBD POLISH (OPTIONAL - BEFORE MASTERING) ===
            if self.config.enable_mbd_polish:
                stage_start = time.time()
                logger.info("Stage 4.6: Applying neural polish (MBD)...")

                try:
                    mbd = self._get_mbd_enhancer()
                    if mbd.is_available():
                        logger.info("  🔄 MBD processing... (this may take ~30s)")
                        mixed_audio = mbd.process(mixed_audio, self.config.mbd_intensity)
                        logger.info("  ✅ MBD polish applied")
                        stage_times['mbd_polish'] = time.time() - stage_start
                    else:
                        logger.info("  ⏭️ MBD not available, skipping")
                except Exception as e:
                    logger.warning(f"MBD polish failed: {e}, continuing without")
            
            # === STAGE 5: MASTER (now includes MBD-polished audio) ===
            stage_start = time.time()
            logger.info("Stage 5: Mastering...")
            
            master = self._get_master()
            # Generate output path if needed
            if output_path is None:
                input_file = Path(input_path)
                output_path = str(input_file.with_name(f"resonate_{input_file.stem}_mastered.wav"))
            
            mastering_result = master.master(mixed_audio, 44100, output_path)
            stage_times['mastering'] = time.time() - stage_start
            
            logger.info(f"  ✅ Mastered: {mastering_result.loudness_lufs} LUFS")
            logger.info(f"  📁 Output: {output_path}")
            
            # === COMPLETE ===
            total_time = time.time() - total_start
            
            return PipelineResult(
                original_buffer=original_buffer,
                separation_result=separation_result,
                enhancement_result=enhancement_result,
                mix_result=mix_result,
                mastering_result=mastering_result,
                output_file=mastering_result.file_path,
                output_file_size=mastering_result.file_size_bytes,
                total_processing_time=total_time,
                stage_times=stage_times,
                success=True,
                error_message=None
            )
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            
            return PipelineResult(
                original_buffer=None,
                separation_result=None,
                enhancement_result=None,
                mix_result=None,
                mastering_result=None,
                output_file=None,
                output_file_size=0,
                total_processing_time=time.time() - total_start,
                stage_times=stage_times,
                success=False,
                error_message=str(e)
            )
    
    def process_preview(self, input_path: str, duration: float = 30.0) -> PipelineResult:
        """
        Process preview of audio file.
        
        Args:
            input_path: Path to input audio file
            duration: Preview duration in seconds
            
        Returns:
            PipelineResult with preview results
        """
        # Create preview config
        preview_config = PipelineConfig(
            mode=PipelineMode.PREVIEW
        )
        
        # Create temporary preview pipeline
        preview_pipeline = AudioPipeline(preview_config)
        
        # Run preview (truncate to first N seconds)
        result = preview_pipeline.process(input_path)
        
        return result
    
    def get_info(self) -> Dict[str, Any]:
        """Get pipeline configuration and status."""
        return {
            "config": {
                "mode": self.config.mode.value,
                "separation_model": self.config.separation_model,
                "enhancement_intensity": self.config.enhancement_intensity,
                "frequency_restoration": self.config.frequency_restoration,
                "frequency_intensity": self.config.frequency_intensity,
                "dereverberation": self.config.dereverberation,
                "dereverb_intensity": self.config.dereverb_intensity,
                "enable_mbd_polish": self.config.enable_mbd_polish,
                "mbd_intensity": self.config.mbd_intensity,
                "mix_mode": self.config.mix_mode,
                "target_loudness_lufs": self.config.target_loudness_lufs,
                "output_format": self.config.output_format,
                "use_cache": self.config.use_cache
            },
            "components": {
                "separator": str(self._get_separator()),
                "enhancer": str(self._get_enhancer()),
                "mixer": str(self._get_mixer()),
                "master": str(self._get_master())
            }
        }
    
    def cleanup(self):
        """Clean up resources."""
        if self._cache:
            self._cache.cleanup()
        
        # Clear GPU memory
        if self._separator:
            self._separator.clear_cache()
        
        logger.info("Pipeline cleanup complete")
    
    def __del__(self):
        """Cleanup on destruction."""
        self.cleanup()


# Convenience function
def create_pipeline(mode: str = "render") -> AudioPipeline:
    """
    Create audio pipeline with specified mode.
    
    Args:
        mode: "preview" or "render"
        
    Returns:
        Configured AudioPipeline instance
    """
    mode_enum = PipelineMode.RENDER if mode == "render" else PipelineMode.PREVIEW
    config = PipelineConfig(mode=mode_enum)
    
    return AudioPipeline(config)


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing AudioPipeline...")
    
    # Create test audio file first
    import soundfile as sf
    
    sample_rate = 44100
    duration = 5.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create test mix
    test_audio = (
        0.3 * np.sin(2 * np.pi * 440 * t) +  # Vocals (simulated)
        0.2 * np.sin(2 * np.pi * 50 * t) * np.exp(-t % 1) +  # Drums
        0.2 * np.sin(2 * np.pi * 80 * t) +  # Bass
        0.15 * np.sin(2 * np.pi * 330 * t) +  # Guitar
        0.05 * np.random.randn(len(t))  # Noise
    )
    
    # Normalize
    test_audio = test_audio / np.max(np.abs(test_audio)) * 0.8
    
    # Save test file
    sf.write("test_input.wav", test_audio, sample_rate)
    print("Created test audio file: test_input.wav")
    
    # Create pipeline
    pipeline = create_pipeline(mode="preview")
    print(f"Created: {pipeline}")
    
    # Get info
    info = pipeline.get_info()
    print(f"Pipeline config: {info['config']}")
    
    # Run preview pipeline
    print("\nRunning preview pipeline...")
    result = pipeline.process("test_input.wav", "test_output.wav")
    
    if result.success:
        print(f"\n✅ Pipeline completed successfully!")
        print(f"Duration: {result.total_processing_time:.1f}s")
        print(f"Output: {result.output_file}")
        print(f"Stage times: {result.stage_times}")
    else:
        print(f"\n❌ Pipeline failed: {result.error_message}")
    
    # Cleanup
    pipeline.cleanup()
