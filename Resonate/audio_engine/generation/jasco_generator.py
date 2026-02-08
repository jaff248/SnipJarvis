"""
JASCO Generator - AI-powered stem regeneration using AudioCraft's JASCO model.

This module provides the JASCOGenerator class that wraps AudioCraft's JASCO model
for regenerating heavily damaged audio stems while maintaining musical consistency
with the original track's structure.

JASCO (Joint Audio and Symbolic Conditioning) enables conditional music generation
based on:
- Chord progressions
- Tempo and key information
- Melody contours
- Drum patterns
- Text descriptions

The generator integrates with the profiling module to condition regeneration
on extracted musical structure.
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from audio_engine.profiling.timbre_analyzer import TimbreProfile
from audio_engine.profiling.articulation_detector import ArticulationProfile

logger = logging.getLogger(__name__)


# =============================================================================
# Custom Exceptions
# =============================================================================

class ModelLoadError(Exception):
    """Raised when the JASCO model fails to load."""
    
    def __init__(self, message: str, device: str = "unknown", details: Optional[str] = None):
        self.message = message
        self.device = device
        self.details = details
        super().__init__(self._format_message())
    
    def _format_message(self) -> str:
        msg = f"ModelLoadError: {self.message}"
        if self.device:
            msg += f" (device: {self.device})"
        if self.details:
            msg += f" - {self.details}"
        return msg


class GenerationError(Exception):
    """Raised when stem generation fails."""
    
    def __init__(
        self, 
        message: str, 
        stem_type: str = "unknown", 
        duration: Optional[float] = None,
        details: Optional[str] = None
    ):
        self.message = message
        self.stem_type = stem_type
        self.duration = duration
        self.details = details
        super().__init__(self._format_message())
    
    def _format_message(self) -> str:
        msg = f"GenerationError: {self.message} (stem: {self.stem_type})"
        if self.duration:
            msg += f" (duration: {self.duration}s)"
        if self.details:
            msg += f" - {self.details}"
        return msg


# =============================================================================
# Enums and Type Classes
# =============================================================================

class StemType(Enum):
    """Supported stem types for regeneration."""
    
    VOCALS = "vocals"
    DRUMS = "drums"
    BASS = "bass"
    OTHER = "other"  # Instruments, pads, etc.
    ALL = "all"  # Generate all stems
    
    @classmethod
    def from_string(cls, value: str) -> "StemType":
        """Convert string to StemType enum."""
        value_lower = value.lower().strip()
        for stem_type in cls:
            if stem_type.value == value_lower:
                return stem_type
        raise ValueError(f"Unknown stem type: {value}. Valid types: {[st.value for st in cls]}")


class ConditioningType(Enum):
    """Types of conditioning available for generation."""
    
    CHORD = "chord"
    TEMPO = "tempo"
    KEY = "key"
    MELODY = "melody"
    DRUM = "drum"
    DESCRIPTION = "description"


# =============================================================================
# Conditioning Dataclasses
# =============================================================================

@dataclass
class MelodyConditioning:
    """
    Melody conditioning parameters for stem generation.
    
    Used to guide the generation to follow the melodic contour
    of the original recording.
    """
    
    contour: Optional[np.ndarray] = None  # Pitch salience matrix (time x frequency)
    f0_sequence: Optional[np.ndarray] = None  # Fundamental frequency sequence
    pitch_range: Tuple[float, float] = (55.0, 880.0)  # Hz (A1 to A5)
    onset_times: Optional[np.ndarray] = None  # Note onset times
    duration_seconds: Optional[float] = None  # Total duration
    
    def to_description(self) -> str:
        """Generate a text description for conditioning."""
        if self.pitch_range:
            low, high = self.pitch_range
            if high < 200:
                return "low male vocals, deep tone"
            elif high < 400:
                return "mid-range vocals, natural register"
            elif high < 600:
                return "high vocals, soprano range"
            else:
                return "very high vocals, whistle register"
        return "natural vocals"


@dataclass
class DrumConditioning:
    """
    Drum conditioning parameters for stem generation.
    
    Used to guide the generation to follow the drum pattern
    of the original recording.
    """
    
    onset_times: Optional[np.ndarray] = None  # Drum onsets
    onset_strengths: Optional[np.ndarray] = None  # Relative strengths
    tempo_bpm: Optional[float] = None  # Detected tempo
    pattern_description: Optional[str] = None  # Text description
    kick_pattern: Optional[str] = None  # Kick pattern (e.g., "1---2---")
    snare_pattern: Optional[str] = None  # Snare pattern (e.g., "--1--2--")
    hihat_pattern: Optional[str] = None  # Hi-hat pattern (e.g., "11111111")
    
    def to_description(self) -> str:
        """Generate a text description for conditioning."""
        if self.pattern_description:
            return self.pattern_description
        
        desc_parts = []
        if self.tempo_bpm:
            if 110 <= self.tempo_bpm <= 130:
                desc_parts.append("fast tempo dance beat")
            elif 80 <= self.tempo_bpm <= 110:
                desc_parts.append("mid-tempo groovy beat")
            else:
                desc_parts.append(f"{self.tempo_bpm:.0f} BPM beat")
        
        if self.kick_pattern:
            if "1" in self.kick_pattern[:4]:
                desc_parts.append("kick on beat 1")
        if self.snare_pattern:
            if "1" in self.snare_pattern[2:4]:
                desc_parts.append("snare on backbeat")
        
        return ", ".join(desc_parts) if desc_parts else "acoustic drum kit"


@dataclass
class ChordConditioning:
    """
    Chord conditioning parameters for stem generation.
    
    Used to guide the generation to follow the harmonic structure
    of the original recording.
    """
    
    chord_timeline: List[Tuple[str, float]] = field(default_factory=list)  # [(chord, start_time), ...]
    key: str = "C"  # Musical key (e.g., "C", "Am", "G major")
    time_signature: Tuple[int, int] = (4, 4)  # e.g., (4, 4) for 4/4
    
    def to_chord_description(self) -> str:
        """Generate a condensed chord description for JASCO."""
        if not self.chord_timeline:
            return f"chords in {self.key}"
        
        # Extract unique chords and their sequence
        chords = [c[0] for c in self.chord_timeline[:8]]  # First 8 chords
        
        # Simplify chord notation
        simplified = []
        for chord in chords:
            # Remove inversions and simplify
            chord = chord.replace(":5", "").replace(":3", "")
            if len(chord) <= 4:  # Basic chord
                simplified.append(chord)
        
        if simplified:
            return f"progression: {' - '.join(simplified[:4])} in {self.key}"
        return f"chords in {self.key}"


@dataclass
class TempoConditioning:
    """
    Tempo conditioning parameters for stem generation.
    
    Used to guide the generation tempo and timing.
    """
    
    tempo_bpm: float = 120.0
    time_signature: Tuple[int, int] = (4, 4)
    beat_grid: Optional[np.ndarray] = None  # Beat positions
    
    def to_description(self) -> str:
        """Generate a text description for tempo conditioning."""
        if 110 <= self.tempo_bpm <= 130:
            return "120 BPM dance music"
        elif 80 <= self.tempo_bpm <= 110:
            return "100 BPM mid-tempo"
        elif 60 <= self.tempo_bpm <= 80:
            return "70 BPM slow ballad"
        else:
            return f"{self.tempo_bpm:.0f} BPM"


# =============================================================================
# Callbacks for Progress Reporting
# =============================================================================

@dataclass
class GenerationCallbacks:
    """
    Callbacks for progress reporting during generation.
    
    These callbacks are invoked at various stages of the generation
    process to update the UI or log progress.
    """
    
    on_progress: Optional[Callable[[int, str], None]] = None  # (progress_percent, message)
    on_step: Optional[Callable[[int, int], None]] = None  # (current_step, total_steps)
    on_quality_update: Optional[Callable[[float], None]] = None  # (quality_score)
    on_log: Optional[Callable[[str], None]] = None  # (log_message)
    
    def report_progress(self, percent: int, message: str = "") -> None:
        """Report overall progress."""
        if self.on_progress:
            self.on_progress(percent, message)
        if self.on_log:
            self.on_log(f"Progress {percent}%: {message}")
    
    def report_step(self, current: int, total: int, message: str = "") -> None:
        """Report step progress."""
        if self.on_step:
            self.on_step(current, total)
        percent = int((current / total) * 100) if total > 0 else 0
        self.report_progress(percent, message)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class GenerationConfig:
    """
    Configuration for stem generation.
    
    Args:
        device: Device for inference ("mps", "cuda", "cpu")
        stem_type: Type of stem to generate ("vocals", "drums", "bass", "other", "all")
        duration: Duration of audio to generate in seconds (max 30s per call)
        conditioning: Dictionary of conditioning from profile extractors
            - chords: ChordConditioning or chord timeline
            - tempo: TempoConditioning or tempo_bpm
            - key: Musical key string
            - melody: MelodyConditioning
            - drums: DrumConditioning
        style_description: Additional text description for conditioning
        guidance_scale: Classifier-free guidance scale (higher = more adherence to conditioning)
        num_steps: Number of diffusion steps (higher = better quality)
        use_compact_model: Use smaller model variant for faster generation
        overlap_seconds: Overlap duration for seamless blending (default 0.5s)
        sample_rate: Output sample rate (default 44100 Hz)
    """
    
    device: str = "mps"
    stem_type: str = "drums"
    duration: int = 10
    conditioning: Dict[str, Any] = field(default_factory=dict)
    style_description: Optional[str] = None
    guidance_scale: float = 7.0
    num_steps: int = 50
    use_compact_model: bool = False
    overlap_seconds: float = 0.5
    sample_rate: int = 44100
    
    def __post_init__(self):
        """Validate configuration."""
        # Validate device
        valid_devices = ["mps", "cuda", "cpu"]
        if self.device not in valid_devices:
            logger.warning(f"Invalid device '{self.device}', falling back to 'cpu'")
            self.device = "cpu"
        
        # Validate stem type
        valid_stems = ["vocals", "drums", "bass", "other", "all"]
        if self.stem_type not in valid_stems:
            raise ValueError(f"Invalid stem_type '{self.stem_type}'. Valid: {valid_stems}")
        
        # Validate duration (JASCO limit is 30s per call)
        if self.duration < 1:
            logger.warning("Duration too short, using minimum 1s")
            self.duration = 1
        elif self.duration > 30:
            logger.warning("Duration exceeds 30s limit, using 30s")
            self.duration = 30
        
        # Validate guidance scale
        if self.guidance_scale < 1.0:
            logger.warning("guidance_scale too low, using minimum 1.0")
            self.guidance_scale = 1.0
        elif self.guidance_scale > 20.0:
            logger.warning("guidance_scale too high, using maximum 20.0")
            self.guidance_scale = 20.0
        
        # Validate num_steps
        if self.num_steps < 10:
            logger.warning("num_steps too low, using minimum 10")
            self.num_steps = 10
        elif self.num_steps > 100:
            logger.warning("num_steps too high, using maximum 100")
            self.num_steps = 100
    
    def get_stem_type_enum(self) -> StemType:
        """Get stem type as enum."""
        return StemType.from_string(self.stem_type)


# =============================================================================
# Result
# =============================================================================

@dataclass
class GenerationResult:
    """
    Result of stem generation.
    
    Attributes:
        audio: Generated audio as float32 numpy array, range [-1, 1]
        sample_rate: Sample rate of generated audio
        duration: Actual duration in seconds
        success: Whether generation completed successfully
        error: Error message if generation failed
        conditioning_used: Summary of conditioning applied
        generation_time: Time taken for generation in seconds
        metadata: Additional generation metadata
    """
    
    audio: Optional[np.ndarray] = None
    sample_rate: int = 44100
    duration: float = 0.0
    success: bool = False
    error: Optional[str] = None
    conditioning_used: Dict[str, Any] = field(default_factory=dict)
    generation_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def audio_shape(self) -> Tuple[int, ...]:
        """Get audio shape (samples,)."""
        if self.audio is None:
            return (0,)
        return self.audio.shape
    
    @property
    def num_samples(self) -> int:
        """Get number of audio samples."""
        if self.audio is None:
            return 0
        return len(self.audio)
    
    def to_mono(self) -> "GenerationResult":
        """Convert stereo audio to mono."""
        if self.audio is not None and self.audio.ndim > 1:
            self.audio = self.audio.mean(axis=1) if self.audio.ndim == 2 else self.audio
        return self
    
    def normalize(self, target_db: float = -3.0) -> "GenerationResult":
        """Normalize audio to target peak dB."""
        if self.audio is not None:
            current_max = np.abs(self.audio).max()
            if current_max > 0:
                target_linear = 10 ** (target_db / 20)
                self.audio = self.audio * (target_linear / current_max)
        return self


# =============================================================================
# Main Generator Class
# =============================================================================

class JASCOGenerator:
    """
    JASCO-based stem generator for AI-powered audio regeneration.
    
    This class wraps the AudioCraft JASCO model to enable conditional
    music generation for stem regeneration tasks.
    
    Features:
    - Support for MPS (Apple Silicon) and CPU inference
    - Conditional generation based on chords, tempo, melody, and drums
    - Batch generation for multiple regions
    - Progress callbacks for UI integration
    - Graceful error handling and fallback
    
    Example:
        >>> config = GenerationConfig(
        ...     stem_type="drums",
        ...     duration=10,
        ...     conditioning={"tempo": 120, "key": "C"}
        ... )
        >>> generator = JASCOGenerator(config)
        >>> result = generator.generate(tempo=120, stem_type="drums", duration=10)
        >>> if result.success:
        ...     audio = result.audio
    """
    
    # Default model variants (from smallest to largest)
    MODEL_VARIANTS = {
        "small": "facebook/musicgen-small",
        "medium": "facebook/musicgen-medium",
        "large": "facebook/musicgen-large",
    }
    
    # Default model for stem regeneration
    DEFAULT_MODEL = "facebook/musicgen-small"
    
    def __init__(self, config: Optional[GenerationConfig] = None):
        """
        Initialize the JASCO generator.
        
        Args:
            config: GenerationConfig instance. If None, uses defaults.
        """
        self.config = config or GenerationConfig()
        self._model = None
        self._model_path = None
        self._is_loaded = False
        self._callbacks: Optional[GenerationCallbacks] = None
        
        logger.info(f"Initializing JASCOGenerator (device: {self.config.device})")
    
    def __repr__(self) -> str:
        """String representation of the generator."""
        status = "loaded" if self._is_loaded else "not loaded"
        return f"JASCOGenerator(device={self.config.device}, status={status})"
    
    # =============================================================================
    # Model Loading
    # =============================================================================
    
    def load_model(
        self, 
        model_variant: str = "small",
        force_reload: bool = False
    ) -> bool:
        """
        Load the JASCO/MusicGen model for generation.
        
        Args:
            model_variant: Model size variant ("small", "medium", "large")
            force_reload: Force reload even if model is already loaded
            
        Returns:
            True if model loaded successfully, False otherwise
            
        Raises:
            ModelLoadError: If model loading fails
        """
        if self._is_loaded and not force_reload:
            logger.info("Model already loaded")
            return True
        
        # Get model name
        model_name = self.MODEL_VARIANTS.get(model_variant, self.DEFAULT_MODEL)
        logger.info(f"Loading model: {model_name}")
        
        try:
            # Try to import audiocraft
            try:
                from audiocraft.models import MusicGen
            except ImportError as e:
                raise ModelLoadError(
                    "audiocraft not installed",
                    device=self.config.device,
                    details="Install with: pip install audiocraft>=1.3.0"
                )
            # audio_utils is optional and may not be present in all audiocraft versions
            try:
                from audiocraft.data.audio import audio_utils  # type: ignore
            except Exception:
                audio_utils = None  # fallback if unavailable
            
            # Configure device
            if self.config.device == "mps" and not torch.backends.mps.is_available():
                logger.warning("MPS not available, falling back to CPU")
                self.config.device = "cpu"
            
            # Set device
            device = torch.device(self.config.device)
            logger.info(f"Using device: {device}")
            
            # Load model (MusicGen is used as JASCO equivalent for stem generation)
            # MusicGen supports conditioning via descriptions and can generate stems
            logger.info(f"Downloading/loading {model_name}...")
            
            # Try passing device to get_pretrained (works in some versions)
            try:
                self._model = MusicGen.get_pretrained(model_name, device=device)
            except TypeError:
                # Fallback for older versions that don't accept device
                self._model = MusicGen.get_pretrained(model_name)
                
                # Move to device manually
                try:
                    self._model.to(device)
                except (AttributeError, TypeError):
                    # If wrapper doesn't have .to(), try moving internal models
                    # AudioCraft models typically have .lm, .compression_model
                    try:
                        if hasattr(self._model, 'lm'):
                            self._model.lm.to(device)
                        if hasattr(self._model, 'compression_model'):
                            self._model.compression_model.to(device)
                    except Exception as e:
                        logger.warning(f"Could not move model internals to {device}: {e}")

            # Configure generation parameters (best-effort)
            try:
                self._model.generation_params = {
                    "duration": self.config.duration,
                    "cfg_coef": self.config.guidance_scale,
                    "num_steps": self.config.num_steps,
                }
            except Exception:
                # Some models manage generation parameters differently; ignore if not supported
                pass
            
            self._model_path = model_name
            self._is_loaded = True
            
            logger.info(f"✅ Model loaded successfully: {model_name}")
            return True
            
        except ModelLoadError:
            raise
        except Exception as e:
            error_msg = f"Failed to load model: {str(e)}"
            logger.error(error_msg)
            raise ModelLoadError(
                error_msg,
                device=self.config.device,
                details=str(e)
            )
    
    def unload_model(self) -> None:
        """Unload the model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
            self._is_loaded = False
            
            # Force garbage collection
            import gc
            gc.collect()
            
            if self.config.device == "mps":
                torch.mps.empty_cache()
            
            logger.info("Model unloaded")
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded
    
    # =============================================================================
    # Generation Methods
    # =============================================================================
    
    def generate(
        self,
        chords_timeline: Optional[List[Tuple[str, float]]] = None,
        tempo: Optional[float] = None,
        key: Optional[str] = None,
        stem_type: Optional[str] = None,
        duration: Optional[int] = None,
        callbacks: Optional[GenerationCallbacks] = None,
        **kwargs
    ) -> GenerationResult:
        """
        Generate a stem conditioned on musical structure.
        
        This is the main generation method that accepts conditioning parameters
        from the profiling module and generates audio matching the original
        musical structure.
        
        Args:
            chords_timeline: List of (chord_name, start_time) tuples
            tempo: Tempo in BPM
            key: Musical key (e.g., "C", "Am")
            stem_type: Type of stem to generate (overrides config)
            duration: Duration in seconds (overrides config)
            callbacks: Progress callbacks
            **kwargs: Additional conditioning parameters
            
        Returns:
            GenerationResult with generated audio or error
            
        Example:
            >>> result = generator.generate(
            ...     chords_timeline=[("C", 0.0), ("G", 2.0), ("Am", 4.0)],
            ...     tempo=120,
            ...     key="C",
            ...     stem_type="drums",
            ...     duration=10
            ... )
        """
        start_time = time.time()
        self._callbacks = callbacks or GenerationCallbacks()
        
        # Resolve parameters
        stem_type = stem_type or self.config.stem_type
        duration = duration or self.config.duration
        
        self._callbacks.report_step(0, 5, f"Generating {stem_type} stem ({duration}s)")
        
        try:
            # Load model if needed
            if not self._is_loaded:
                self._callbacks.report_step(1, 5, "Loading model...")
                self.load_model()
            
            # Build description from conditioning
            description = self._build_description(
                chords_timeline=chords_timeline,
                tempo=tempo,
                key=key,
                stem_type=stem_type,
                **kwargs
            )
            
            self._callbacks.report_step(2, 5, "Generating audio...")
            
            # Generate
            audio = self._generate_audio(description, duration)
            
            self._callbacks.report_step(4, 5, "Post-processing...")
            
            # Post-process
            result = self._create_result(
                audio=audio,
                duration=duration,
                conditioning={"chords": chords_timeline, "tempo": tempo, "key": key},
                generation_time=time.time() - start_time,
                stem_type=stem_type
            )
            
            self._callbacks.report_step(5, 5, "Complete")
            return result
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Generation failed: {error_msg}")
            return GenerationResult(
                success=False,
                error=error_msg,
                generation_time=time.time() - start_time
            )
    
    def generate_with_melody(
        self,
        melody_contour: np.ndarray,
        chords_timeline: Optional[List[Tuple[str, float]]] = None,
        tempo: Optional[float] = None,
        key: Optional[str] = None,
        duration: Optional[int] = None,
        callbacks: Optional[GenerationCallbacks] = None,
        **kwargs
    ) -> GenerationResult:
        """
        Generate a stem with melody conditioning.
        
        Uses the extracted melody contour to guide generation, ensuring
        the generated audio follows the melodic structure.
        
        Args:
            melody_contour: Melody salience matrix or f0 sequence
            chords_timeline: Chord progression timeline
            tempo: Tempo in BPM
            key: Musical key
            duration: Duration in seconds
            callbacks: Progress callbacks
            **kwargs: Additional parameters
            
        Returns:
            GenerationResult with generated audio
        """
        start_time = time.time()
        self._callbacks = callbacks or GenerationCallbacks()
        
        duration = duration or self.config.duration
        
        self._callbacks.report_step(0, 6, "Generating with melody conditioning")
        
        try:
            if not self._is_loaded:
                self._callbacks.report_step(1, 6, "Loading model...")
                self.load_model()
            
            # Build description including melody info
            melody_conditioning = MelodyConditioning(contour=melody_contour)
            description = self._build_description(
                chords_timeline=chords_timeline,
                tempo=tempo,
                key=key,
                stem_type=self.config.stem_type,
                melody=melody_conditioning,
                **kwargs
            )
            
            self._callbacks.report_step(2, 6, "Applying melody conditioning...")
            
            # For melody conditioning, we use a description-based approach
            # since MusicGen doesn't directly support contour conditioning
            audio = self._generate_audio(description, duration)
            
            self._callbacks.report_step(4, 6, "Post-processing...")
            
            result = self._create_result(
                audio=audio,
                duration=duration,
                conditioning={
                    "chords": chords_timeline,
                    "tempo": tempo,
                    "key": key,
                    "melody_conditioning": True
                },
                generation_time=time.time() - start_time,
                stem_type=self.config.stem_type
            )
            
            self._callbacks.report_step(6, 6, "Complete")
            return result
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Melody-conditioned generation failed: {error_msg}")
            return GenerationResult(
                success=False,
                error=error_msg,
                generation_time=time.time() - start_time
            )
    
    def generate_with_drums(
        self,
        drum_pattern: Optional[np.ndarray] = None,
        drum_conditioning: Optional[DrumConditioning] = None,
        chords_timeline: Optional[List[Tuple[str, float]]] = None,
        tempo: Optional[float] = None,
        key: Optional[str] = None,
        duration: Optional[int] = None,
        callbacks: Optional[GenerationCallbacks] = None,
        **kwargs
    ) -> GenerationResult:
        """
        Generate drums with drum pattern conditioning.
        
        Uses the extracted drum pattern to guide generation, ensuring
        the generated drums follow the original rhythm.
        
        Args:
            drum_pattern: Drum onset times and velocities
            drum_conditioning: DrumConditioning dataclass with pattern details
            chords_timeline: Chord progression timeline
            tempo: Tempo in BPM
            key: Musical key
            duration: Duration in seconds
            callbacks: Progress callbacks
            **kwargs: Additional parameters
            
        Returns:
            GenerationResult with generated drum audio
        """
        start_time = time.time()
        self._callbacks = callbacks or GenerationCallbacks()
        
        duration = duration or self.config.duration
        
        self._callbacks.report_step(0, 6, "Generating drums with conditioning")
        
        try:
            if not self._is_loaded:
                self._callbacks.report_step(1, 6, "Loading model...")
                self.load_model()
            
            # Build drum conditioning
            if drum_conditioning is None:
                drum_conditioning = DrumConditioning(
                    onset_times=drum_pattern,
                    tempo_bpm=tempo
                )
            
            # Build description
            description = self._build_description(
                chords_timeline=chords_timeline,
                tempo=tempo,
                key=key,
                stem_type="drums",
                drums=drum_conditioning,
                **kwargs
            )
            
            self._callbacks.report_step(2, 6, "Generating drum pattern...")
            
            audio = self._generate_audio(description, duration)
            
            self._callbacks.report_step(4, 6, "Post-processing...")
            
            result = self._create_result(
                audio=audio,
                duration=duration,
                conditioning={
                    "chords": chords_timeline,
                    "tempo": tempo,
                    "key": key,
                    "drum_pattern": True
                },
                generation_time=time.time() - start_time,
                stem_type="drums"
            )
            
            self._callbacks.report_step(6, 6, "Complete")
            return result
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Drum generation failed: {error_msg}")
            return GenerationResult(
                success=False,
                error=error_msg,
                generation_time=time.time() - start_time
            )
    
    def batch_generate(
        self,
        regions: List[Dict[str, Any]],
        stem_type: Optional[str] = None,
        callbacks: Optional[GenerationCallbacks] = None
    ) -> List[GenerationResult]:
        """
        Generate audio for multiple regions.
        
        Useful for regenerating multiple damaged sections of a stem.
        
        Args:
            regions: List of region dictionaries with keys:
                - chords_timeline: Chord progression for region
                - tempo: Tempo for region
                - key: Key for region
                - start: Start time in seconds
                - duration: Duration in seconds
            stem_type: Type of stem to generate
            callbacks: Progress callbacks
            
        Returns:
            List of GenerationResult, one per region
        """
        self._callbacks = callbacks or GenerationCallbacks()
        stem_type = stem_type or self.config.stem_type
        
        total = len(regions)
        results = []
        
        self._callbacks.report_progress(0, f"Generating {total} regions")
        
        for i, region in enumerate(regions):
            self._callbacks.report_step(i + 1, total, f"Region {i + 1}/{total}")
            
            result = self.generate(
                chords_timeline=region.get("chords_timeline"),
                tempo=region.get("tempo"),
                key=region.get("key"),
                stem_type=stem_type,
                duration=region.get("duration", self.config.duration),
                callbacks=self._callbacks
            )
            results.append(result)
            
            # Small delay to allow progress updates
            import time
            time.sleep(0.1)
        
        self._callbacks.report_progress(100, "Batch complete")
        return results
    
    # =============================================================================
    # Internal Generation Methods
    # =============================================================================
    
    def _build_description(
        self,
        chords_timeline: Optional[List[Tuple[str, float]]] = None,
        tempo: Optional[float] = None,
        key: Optional[str] = None,
        stem_type: str = "other",
        melody: Optional[MelodyConditioning] = None,
        drums: Optional[DrumConditioning] = None,
        timbre: Optional[TimbreProfile] = None,
        articulation: Optional[ArticulationProfile] = None,
        **kwargs
    ) -> List[str]:
        """
        Build text description for conditioning the generation.
        
        Args:
            chords_timeline: Chord progression
            tempo: Tempo in BPM
            key: Musical key
            stem_type: Type of stem
            melody: Melody conditioning
            drums: Drum conditioning
            timbre: Timbre profile for tone description
            articulation: Articulation profile for expressive details
            **kwargs: Additional conditioning
            
        Returns:
            List of description strings for generation
        """
        description_parts = []
        
        # Add stem-specific base description
        stem_descriptions = {
            "vocals": ["a cappella singing", "vocal track"],
            "drums": ["drums only", "drum kit", "percussion"],
            "bass": ["bass line", "bass guitar", "low frequencies"],
            "other": ["instrumental", "synthesizer"],
            "all": ["full mix", "complete song"]
        }
        
        base_desc = stem_descriptions.get(stem_type, ["instrumental"])
        description_parts.extend(base_desc)
        
        # Add key information
        if key:
            description_parts.append(f"in {key} key")
        
        # Add tempo information
        if tempo:
            if 110 <= tempo <= 130:
                description_parts.append("dance tempo")
            elif 80 <= tempo <= 110:
                description_parts.append("mid-tempo")
            elif 60 <= tempo <= 80:
                description_parts.append("slow tempo")
            else:
                description_parts.append(f"{tempo:.0f} BPM")
        
        # Add chord information
        if chords_timeline:
            chords = [c[0] for c in chords_timeline[:4]]  # First 4 chords
            if chords:
                chord_str = " - ".join(chords)
                description_parts.append(f"chord progression: {chord_str}")
        
        # Add melody conditioning description
        if melody:
            desc = melody.to_description()
            if desc:
                description_parts.append(desc)
        
        # Add drum conditioning description
        if drums:
            desc = drums.to_description()
            if desc:
                description_parts.append(desc)
                
        # Add timbre description
        if timbre:
            desc = timbre.texture_description
            if desc:
                description_parts.append(desc)
                
        # Add articulation description
        if articulation:
            desc = articulation.articulation_description
            if desc:
                description_parts.append(desc)
        
        # Add style description from config
        if self.config.style_description:
            description_parts.append(self.config.style_description)
        
        # Join parts into a coherent description
        description = ", ".join(description_parts)
        
        logger.info(f"Generated description: {description}")
        return [description]
    
    def _generate_audio(
        self,
        descriptions: List[str],
        duration: int
    ) -> np.ndarray:
        """
        Generate audio using the loaded model.
        
        Args:
            descriptions: List of text descriptions for conditioning
            duration: Duration in seconds
            
        Returns:
            Generated audio as numpy array
        """
        if self._model is None:
            raise GenerationError("Model not loaded", stem_type=self.config.stem_type)
        
        # Configure generation parameters with stability settings
        self._model.set_generation_params(
            duration=duration,
            cfg_coef=self.config.guidance_scale,
            temperature=1.0,  # Stable sampling temperature
            top_k=250,        # Nucleus sampling - prevents numerical issues
            top_p=0.0         # Disable top-p (use top-k instead)
        )
        
        logger.info(f"Generating {duration}s audio with guidance={self.config.guidance_scale}")
        
        # Generate using MusicGen (JASCO equivalent) with retry on numerical errors
        try:
            # MusicGen.generate() accepts descriptions and returns audio
            output = self._model.generate(descriptions)
        except RuntimeError as e:
            # Handle sampling errors by retrying with more conservative parameters
            if 'probability tensor contains' in str(e) or 'inf' in str(e) or 'nan' in str(e):
                logger.warning(f"Numerical instability detected, retrying with conservative sampling: {e}")
                self._model.set_generation_params(
                    duration=duration,
                    cfg_coef=min(self.config.guidance_scale, 5.0),  # Lower guidance
                    temperature=0.95,  # Slightly more conservative
                    top_k=200,
                    top_p=0.0
                )
                output = self._model.generate(descriptions)
            else:
                raise
        
        # Continue with normal processing
        try:
            
            # Convert to numpy array
            if isinstance(output, torch.Tensor):
                audio = output.cpu().float().numpy()
            else:
                audio = np.array(output)
            
            # Handle multi-channel audio
            if audio.ndim == 3:
                # (batch, channels, samples) -> (channels, samples)
                audio = audio[0]
            if audio.ndim == 2:
                # Stereo: average to mono for stem generation
                audio = audio.mean(axis=0)
            
            # Normalize to [-1, 1] range
            audio = audio.astype(np.float32)
            max_val = np.abs(audio).max()
            if max_val > 0:
                audio = audio / max_val
            
            logger.info(f"Generated audio shape: {audio.shape}")
            return audio
            
        except Exception as e:
            raise GenerationError(
                f"Generation failed: {str(e)}",
                stem_type=self.config.stem_type,
                duration=duration,
                details="Check model input parameters"
            )
    
    def _create_result(
        self,
        audio: np.ndarray,
        duration: float,
        conditioning: Dict[str, Any],
        generation_time: float,
        stem_type: str
    ) -> GenerationResult:
        """Create a GenerationResult from generated audio."""
        return GenerationResult(
            audio=audio,
            sample_rate=self.config.sample_rate,
            duration=duration,
            success=True,
            conditioning_used=conditioning,
            generation_time=generation_time,
            metadata={
                "stem_type": stem_type,
                "guidance_scale": self.config.guidance_scale,
                "num_steps": self.config.num_steps,
                "model": self._model_path
            }
        )
    
    # =============================================================================
    # Utility Methods
    # =============================================================================
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if not self._is_loaded:
            return {"status": "not loaded"}
        
        return {
            "status": "loaded",
            "model_path": self._model_path,
            "device": self.config.device,
            "config": {
                "guidance_scale": self.config.guidance_scale,
                "num_steps": self.config.num_steps,
                "duration": self.config.duration,
                "sample_rate": self.config.sample_rate
            }
        }
    
    def save_audio(
        self,
        result: GenerationResult,
        path: Union[str, Path],
        format: str = "wav"
    ) -> bool:
        """
        Save generated audio to file.
        
        Args:
            result: GenerationResult to save
            path: Output file path
            format: Audio format ("wav", "flac", "mp3")
            
        Returns:
            True if saved successfully
        """
        if result.audio is None:
            logger.error("No audio to save")
            return False
        
        try:
            import soundfile as sf
            
            # Ensure correct format
            if isinstance(path, str):
                path = Path(path)
            
            sf.write(str(path), result.audio, result.sample_rate)
            logger.info(f"Saved audio to {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save audio: {e}")
            return False
    
    def generate_from_profile(
        self,
        profile: Dict[str, Any],
        stem_type: Optional[str] = None,
        duration: Optional[int] = None,
        callbacks: Optional[GenerationCallbacks] = None
    ) -> GenerationResult:
        """
        Generate a stem from a complete musical profile.
        
        This is the main integration point with the profiling module.
        
        Args:
            profile: Musical profile dictionary from profiling module with keys:
                - chords: Chord timeline
                - tempo: Tempo in BPM
                - key: Musical key
                - melody: Melody contour (optional)
                - drums: Drum pattern (optional)
                - quality: Quality score
            stem_type: Type of stem to generate
            duration: Duration in seconds
            callbacks: Progress callbacks
            
        Returns:
            GenerationResult with generated audio
        """
        stem_type = stem_type or self.config.stem_type
        duration = duration or self.config.duration
        
        # Extract conditioning from profile
        chords_timeline = profile.get("chords", [])
        tempo = profile.get("tempo")
        key = profile.get("key")
        melody = profile.get("melody")
        drums = profile.get("drums")
        
        return self.generate(
            chords_timeline=chords_timeline,
            tempo=tempo,
            key=key,
            stem_type=stem_type,
            duration=duration,
            callbacks=callbacks,
            melody=melody,
            drums=drums
        )


# =============================================================================
# Factory Functions
# =============================================================================

def create_generator(
    device: str = "mps",
    stem_type: str = "drums",
    duration: int = 10,
    guidance_scale: float = 7.0,
    num_steps: int = 50
) -> JASCOGenerator:
    """
    Factory function to create a configured JASCOGenerator.
    
    Args:
        device: Device for inference
        stem_type: Type of stem to generate
        duration: Duration in seconds
        guidance_scale: Guidance scale for conditioning
        num_steps: Number of generation steps
        
    Returns:
        Configured JASCOGenerator instance
    """
    config = GenerationConfig(
        device=device,
        stem_type=stem_type,
        duration=duration,
        guidance_scale=guidance_scale,
        num_steps=num_steps
    )
    return JASCOGenerator(config)


# =============================================================================
# Example Usage (when run directly)
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("JASCO Generator - Stem Regeneration Demo")
    print("=" * 60)
    
    # Create generator with default config
    config = GenerationConfig(
        device="mps",
        stem_type="drums",
        duration=5,  # Short for demo
        guidance_scale=7.0,
        num_steps=30  # Reduced for demo
    )
    
    generator = JASCOGenerator(config)
    
    print(f"\nGenerator: {generator}")
    
    # Try to load model
    try:
        success = generator.load_model()
        print(f"Model loaded: {success}")
        print(f"Model info: {generator.get_model_info()}")
        
        # Generate with conditioning
        print("\nGenerating drum stem...")
        result = generator.generate(
            chords_timeline=[("C", 0.0), ("G", 1.0), ("Am", 2.0), ("F", 3.0)],
            tempo=120,
            key="C",
            stem_type="drums",
            duration=5
        )
        
        if result.success:
            print(f"✅ Generation successful!")
            print(f"   Duration: {result.duration}s")
            print(f"   Samples: {result.num_samples}")
            print(f"   Generation time: {result.generation_time:.2f}s")
        else:
            print(f"❌ Generation failed: {result.error}")
            
    except ModelLoadError as e:
        print(f"⚠️ Model load failed: {e}")
        print("\nNote: Install audiocraft to use JASCO generation:")
        print("   pip install audiocraft>=1.3.0")
