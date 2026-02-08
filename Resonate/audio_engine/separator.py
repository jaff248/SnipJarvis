"""
Source Separation Module - Demucs integration for stem extraction

Uses Facebook Research's Demucs (HTDemucs) to separate audio into
vocals, drums, bass, and other instruments from phone recordings.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import torch
from demucs import separate

# Configure PyTorch to use all available CPU cores
NUM_CORES = os.cpu_count() or 8
torch.set_num_threads(NUM_CORES)
torch.set_num_interop_threads(NUM_CORES)

from .device import get_device, get_device_manager, DeviceManager
from .utils import tensor_to_numpy, format_duration

logger = logging.getLogger(__name__)


class DemucsModel(Enum):
    """Available Demucs models."""
    HTDEMUCS = "htdemucs"        # Default, balanced speed/quality
    HTDEMUCS_FT = "htdemucs_ft"  # Fine-tuned, best quality, 4x slower
    HTDEMUCS_6S = "htdemucs_6s"  # 6-stem model (experimental, adds piano/guitar)


class SeparationMode(Enum):
    """Separation processing modes."""
    PREVIEW = "preview"  # Fast, lower quality for preview
    RENDER = "render"    # Slow, highest quality for final output


@dataclass
class SeparationResult:
    """Result of source separation."""
    stems: Dict[str, np.ndarray]  # Stem audio data
    sample_rate: int
    model: str
    duration: float
    processing_time: float
    device: str
    
    def __post_init__(self):
        """Validate stems dictionary."""
        expected_stems = {'vocals', 'drums', 'bass', 'other'}
        if not all(stem in self.stems for stem in expected_stems):
            missing = expected_stems - set(self.stems.keys())
            logger.warning(f"Missing stems: {missing}")
    
    @property
    def stem_durations(self) -> Dict[str, float]:
        """Get duration of each stem."""
        return {stem: len(audio) / self.sample_rate 
                for stem, audio in self.stems.items()}
    
    def get_stem(self, name: str) -> Optional[np.ndarray]:
        """Get specific stem by name."""
        return self.stems.get(name)
    
    def __repr__(self) -> str:
        stem_count = len(self.stems)
        return (f"SeparationResult({stem_count} stems, "
                f"duration={format_duration(self.duration)}, "
                f"model={self.model})")


class SeparatorConfig:
    """Configuration for Demucs separation."""
    
    def __init__(
        self,
        model: DemucsModel = DemucsModel.HTDEMUCS_FT,
        mode: SeparationMode = SeparationMode.RENDER,
        segment: int = 10,
        shifts: int = 5,
        device_manager: DeviceManager = None
    ):
        """
        Initialize separator configuration.
        
        Args:
            model: Demucs model to use
            mode: Processing mode (preview/render)
            segment: Segment length in seconds
            shifts: Number of shifts for overlap-add (render only)
            device_manager: Device manager instance
        """
        self.model = model
        self.mode = mode
        self.segment = segment
        
        # Configure shifts based on mode
        if mode == SeparationMode.PREVIEW:
            self.shifts = 0  # No overlap-add for speed
        else:
            self.shifts = shifts  # Use overlap-add for quality
        
        # Get device manager
        self.device_manager = device_manager or get_device_manager()
        
        logger.info(f"Separator config: model={model.value}, "
                   f"mode={mode.value}, segment={segment}s, shifts={self.shifts}")
    
    @classmethod
    def preview(cls) -> 'SeparatorConfig':
        """Create preview configuration (fast)."""
        return cls(
            model=DemucsModel.HTDEMUCS,
            mode=SeparationMode.PREVIEW,
            segment=5,
            shifts=0
        )
    
    @classmethod
    def render(cls) -> 'SeparatorConfig':
        """Create render configuration (high quality)."""
        return cls(
            model=DemucsModel.HTDEMUCS_FT,
            mode=SeparationMode.RENDER,
            segment=10,
            shifts=5
        )


class SeparatorEngine:
    """
    Demucs source separation engine.
    
    Handles:
    - Model loading and management
    - Audio separation with optimal settings
    - Memory management for long files
    - Progress tracking
    """
    
    # Default stems from Demucs
    DEFAULT_STEMS = ['vocals', 'drums', 'bass', 'other']
    
    def __init__(self, config: SeparatorConfig = None):
        """
        Initialize separator engine.
        
        Args:
            config: Separator configuration (uses defaults if None)
        """
        self.config = config or SeparatorConfig.render()
        self._model: Optional[Any] = None
        self._device = self.config.device_manager.device
        
        logger.info(f"Initializing SeparatorEngine on {self._device}")
    
    @property
    def model(self):
        """Lazy-load Demucs model."""
        if self._model is None:
            self._load_model()
        return self._model
    
    def _load_model(self):
        """Load pretrained Demucs model."""
        import time
        start_time = time.time()
        
        model_name = self.config.model.value
        
        logger.info(f"Loading pretrained Demucs model: {model_name}")
        logger.info(f"Using {NUM_CORES} CPU cores for parallel processing")
        
        try:
            # CRITICAL: Use get_model to load PRETRAINED weights
            # NOT HTDemucs() which creates an untrained model!
            from demucs.pretrained import get_model
            
            # Load the pretrained model
            self._model = get_model(model_name)
            
            # Move to device
            device_str = str(self._device)
            if device_str == 'mps':
                try:
                    # Try MPS first for GPU acceleration
                    self._model = self._model.to('mps')
                    logger.info("✅ Using MPS (Apple Silicon GPU) for Demucs")
                except Exception as mps_err:
                    # Fall back to CPU if MPS fails
                    logger.warning(f"MPS failed: {mps_err}, falling back to CPU")
                    self._model = self._model.cpu()
            elif device_str == 'cuda':
                self._model = self._model.cuda()
            else:
                self._model = self._model.cpu()
            
            # Set to eval mode
            self._model.eval()
            
            load_time = time.time() - start_time
            logger.info(f"✅ Pretrained model loaded in {load_time:.1f}s")
            logger.info(f"   Model: {model_name}")
            logger.info(f"   Sources: {self._model.sources}")
            logger.info(f"   Device: {next(self._model.parameters()).device}")
            
        except Exception as e:
            logger.error(f"Failed to load pretrained model: {e}")
            raise RuntimeError(f"Demucs model loading failed: {e}") from e
    
    def separate(self, audio: np.ndarray, sample_rate: int) -> SeparationResult:
        """
        Separate audio into stems.
        
        Args:
            audio: Input audio as numpy array (float32, mono)
            sample_rate: Sample rate
            
        Returns:
            SeparationResult with separated stems
        """
        import time
        
        logger.info(f"Starting separation: {len(audio)/sample_rate:.1f}s audio")
        start_time = time.time()
        
        # Validate input
        if audio.ndim > 1:
            logger.warning("Input audio is multi-channel, mixing to mono")
            audio = np.mean(audio, axis=1)
        
        if sample_rate != 44100:
            logger.warning(f"Non-standard sample rate: {sample_rate} Hz")
        
        # Try Demucs separation first
        try:
            stems = self._demucs_separate(audio, sample_rate)
            duration = len(audio) / sample_rate
            processing_time = time.time() - start_time
            
            logger.info(f"✅ Separation complete in {processing_time:.1f}s")
            
            return SeparationResult(
                stems=stems,
                sample_rate=44100,
                model=self.config.model.value,
                duration=duration,
                processing_time=processing_time,
                device=str(self._device)
            )
            
        except Exception as e:
            logger.warning(f"Demucs separation failed: {e}, using fallback spectral separation")
            return self._fallback_separation(audio, sample_rate, start_time)
    
    def _demucs_separate(self, audio: np.ndarray, sample_rate: int) -> Dict[str, np.ndarray]:
        """Run Demucs separation using apply_model."""
        from demucs.apply import apply_model
        
        # Demucs expects stereo input at 44100 Hz
        # Shape should be (batch, channels, samples)
        if audio.ndim == 1:
            # Mono to stereo
            audio_stereo = np.stack([audio, audio], axis=0)
        else:
            audio_stereo = audio.T  # (samples, channels) -> (channels, samples)
        
        # Add batch dimension: (1, 2, samples)
        tensor = torch.from_numpy(audio_stereo.astype(np.float32)).unsqueeze(0)
        
        # Get model device
        model_device = next(self.model.parameters()).device
        tensor = tensor.to(model_device)
        
        logger.info(f"Running Demucs separation on {model_device}...")
        logger.info(f"  Input shape: {tensor.shape}")
        logger.info(f"  Model sources: {self.model.sources}")
        
        # Use apply_model for proper chunked processing
        # This handles long audio files properly
        with torch.no_grad():
            sources = apply_model(
                self.model,
                tensor,
                device=model_device,
                shifts=self.config.shifts,
                split=True,  # Enable chunked processing for long files
                overlap=0.25,
                progress=True,
            )
        
        # sources shape: (batch, num_sources, channels, samples)
        # Squeeze batch: (num_sources, channels, samples)
        sources = sources.squeeze(0)
        
        logger.info(f"  Output shape: {sources.shape}")
        
        # Convert to numpy and create stems dict
        stems = {}
        source_names = self.model.sources  # e.g., ['drums', 'bass', 'other', 'vocals']
        
        for i, name in enumerate(source_names):
            # Get this source: (channels, samples)
            source_audio = sources[i].cpu().numpy()
            
            # Convert stereo to mono by averaging channels
            if source_audio.ndim > 1:
                source_audio = np.mean(source_audio, axis=0)
            
            stems[name] = source_audio.astype(np.float32)
            logger.info(f"  {name}: {source_audio.shape}, peak={np.max(np.abs(source_audio)):.3f}")
        
        logger.info(f"✅ Successfully extracted {len(stems)} stems: {list(stems.keys())}")
        
        return stems
    
    def _create_spectral_fallback(self, audio: np.ndarray, sample_rate: int) -> Dict[str, np.ndarray]:
        """Create synthetic stems via multi-band filtering.

        Bands (per spec):
        - Vocals: 200-4000 Hz (bandpass)
        - Drums: 60-250 Hz (body) + 2000-6000 Hz (transient clicks)
        - Bass: 20-120 Hz (lowpass)
        - Other: 4000+ Hz (highpass residual)
        """
        import scipy.signal as signal

        nyquist = sample_rate / 2.0
        stems: Dict[str, np.ndarray] = {}

        def safe_filter(btype: str, Wn, order: int = 4, default_scale: float = 0.25) -> np.ndarray:
            try:
                b, a = signal.butter(order, Wn, btype=btype)
                return signal.filtfilt(b, a, audio).astype(np.float32)
            except Exception as err:
                logger.debug(f"Fallback filter {btype} {Wn} failed: {err}")
                return (audio * default_scale).astype(np.float32)

        # Vocals band
        vocals = safe_filter('band', [200 / nyquist, 4000 / nyquist], default_scale=0.3)
        stems['vocals'] = vocals

        # Drums body + transient band
        drums_body = safe_filter('band', [60 / nyquist, 250 / nyquist], default_scale=0.2)
        drums_click = safe_filter('band', [2000 / nyquist, 6000 / nyquist], default_scale=0.1)
        stems['drums'] = (drums_body + 0.5 * drums_click).astype(np.float32)

        # Bass lowpass
        bass = safe_filter('low', 120 / nyquist, default_scale=0.25)
        stems['bass'] = bass

        # Other high content
        other_high = safe_filter('high', 4000 / nyquist, default_scale=0.2)

        # Residual mitigation to reduce bleed
        residual = (audio - 0.4 * vocals - 0.4 * drums_body - 0.2 * bass)
        stems['other'] = ((0.6 * residual) + (0.4 * other_high)).astype(np.float32)

        # Normalize and guard against NaN
        for name, stem in stems.items():
            stem = np.nan_to_num(stem, nan=0.0, posinf=0.0, neginf=0.0)
            peak = np.max(np.abs(stem)) + 1e-6
            stems[name] = (stem / peak * 0.9).astype(np.float32)

        return stems

    def _fallback_separation(self, audio: np.ndarray, sample_rate: int,
                            start_time: float) -> SeparationResult:
        """Fallback spectral separation when Demucs fails.

        - Uses multi-band filtering to synthesize stems.
        - Gracefully degrades without aborting pipeline.
        - Logs timing and emits user-friendly warnings.
        """
        import time

        logger.warning("Using fallback spectral separation (reduced quality)")

        duration = len(audio) / sample_rate if sample_rate else 0

        try:
            stems = self._create_spectral_fallback(audio, sample_rate)
        except Exception as err:
            logger.error(f"Fallback spectral separation failed: {err}")
            stems = {
                'vocals': audio.astype(np.float32) * 0.25,
                'drums': audio.astype(np.float32) * 0.25,
                'bass': audio.astype(np.float32) * 0.25,
                'other': audio.astype(np.float32) * 0.25,
            }

        processing_time = time.time() - start_time
        logger.info(f"✅ Fallback separation complete in {processing_time:.1f}s (spectral bands)")

        return SeparationResult(
            stems=stems,
            sample_rate=sample_rate,
            model="fallback_spectral",
            duration=duration,
            processing_time=processing_time,
            device="cpu_fallback"
        )
    
    def separate_file(self, file_path: str) -> SeparationResult:
        """
        Separate audio file into stems.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            SeparationResult with separated stems
        """
        import time
        import soundfile as sf
        
        logger.info(f"Loading audio file: {file_path}")
        
        # Load audio
        audio, sr = sf.read(file_path)
        
        # Mix to mono if stereo
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        
        # Run separation
        return self.separate(audio, sr)
    
    def preload_model(self):
        """Preload model to avoid first-run delay."""
        _ = self.model
    
    def clear_cache(self):
        """Clear GPU memory cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif self._device.type == 'mps':
            torch.mps.empty_cache()
    
    def __repr__(self) -> str:
        return (f"SeparatorEngine(model={self.config.model.value}, "
                f"device={self._device}, mode={self.config.mode.value})")
    
    def __del__(self):
        """Cleanup GPU memory."""
        self.clear_cache()


# Convenience functions

def create_separator(mode: str = "render", model: str = "htdemucs_ft") -> SeparatorEngine:
    """
    Create separator engine with specified settings.
    
    Args:
        mode: "preview" or "render"
        model: "htdemucs", "htdemucs_ft", or "htdemucs_6s"
        
    Returns:
        Configured SeparatorEngine instance
    """
    # Parse model
    model_map = {
        "htdemucs": DemucsModel.HTDEMUCS,
        "htdemucs_ft": DemucsModel.HTDEMUCS_FT,
        "htdemucs_6s": DemucsModel.HTDEMUCS_6S
    }
    demucs_model = model_map.get(model, DemucsModel.HTDEMUCS_FT)
    
    # Parse mode
    mode_enum = SeparationMode.RENDER if mode == "render" else SeparationMode.PREVIEW
    
    # Create config
    if mode == "render":
        config = SeparatorConfig.render()
    else:
        config = SeparatorConfig.preview()
    
    # Override model
    config.model = demucs_model
    
    return SeparatorEngine(config)


# Example usage and testing
if __name__ == "__main__":
    import soundfile as sf
    
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Demucs Separator...")
    
    # Create separator for preview
    separator = create_separator(mode="preview")
    print(f"Created: {separator}")
    
    # Create test audio (simple tones for each stem)
    sample_rate = 44100
    duration = 3.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Mix of frequencies representing different instruments
    test_audio = (
        0.3 * np.sin(2 * np.pi * 440 * t) +      # Vocals (midrange)
        0.2 * np.sin(2 * np.pi * 80 * t) +        # Bass (low)
        0.2 * np.sin(2 * np.pi * 200 * t) +       # Other (low-mid)
        0.1 * np.sin(2 * np.pi * 800 * t)         # Drums (high freq click)
    )
    
    # Add some noise to simulate phone recording
    noise = np.random.randn(len(t)) * 0.05
    test_audio += noise
    
    # Normalize
    test_audio = test_audio / np.max(np.abs(test_audio)) * 0.8
    
    print(f"Test audio: {test_audio.shape}, {sample_rate} Hz")
    
    # Run separation
    try:
        result = separator.separate(test_audio, sample_rate)
        print(f"✅ Separation result: {result}")
        
        # Print stem info
        for stem_name, stem_audio in result.stems.items():
            print(f"  {stem_name}: {stem_audio.shape}, "
                 f"range=[{stem_audio.min():.3f}, {stem_audio.max():.3f}]")
        
        # Save stems for verification
        output_dir = Path("test_stems")
        output_dir.mkdir(exist_ok=True)
        
        for stem_name, stem_audio in result.stems.items():
            sf.write(output_dir / f"{stem_name}.wav", stem_audio, 44100)
        
        print(f"✅ Stems saved to {output_dir}/")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Cleanup
    separator.clear_cache()
