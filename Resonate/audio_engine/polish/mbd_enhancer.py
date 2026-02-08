"""MultiBand Diffusion neural polishing helper."""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def _patch_torch_load_for_audiocraft():
    """
    Temporarily patch torch.load to use weights_only=False for AudioCraft checkpoints.
    
    AudioCraft models from Meta are trusted, and require pickle loading due to
    omegaconf configs stored in checkpoints. PyTorch 2.6+ defaults to weights_only=True.
    """
    try:
        import torch
        
        _original_torch_load = torch.load
        
        def _patched_load(f, *args, **kwargs):
            # Default to weights_only=False for AudioCraft compatibility
            if 'weights_only' not in kwargs:
                kwargs['weights_only'] = False
            return _original_torch_load(f, *args, **kwargs)
        
        torch.load = _patched_load
        logger.debug("Patched torch.load for AudioCraft compatibility")
        return _original_torch_load
    except Exception as e:
        logger.debug(f"Could not patch torch.load: {e}")
        return None


def _restore_torch_load(original_load):
    """Restore original torch.load after AudioCraft model loading."""
    if original_load is not None:
        try:
            import torch
            torch.load = original_load
            logger.debug("Restored original torch.load")
        except Exception:
            pass


class MBDEnhancer:
    """
    MultiBandDiffusion neural polishing (optional).

    Uses AudioCraft's MBD for artifact reduction without changing content.

    What it does:
    - Audio → EnCodec compress → 4 parallel diffusion models → cleaner audio
    - Reduces artifacts (metallic, clicks) without fabricating content
    - ~30s processing time for 3-min track

    Philosophy: "Polish, don't fabricate"
    """

    def __init__(self, sample_rate: int = 24000):
        """Load MBD model (with graceful fallback)."""
        self.sample_rate = sample_rate
        self.available: bool = False
        self.mbd: Optional[object] = None

        # Patch torch.load for AudioCraft compatibility (PyTorch 2.6+)
        original_load = _patch_torch_load_for_audiocraft()

        try:
            from audiocraft.models import MultiBandDiffusion  # type: ignore

            self.mbd = MultiBandDiffusion.get_mbd_24khz(bw=6.0)
            self.available = True
            logger.info("✅ MBD model loaded successfully")
        except ImportError:
            logger.warning("AudioCraft not installed, MBD polish unavailable")
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(f"MBD initialization failed: {e}, skipping")
        finally:
            # Restore original torch.load
            _restore_torch_load(original_load)

    def process(self, audio: np.ndarray, intensity: float = 0.5) -> np.ndarray:
        """
        Apply MBD polishing.

        Args:
            audio: Input audio (float32, [-1, 1])
            intensity: 0-1, blend amount (0=no change, 1=full polish)

        Returns:
            Polished audio (float32) or original if MBD unavailable
        """
        # Always ensure float32 output
        if not self.available or self.mbd is None:
            return audio.astype(np.float32)

        try:
            import librosa

            # 1. Resample to 24kHz if needed
            if self.sample_rate != 24000:
                audio_24k = librosa.resample(audio, orig_sr=self.sample_rate, target_sr=24000)
            else:
                audio_24k = audio

            # 2. Run MBD regeneration
            # MBD.regenerate() returns a tuple (audio, sample_rate, ...), need to unpack
            try:
                regenerate_result = self.mbd.regenerate(audio_24k, sample_rate=24000)
                
                # Handle different return formats from AudioCraft MBD
                if isinstance(regenerate_result, tuple):
                    # Expected: (audio, sample_rate, ...) - unpack first value
                    if len(regenerate_result) >= 1:
                        polished = regenerate_result[0]
                        logger.debug(f"MBD returned tuple with {len(regenerate_result)} elements")
                    else:
                        logger.warning("MBD returned empty tuple, using original audio")
                        return audio.astype(np.float32)
                else:
                    # Returned directly as audio array
                    polished = regenerate_result
                    
            except ValueError as e:
                # Handle unpacking errors gracefully
                logger.error(f"MBD regeneration unpacking error: {e}")
                logger.warning("Falling back to original audio")
                return audio.astype(np.float32)

            # 3. Resample back if needed
            if self.sample_rate != 24000:
                polished = librosa.resample(polished, orig_sr=24000, target_sr=self.sample_rate)

            # 4. Blend with original using intensity
            blended = (1 - intensity) * audio + intensity * polished

            # 5. Safeguard against NaN/clipping
            blended = np.nan_to_num(blended, nan=0.0, posinf=0.0, neginf=0.0)
            blended = np.clip(blended, -1.0, 1.0)

            return blended.astype(np.float32)

        except Exception as e:  # pragma: no cover - runtime guard
            logger.warning(f"MBD processing failed: {e}, returning original audio")
            return audio.astype(np.float32)

    def is_available(self) -> bool:
        """Check if MBD is available."""
        return self.available

