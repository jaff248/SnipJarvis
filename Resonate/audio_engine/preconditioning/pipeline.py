from dataclasses import dataclass
from typing import Optional
import numpy as np
import time

@dataclass
class PreConditioningConfig:
    """Configuration for the pre-conditioning pipeline."""
    enable_noise_reduction: bool = True
    noise_reduction_strength: float = 0.5  # 0-1 scale
    enable_declipping: bool = True
    declip_threshold: float = 0.99
    enable_dynamics_restoration: bool = True
    dynamics_expansion_ratio: float = 1.5

@dataclass
class PreConditioningResult:
    """Results and metrics from the pre-conditioning pipeline."""
    audio: np.ndarray  # The processed audio
    sample_rate: int
    noise_reduced: bool  # Was noise reduction applied?
    clips_repaired: int  # Number of clipped regions repaired
    dynamics_restored: bool  # Was dynamics restoration applied?
    input_snr_estimate: float  # Estimated input SNR in dB
    output_snr_estimate: float  # Estimated output SNR in dB
    processing_time: float  # Time taken in seconds

class PreConditioningPipeline:
    """
    Orchestrates the audio pre-conditioning stages:
    1. Noise Reduction
    2. De-clipping
    3. Dynamics Restoration
    """
    
    def __init__(self, sample_rate: int, config: Optional[PreConditioningConfig] = None):
        """
        Initialize the pipeline.
        
        Args:
            sample_rate: Audio sample rate in Hz
            config: Optional configuration object. If None, default config is used.
        """
        self.sample_rate = sample_rate
        self.config = config if config is not None else PreConditioningConfig()
        
        # Lazy-initialized processing modules
        self._noise_reducer = None
        self._declipper = None
        self._dynamics_restorer = None

    def process(self, audio: np.ndarray) -> PreConditioningResult:
        """
        Process audio through the enabled pre-conditioning stages.
        
        Args:
            audio: Input audio array (float32)
            
        Returns:
            PreConditioningResult containing processed audio and metrics
        """
        start_time = time.time()
        
        # Estimate input SNR
        input_snr = self._estimate_snr(audio)
        
        # Initialize counters/flags
        clips_repaired = 0
        noise_reduced_flag = False
        dynamics_restored_flag = False
        
        # === Stage 1: Noise Reduction ===
        if self.config.enable_noise_reduction:
            if self._noise_reducer is None:
                from .noise_reducer import NoiseReducer, NoiseReductionConfig
                nr_config = NoiseReductionConfig(strength=self.config.noise_reduction_strength)
                self._noise_reducer = NoiseReducer(self.sample_rate, nr_config)
            
            audio = self._noise_reducer.process(audio)
            noise_reduced_flag = True

        # === Stage 2: De-Clipping ===
        if self.config.enable_declipping:
            if self._declipper is None:
                from .declip import Declipper, DeclipConfig
                declip_config = DeclipConfig(detection_threshold=self.config.declip_threshold)
                self._declipper = Declipper(self.sample_rate, declip_config)
            
            # Detect clips BEFORE repair to count them
            clip_regions = self._declipper.detect_clipping(audio)
            clips_repaired = len(clip_regions)
            
            # Now repair them
            audio = self._declipper.process(audio)

        # === Stage 3: Dynamics Restoration ===
        if self.config.enable_dynamics_restoration:
            if self._dynamics_restorer is None:
                from .dynamics import DynamicsRestorer, DynamicsConfig
                dyn_config = DynamicsConfig(expansion_ratio=self.config.dynamics_expansion_ratio)
                self._dynamics_restorer = DynamicsRestorer(self.sample_rate, dyn_config)
            
            audio = self._dynamics_restorer.process(audio)
            dynamics_restored_flag = True

        # Estimate output SNR
        output_snr = self._estimate_snr(audio)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        return PreConditioningResult(
            audio=audio,
            sample_rate=self.sample_rate,
            noise_reduced=noise_reduced_flag,
            clips_repaired=clips_repaired,
            dynamics_restored=dynamics_restored_flag,
            input_snr_estimate=input_snr,
            output_snr_estimate=output_snr,
            processing_time=processing_time
        )

    def _estimate_snr(self, audio: np.ndarray) -> float:
        """
        Estimate SNR using a simple algorithm:
        - Divide audio into short frames (e.g., 512 samples)
        - Calculate RMS for each frame
        - Assume the lowest 10% of frames are "mostly noise"
        - SNR = mean(signal frames) / mean(noise frames) in dB
        """
        frame_size = 512
        if len(audio) < frame_size:
            return 0.0
            
        frames = [audio[i:i+frame_size] for i in range(0, len(audio), frame_size)]
        
        # Handle the last frame if it's incomplete
        if len(frames[-1]) < frame_size:
            frames.pop()
            
        if not frames:
            return 0.0
            
        rms_values = [np.sqrt(np.mean(frame**2)) for frame in frames]
        
        # Need enough frames to calculate percentiles
        if not rms_values:
            return 0.0
            
        noise_threshold = np.percentile(rms_values, 10)  # Lowest 10%
        
        noise_frames = [rms for rms in rms_values if rms < noise_threshold]
        signal_frames = [rms for rms in rms_values if rms >= noise_threshold]
        
        if len(noise_frames) == 0 or len(signal_frames) == 0:
            return 0.0  # Can't estimate
        
        mean_signal = np.mean(signal_frames)
        mean_noise = np.mean(noise_frames)
        
        # Avoid division by zero
        if mean_noise == 0:
            mean_noise = 1e-10
            
        snr = 20 * np.log10((mean_signal + 1e-10) / mean_noise)
        return snr
