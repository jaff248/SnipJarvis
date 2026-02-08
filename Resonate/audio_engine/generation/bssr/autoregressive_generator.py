"""
Autoregressive Generator - Context-aware audio generation wrapper.

This module provides the AutoregressiveGenerator, which wraps the JASCO/MusicGen
model to support continuation-based generation, allowing for long-form audio
generation by using previous chunks as context.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..jasco_generator import (
    GenerationCallbacks,
    GenerationResult,
    JASCOGenerator,
)
from .bar_aligned_chunker import BarAlignedChunk

logger = logging.getLogger(__name__)


@dataclass
class ContinuationContext:
    """
    Context for autoregressive generation.
    
    Attributes:
        previous_audio: Audio from previous chunk to continue from
        sample_rate: Sample rate of previous audio
        duration: Duration of context to use (MusicGen typically uses ~3s max effectively)
        tempo: Tempo context
        key: Key context
        chords: Chord context
    """
    previous_audio: np.ndarray
    sample_rate: int
    duration: float = 2.0
    tempo: Optional[float] = None
    key: Optional[str] = None
    chords: Optional[List[Tuple[str, float]]] = None


class AutoregressiveGenerator:
    """
    Wraps JASCOGenerator to support autoregressive continuation.
    
    Uses the output of previous chunks to condition the generation of
    subsequent chunks, ensuring musical continuity across boundaries.
    """
    
    def __init__(self, base_generator: JASCOGenerator):
        """
        Initialize with a base JASCO generator.
        
        Args:
            base_generator: Configured JASCOGenerator instance
        """
        self.generator = base_generator
        
    def generate_chunk(self,
                       chunk: BarAlignedChunk,
                       context: Optional[ContinuationContext],
                       musical_profile: Dict[str, Any],
                       stem_type: str,
                       callbacks: Optional[GenerationCallbacks] = None) -> GenerationResult:
        """
        Generate a single chunk, optionally continuing from context.
        
        Args:
            chunk: BarAlignedChunk definition
            context: ContinuationContext from previous chunk (if any)
            musical_profile: Full musical profile for conditioning
            stem_type: Type of stem to generate
            callbacks: Progress callbacks
            
        Returns:
            GenerationResult containing the new audio
        """
        duration = chunk.duration
        
        if context is not None:
            return self._generate_with_continuation(
                chunk=chunk,
                context=context,
                profile=musical_profile,
                stem_type=stem_type,
                callbacks=callbacks
            )
        else:
            return self._generate_fresh(
                chunk=chunk,
                profile=musical_profile,
                stem_type=stem_type,
                callbacks=callbacks
            )
            
    def _generate_with_continuation(self,
                                    chunk: BarAlignedChunk,
                                    context: ContinuationContext,
                                    profile: Dict[str, Any],
                                    stem_type: str,
                                    callbacks: Optional[GenerationCallbacks] = None) -> GenerationResult:
        """Generate audio continuing from the provided context."""
        logger.info(f"Generating chunk {chunk.chunk_index} with continuation ({context.duration}s context)")
        
        # Ensure model is loaded
        if not self.generator.is_loaded:
            self.generator.load_model()
            
        # Extract prompt audio
        prompt_samples = int(context.duration * context.sample_rate)
        if len(context.previous_audio) > prompt_samples:
            prompt_audio = context.previous_audio[-prompt_samples:]
        else:
            prompt_audio = context.previous_audio
            
        # Build descriptions for this specific chunk
        # Note: We need to slice chords/melody/drums to match the chunk time
        # The chunk.start_time is absolute time in the song
        chunk_profile = self._slice_profile(profile, chunk.start_time, chunk.duration)
        
        # Build text description
        descriptions = self.generator._build_description(
            chords_timeline=chunk_profile.get('chords'),
            tempo=chunk_profile.get('tempo'),
            key=chunk_profile.get('key'),
            stem_type=stem_type,
            melody=chunk_profile.get('melody'),
            drums=chunk_profile.get('drums'),
            timbre=chunk_profile.get('timbre'),
            articulation=chunk_profile.get('articulation')
        )
        
        # Generate with continuation
        try:
            # We need to access the underlying MusicGen model for continuation
            # MusicGen.generate_continuation(prompt, prompt_sample_rate, descriptions, duration)
            if self.generator._model is None:
                raise RuntimeError("Model not loaded")
                
            # Convert prompt to tensor if needed (MusicGen handles numpy usually, but let's be safe)
            import torch
            
            # Ensure prompt is float32 and correct shape
            prompt_tensor = torch.from_numpy(prompt_audio).float()
            if prompt_tensor.ndim == 1:
                prompt_tensor = prompt_tensor.unsqueeze(0).unsqueeze(0) # (1, 1, samples)
            elif prompt_tensor.ndim == 2:
                prompt_tensor = prompt_tensor.unsqueeze(0) # (1, channels, samples)
                
            # Determine actual model device
            device = torch.device('cpu') # Default to CPU to be safe
            
            try:
                if hasattr(self.generator._model, 'lm') and hasattr(self.generator._model.lm, 'parameters'):
                    # This is the most reliable way for AudioCraft models
                    device = next(self.generator._model.lm.parameters()).device
                elif hasattr(self.generator._model, 'parameters'):
                    device = next(self.generator._model.parameters()).device
                elif hasattr(self.generator._model, 'device'):
                    device = torch.device(self.generator._model.device)
            except Exception:
                # If inspection fails, assume CPU
                device = torch.device('cpu')
            
            # Move prompt to confirmed model device
            prompt_tensor = prompt_tensor.to(device)
            
            # Configure params with stability settings
            self.generator._model.set_generation_params(
                duration=chunk.duration,
                cfg_coef=self.generator.config.guidance_scale,
                temperature=1.0,  # Stable sampling
                top_k=250,        # Nucleus sampling for stability
                top_p=0.0         # Disable top-p
            )
            
            # Generate with robust error handling
            try:
                output = self.generator._model.generate_continuation(
                    prompt=prompt_tensor,
                    prompt_sample_rate=context.sample_rate,
                    descriptions=descriptions,
                    progress=True
                )
            except RuntimeError as e:
                error_str = str(e)
                
                # Handle numerical instability (inf/nan in probability tensor)
                if 'probability tensor contains' in error_str or 'inf' in error_str or 'nan' in error_str:
                    logger.warning(f"Numerical instability in continuation, retrying with conservative params: {e}")
                    
                    # Retry with more conservative sampling
                    self.generator._model.set_generation_params(
                        duration=chunk.duration,
                        cfg_coef=min(self.generator.config.guidance_scale, 5.0),
                        temperature=0.95,
                        top_k=200,
                        top_p=0.0
                    )
                    
                    output = self.generator._model.generate_continuation(
                        prompt=prompt_tensor,
                        prompt_sample_rate=context.sample_rate,
                        descriptions=descriptions,
                        progress=True
                    )
                
                # Handle device mismatch (common with MPS)
                elif 'Input type' in error_str or 'Expected all tensors' in error_str:
                    logger.warning(f"Device mismatch detected, forcing CPU fallback: {e}")
                    
                    # Force prompt to CPU
                    prompt_tensor = prompt_tensor.cpu()
                    
                    # Force model to CPU (if possible)
                    try:
                        if hasattr(self.generator._model, 'cpu'):
                            self.generator._model.cpu()
                        elif hasattr(self.generator._model, 'to'):
                            self.generator._model.to('cpu')
                    except Exception:
                        pass # Can't move model, hope it's already on CPU
                        
                    # Retry
                    output = self.generator._model.generate_continuation(
                        prompt=prompt_tensor,
                        prompt_sample_rate=context.sample_rate,
                        descriptions=descriptions,
                        progress=True
                    )
                else:
                    raise
            
            # Process output similar to JASCOGenerator
            if isinstance(output, torch.Tensor):
                audio = output.cpu().float().numpy()
            else:
                audio = np.array(output)
                
            # Handle dimensions
            if audio.ndim == 3:
                audio = audio[0]
            if audio.ndim == 2:
                audio = audio.mean(axis=0)
                
            # Normalize
            max_val = np.abs(audio).max()
            if max_val > 0:
                audio = audio / max_val
                
            return GenerationResult(
                audio=audio,
                sample_rate=self.generator.config.sample_rate,
                duration=chunk.duration,
                success=True,
                metadata={"type": "continuation", "context_duration": context.duration}
            )
            
        except Exception as e:
            logger.error(f"Continuation generation failed: {e}")
            return GenerationResult(success=False, error=str(e))

    def _generate_fresh(self,
                        chunk: BarAlignedChunk,
                        profile: Dict[str, Any],
                        stem_type: str,
                        callbacks: Optional[GenerationCallbacks] = None) -> GenerationResult:
        """Generate a chunk from scratch (no context)."""
        logger.info(f"Generating chunk {chunk.chunk_index} fresh")
        
        # Slice profile for this chunk
        chunk_profile = self._slice_profile(profile, chunk.start_time, chunk.duration)
        
        # Delegate to standard generate
        return self.generator.generate_from_profile(
            profile=chunk_profile,
            stem_type=stem_type,
            duration=int(chunk.duration),
            callbacks=callbacks
        )
        
    def _slice_profile(self, profile: Dict[str, Any], start_time: float, duration: float) -> Dict[str, Any]:
        """
        Create a partial profile valid for the specific time window.
        
        This is critical because we want the chords/drums/etc. for THIS chunk,
        not the whole song (which might confuse the model or be too long).
        """
        end_time = start_time + duration
        
        sliced = profile.copy()
        
        # Slice chords
        if 'chords' in profile and isinstance(profile['chords'], list):
            # Filter chords that overlap with this window
            # Adjust times to be relative to start_time (0-based for this chunk)
            chunk_chords = []
            for chord, time in profile['chords']:
                if start_time <= time < end_time:
                    chunk_chords.append((chord, time - start_time))
                elif time < start_time and (not chunk_chords):
                    # Include the chord active at start_time if no chord starts immediately
                    # But actually MusicGen expects relative timing.
                    # Ideally we find the chord active at start_time and add it at 0.0
                    pass
            
            # Simple fallback: if no chords found but song has chords, find the one active at start
            if not chunk_chords and profile['chords']:
                # Find last chord before start_time
                last_chord = None
                for c, t in profile['chords']:
                    if t <= start_time:
                        last_chord = c
                    else:
                        break
                if last_chord:
                    chunk_chords.append((last_chord, 0.0))
                    
            sliced['chords'] = chunk_chords
            
        # TODO: Slice drum patterns and melody contours similarly if they are time-series
        # For now, we assume high-level descriptors or handle them in the future
        
        return sliced
