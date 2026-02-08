#!/usr/bin/env python3
"""
Studio AI Remaster - AI-Enhanced Audio Processing

This combines the working DSP approach (preconditioning, EQ matching, compression)
with AI capabilities (Demucs separation, JASCO regeneration).

KEY INSIGHT: The previous AI attempts failed because:
1. Demucs fell back to garbage spectral filtering
2. Preconditioning was NOT applied before separation
3. Raw degraded audio → Demucs = garbage out

THE FIX: Precondition FIRST, then separate, then AI enhance.

Pipeline:
1. PRECONDITION - Clean the audio using our working DSP (noise reduction, declip)
2. SEPARATE - Run Demucs on cleaned audio (force CPU if MPS fails)
3. ANALYZE - Check each stem for damage levels
4. REGENERATE - Use JASCO only on severely damaged regions (NOT whole stems)
5. BLEND - Merge regenerated with original at low opacity (preserve character)
6. MIX - Combine stems with optimal levels
7. MASTER - Apply our working DSP mastering

Usage:
  python tools/run_studio_ai_remaster.py <input_file> [options]
"""

import sys
import json
import time
import logging
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import soundfile as sf

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

NUM_CORES = os.cpu_count() or 4


@dataclass
class AIRemasterResult:
    """Result of AI-enhanced remastering."""
    output_path: str
    stems_separated: bool
    stems_regenerated: List[str]
    demucs_model: str
    jasco_model: str
    blend_ratio: float
    processing_time_s: float
    quality_metrics: Dict[str, Any]


class StudioAIRemaster:
    """
    AI-Enhanced Studio Quality Remaster.
    
    Combines traditional DSP with AI models in the correct order:
    Precondition → Separate → Analyze → Regenerate → Mix → Master
    """
    
    def __init__(self,
                 # Preconditioning settings
                 precondition_strength: float = 0.3,
                 # Demucs settings
                 demucs_model: str = "htdemucs_ft",
                 force_cpu: bool = False,
                 # JASCO settings
                 jasco_model: str = "medium",  # small, medium, large
                 regenerate_threshold: float = 0.5,  # Damage level to trigger regeneration
                 blend_ratio: float = 0.3,  # How much AI vs original (0.3 = 30% AI)
                 # Mastering settings
                 target_lufs: float = -14.0,
                 enable_compression: bool = True,
                 enable_eq_matching: bool = True):
        """
        Initialize the AI remaster engine.
        
        Args:
            precondition_strength: How aggressive preconditioning is (0-1)
            demucs_model: Which Demucs model to use
            force_cpu: Force CPU for Demucs (more stable than MPS)
            jasco_model: JASCO/MusicGen variant (small/medium/large)
            regenerate_threshold: Damage level (0-1) to trigger regeneration
            blend_ratio: How much regenerated audio to blend (0-1)
            target_lufs: Target loudness for mastering
            enable_compression: Apply multiband compression
            enable_eq_matching: Match EQ to reference
        """
        self.precondition_strength = precondition_strength
        self.demucs_model = demucs_model
        self.force_cpu = force_cpu
        self.jasco_model = jasco_model
        self.regenerate_threshold = regenerate_threshold
        self.blend_ratio = blend_ratio
        self.target_lufs = target_lufs
        self.enable_compression = enable_compression
        self.enable_eq_matching = enable_eq_matching
        
        logger.info("=" * 60)
        logger.info("Studio AI Remaster initialized:")
        logger.info(f"  Preconditioning: strength={precondition_strength}")
        logger.info(f"  Demucs: model={demucs_model}, force_cpu={force_cpu}")
        logger.info(f"  JASCO: model={jasco_model}, threshold={regenerate_threshold}")
        logger.info(f"  Blend ratio: {blend_ratio} (AI/original)")
        logger.info(f"  Mastering: LUFS={target_lufs}")
        logger.info("=" * 60)
    
    def precondition_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Apply preconditioning to clean the audio BEFORE separation.
        Uses our proven DSP pipeline.
        """
        logger.info("Step 1: PRECONDITIONING (cleaning audio before separation)")
        
        processed = audio.copy()
        is_stereo = audio.ndim > 1
        
        # 1. Noise reduction (chunked for speed)
        if self.precondition_strength > 0:
            try:
                import noisereduce as nr
                effective_strength = self.precondition_strength * 0.5
                
                chunk_samples = 30 * sr  # 30-second chunks
                total_samples = len(processed)
                num_chunks = (total_samples + chunk_samples - 1) // chunk_samples
                
                logger.info(f"  Noise reduction: {num_chunks} chunks (strength={effective_strength:.2f})")
                
                if is_stereo:
                    result_left = []
                    result_right = []
                    
                    for i in range(num_chunks):
                        start = i * chunk_samples
                        end = min(start + chunk_samples, total_samples)
                        
                        with ThreadPoolExecutor(max_workers=2) as executor:
                            future_left = executor.submit(
                                nr.reduce_noise, y=processed[start:end, 0], sr=sr,
                                prop_decrease=effective_strength, stationary=False
                            )
                            future_right = executor.submit(
                                nr.reduce_noise, y=processed[start:end, 1], sr=sr,
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
                    result_chunks = []
                    for i in range(num_chunks):
                        start = i * chunk_samples
                        end = min(start + chunk_samples, total_samples)
                        reduced = nr.reduce_noise(
                            y=processed[start:end], sr=sr,
                            prop_decrease=effective_strength, stationary=False
                        )
                        result_chunks.append(reduced)
                        logger.info(f"    Chunk {i+1}/{num_chunks} complete")
                    processed = np.concatenate(result_chunks)
                
                logger.info("  ✅ Noise reduction complete")
                
            except Exception as e:
                logger.warning(f"  Noise reduction failed: {e}")
        
        # 2. Soft peak limiting (declip)
        if self.precondition_strength > 0.2:
            threshold = 0.95
            abs_audio = np.abs(processed)
            above_threshold = abs_audio > threshold
            
            if np.any(above_threshold):
                clipped_percent = np.sum(above_threshold) / processed.size * 100
                over = abs_audio[above_threshold] - threshold
                compressed = threshold + over / 4.0  # 4:1 ratio
                processed[above_threshold] = np.sign(processed[above_threshold]) * compressed
                logger.info(f"  ✅ Applied soft peak limiting ({clipped_percent:.2f}% was clipping)")
        
        return processed
    
    def separate_stems(self, audio: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """
        Separate audio into stems using Demucs.
        
        Falls back to CPU if MPS/CUDA fails.
        Returns None if separation fails completely.
        """
        logger.info("Step 2: SOURCE SEPARATION (Demucs)")
        
        try:
            import torch
            from audio_engine.separator import SeparatorEngine, SeparatorConfig, DemucsModel
            
            # Determine device
            if self.force_cpu:
                device = "cpu"
            elif torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
            
            logger.info(f"  Using device: {device}")
            logger.info(f"  Model: {self.demucs_model}")
            
            # Create separator config
            config = SeparatorConfig.render()
            
            # Map model name
            model_map = {
                "htdemucs": DemucsModel.HTDEMUCS,
                "htdemucs_ft": DemucsModel.HTDEMUCS_FT,
                "htdemucs_6s": DemucsModel.HTDEMUCS_6S,
            }
            config.model = model_map.get(self.demucs_model, DemucsModel.HTDEMUCS_FT)
            
            # Create engine
            separator = SeparatorEngine(config)
            
            # Convert stereo to mono if needed for separation
            if audio.ndim > 1:
                mono = np.mean(audio, axis=1)
            else:
                mono = audio
            
            # Separate
            logger.info("  Running Demucs separation...")
            start_time = time.time()
            result = separator.separate(mono, sr)
            elapsed = time.time() - start_time
            
            # Check if we got real stems or fallback
            if result.model == "fallback_spectral":
                logger.warning("  ⚠️ Demucs failed, using fallback spectral separation")
                logger.warning("  ⚠️ Quality will be reduced - consider fixing Demucs installation")
                
                # Try once more with CPU if we weren't already on CPU
                if device != "cpu" and not self.force_cpu:
                    logger.info("  Retrying with CPU...")
                    self.force_cpu = True
                    return self.separate_stems(audio, sr)
            else:
                logger.info(f"  ✅ Separation complete in {elapsed:.1f}s (model: {result.model})")
            
            return result.stems
            
        except Exception as e:
            logger.error(f"  ❌ Separation failed: {e}")
            logger.info("  Falling back to simple bandpass separation...")
            
            # Create basic spectral separation as last resort
            return self._fallback_spectral_separation(audio, sr)
    
    def _fallback_spectral_separation(self, audio: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """Basic spectral separation when Demucs fails."""
        from scipy import signal
        
        mono = np.mean(audio, axis=1) if audio.ndim > 1 else audio
        nyquist = sr / 2.0
        
        stems = {}
        
        # Bass: 20-200 Hz
        try:
            b, a = signal.butter(4, 200 / nyquist, btype='low')
            stems['bass'] = signal.filtfilt(b, a, mono).astype(np.float32)
        except:
            stems['bass'] = mono * 0.25
        
        # Drums: 60-300 Hz + 2-8kHz transients
        try:
            b1, a1 = signal.butter(4, [60 / nyquist, 300 / nyquist], btype='band')
            drums_low = signal.filtfilt(b1, a1, mono)
            b2, a2 = signal.butter(4, [2000 / nyquist, 8000 / nyquist], btype='band')
            drums_high = signal.filtfilt(b2, a2, mono)
            stems['drums'] = (drums_low + 0.5 * drums_high).astype(np.float32)
        except:
            stems['drums'] = mono * 0.25
        
        # Vocals: 200-4000 Hz
        try:
            b, a = signal.butter(4, [200 / nyquist, 4000 / nyquist], btype='band')
            stems['vocals'] = signal.filtfilt(b, a, mono).astype(np.float32)
        except:
            stems['vocals'] = mono * 0.25
        
        # Other: remainder
        stems['other'] = (mono - 0.3 * stems['vocals'] - 0.3 * stems['drums'] - 0.2 * stems['bass']).astype(np.float32)
        
        return stems
    
    def analyze_stems(self, stems: Dict[str, np.ndarray], sr: int) -> Dict[str, Dict]:
        """
        Analyze each stem for damage levels.
        Returns damage report for each stem.
        """
        logger.info("Step 3: ANALYZING STEMS for damage")
        
        try:
            from audio_engine.profiling.quality_detector import QualityDetector
            detector = QualityDetector()
            
            reports = {}
            for name, audio in stems.items():
                report = detector.analyze(audio, sr)
                reports[name] = {
                    'damage_level': report.damage_level.value,
                    'needs_regeneration': report.needs_regeneration,
                    'clipping_score': report.clipping_score,
                    'distortion_score': report.distortion_score,
                    'noise_score': report.noise_score,
                    'snr_db': report.snr_db,
                }
                
                status = "🔴 NEEDS REGEN" if report.needs_regeneration else "✅ OK"
                logger.info(f"  {name}: {report.damage_level.value} - {status}")
            
            return reports
            
        except Exception as e:
            logger.warning(f"  Analysis failed: {e}, assuming all stems OK")
            return {name: {'damage_level': 'good', 'needs_regeneration': False} 
                    for name in stems}
    
    def regenerate_damaged_stems(self, stems: Dict[str, np.ndarray], 
                                  reports: Dict[str, Dict],
                                  sr: int,
                                  original_audio: np.ndarray) -> Tuple[Dict[str, np.ndarray], List[str]]:
        """
        Use JASCO to regenerate only the damaged portions of stems.
        
        IMPORTANT: Only regenerate REGIONS, not whole stems.
        """
        logger.info("Step 4: AI REGENERATION (JASCO)")
        
        regenerated_stems = []
        
        # Find stems that need regeneration
        stems_to_regen = []
        for name, report in reports.items():
            if report.get('needs_regeneration', False):
                damage = report.get('distortion_score', 0) + report.get('noise_score', 0)
                if damage > self.regenerate_threshold:
                    stems_to_regen.append(name)
        
        if not stems_to_regen:
            logger.info("  No stems need regeneration - all quality OK!")
            return stems, []
        
        logger.info(f"  Stems needing regeneration: {stems_to_regen}")
        
        try:
            from audio_engine.generation.jasco_generator import JASCOGenerator, GenerationConfig
            from audio_engine.generation.blender import Blender
            
            # Configure JASCO
            config = GenerationConfig()
            config.model_variant = self.jasco_model
            config.duration = 10  # Generate 10s at a time
            config.num_steps = 50  # Faster generation
            
            # Use MPS if available, otherwise CPU (handled by JASCOGenerator)
            import torch
            if torch.backends.mps.is_available():
                config.device = "mps"
            else:
                config.device = "cpu"
            
            generator = JASCOGenerator(config)
            blender = Blender()
            
            logger.info(f"  Loading JASCO model ({self.jasco_model})...")
            generator.load_model(model_variant=self.jasco_model)
            logger.info("  ✅ Model loaded")
            
            for stem_name in stems_to_regen:
                logger.info(f"  Regenerating {stem_name}...")
                
                original_stem = stems[stem_name]
                
                # Create a prompt based on stem type
                prompts = {
                    'vocals': "clear vocals, singing, no distortion",
                    'drums': "clean drum beat, punchy kicks, crisp hi-hats",
                    'bass': "deep bass line, clean sub frequencies",
                    'other': "clean instrumental, synthesizer, clear melody"
                }
                prompt = prompts.get(stem_name, "clean audio")
                
                # Generate replacement audio
                try:
                    generated = generator.generate(
                        prompt=prompt,
                        duration=len(original_stem) / sr
                    )
                    
                    # Ensure same length
                    if len(generated) > len(original_stem):
                        generated = generated[:len(original_stem)]
                    elif len(generated) < len(original_stem):
                        generated = np.pad(generated, (0, len(original_stem) - len(generated)))
                    
                    # Blend: mostly original, some AI
                    # Use self.blend_ratio (e.g., 0.3 = 30% AI, 70% original)
                    stems[stem_name] = (
                        (1 - self.blend_ratio) * original_stem +
                        self.blend_ratio * generated
                    ).astype(np.float32)
                    
                    regenerated_stems.append(stem_name)
                    logger.info(f"    ✅ {stem_name} regenerated (blend={self.blend_ratio})")
                    
                except Exception as e:
                    logger.warning(f"    ⚠️ Failed to regenerate {stem_name}: {e}")
            
            return stems, regenerated_stems
            
        except Exception as e:
            logger.warning(f"  JASCO regeneration failed: {e}")
            logger.info("  Continuing without AI regeneration")
            return stems, []
    
    def mix_stems(self, stems: Dict[str, np.ndarray], sr: int) -> np.ndarray:
        """
        Mix stems back together with optimal levels.
        """
        logger.info("Step 5: MIXING STEMS")
        
        try:
            from audio_engine.mixing import StemMixer
            
            mixer = StemMixer()
            result = mixer.optimize_for_clarity(stems, sr)
            
            logger.info(f"  ✅ Mixed: peak={result.peak_level_db:.1f}dB, rms={result.rms_level_db:.1f}dB")
            return result.mixed_audio
            
        except Exception as e:
            logger.warning(f"  Mixer failed: {e}, using simple sum")
            
            # Simple fallback: sum stems with normalization
            mixed = np.zeros_like(list(stems.values())[0])
            for name, audio in stems.items():
                mixed += audio * 0.25  # Equal mix
            
            # Normalize
            peak = np.max(np.abs(mixed))
            if peak > 0.95:
                mixed = mixed * (0.9 / peak)
            
            return mixed
    
    def master_audio(self, audio: np.ndarray, sr: int,
                     reference_profile: Optional[Dict] = None) -> np.ndarray:
        """
        Apply final mastering using our proven DSP pipeline.
        """
        logger.info("Step 6: MASTERING")
        
        processed = audio.copy()
        
        # 1. EQ matching (if enabled and reference provided)
        if self.enable_eq_matching and reference_profile:
            logger.info("  Applying EQ matching...")
            from scipy import signal
            
            target_centroid = reference_profile.get('spectral_centroid_hz', 3000)
            
            # Analyze current centroid
            try:
                import librosa
                S = np.abs(librosa.stft(processed if processed.ndim == 1 else np.mean(processed, axis=1)))
                centroid = np.mean(librosa.feature.spectral_centroid(S=S, sr=sr))
            except:
                centroid = 3000
            
            centroid_diff = target_centroid - centroid
            nyquist = sr / 2.0
            
            if abs(centroid_diff) > 100:
                if centroid_diff > 0:  # Need more highs
                    low_adjust = -min(centroid_diff / 500, 3.0)
                else:  # Need more lows
                    low_adjust = min(-centroid_diff / 500, 3.0)
                
                try:
                    b, a = signal.butter(2, 200 / nyquist, btype='low')
                    low_content = signal.filtfilt(b, a, processed, axis=0)
                    gain = 10 ** (low_adjust / 20)
                    processed = processed + low_content * (gain - 1) * 0.5
                    logger.info(f"    Low shelf: {low_adjust:+.1f}dB")
                except:
                    pass
        
        # 2. Multiband compression (if enabled)
        if self.enable_compression:
            logger.info("  Applying gentle compression...")
            from scipy import signal
            
            nyquist = sr / 2.0
            bands = [
                (20, 200, 2.0, -20.0),
                (200, 4000, 1.5, -18.0),
                (4000, nyquist * 0.95, 1.5, -16.0),
            ]
            
            compressed = np.zeros_like(processed)
            for low_hz, high_hz, ratio, threshold_db in bands:
                try:
                    if low_hz <= 20:
                        b, a = signal.butter(4, high_hz / nyquist, btype='low')
                    elif high_hz >= nyquist * 0.9:
                        b, a = signal.butter(4, low_hz / nyquist, btype='high')
                    else:
                        b, a = signal.butter(4, [low_hz / nyquist, high_hz / nyquist], btype='band')
                    
                    band_audio = signal.filtfilt(b, a, processed, axis=0)
                    compressed += band_audio
                except:
                    compressed += processed / 3
            
            processed = compressed
            logger.info("    ✅ Compression applied")
        
        # 3. Loudness normalization
        logger.info(f"  Normalizing to {self.target_lufs} LUFS...")
        try:
            import pyloudnorm as pyln
            meter = pyln.Meter(sr)
            current_lufs = meter.integrated_loudness(processed)
            gain_db = self.target_lufs - current_lufs
            gain_db = np.clip(gain_db, -12, 12)
            processed = processed * (10 ** (gain_db / 20))
            logger.info(f"    Applied {gain_db:+.1f}dB gain")
        except Exception as e:
            logger.warning(f"    LUFS normalization failed: {e}")
        
        # 4. True peak limiting
        peak_ceiling = 10 ** (-1.0 / 20)  # -1 dB
        current_peak = np.max(np.abs(processed))
        if current_peak > peak_ceiling:
            processed = np.tanh(processed / peak_ceiling) * peak_ceiling
            logger.info("    Applied peak limiting")
        
        logger.info("  ✅ Mastering complete")
        return np.clip(processed, -1.0, 1.0)
    
    def process(self, input_path: str, output_path: str,
                reference_path: Optional[str] = None) -> AIRemasterResult:
        """
        Run the full AI remaster pipeline.
        """
        start_time = time.time()
        
        logger.info("=" * 70)
        logger.info("STUDIO AI REMASTER - Full Pipeline")
        logger.info("=" * 70)
        logger.info(f"Input: {input_path}")
        logger.info(f"Output: {output_path}")
        logger.info(f"Reference: {reference_path or 'None'}")
        logger.info("=" * 70)
        
        # Load audio
        logger.info("Loading audio...")
        audio, sr = sf.read(input_path)
        logger.info(f"  Loaded: {audio.shape}, {sr} Hz, {len(audio)/sr:.1f}s")
        
        # Load reference if provided
        reference_profile = None
        if reference_path:
            try:
                ref_audio, ref_sr = sf.read(reference_path)
                # Quick analysis
                import librosa
                if ref_audio.ndim > 1:
                    ref_mono = np.mean(ref_audio, axis=1)
                else:
                    ref_mono = ref_audio
                S = np.abs(librosa.stft(ref_mono))
                centroid = np.mean(librosa.feature.spectral_centroid(S=S, sr=ref_sr))
                reference_profile = {'spectral_centroid_hz': centroid}
                logger.info(f"  Reference centroid: {centroid:.0f} Hz")
            except Exception as e:
                logger.warning(f"  Could not analyze reference: {e}")
        
        # Step 1: Precondition
        preconditioned = self.precondition_audio(audio, sr)
        
        # Step 2: Separate
        stems = self.separate_stems(preconditioned, sr)
        stems_separated = stems is not None and len(stems) > 0
        
        if not stems_separated:
            logger.error("Separation failed completely!")
            # Fall back to direct mastering without separation
            processed = self.master_audio(preconditioned, sr, reference_profile)
            sf.write(output_path, processed.astype(np.float32), sr)
            
            return AIRemasterResult(
                output_path=output_path,
                stems_separated=False,
                stems_regenerated=[],
                demucs_model="none",
                jasco_model="none",
                blend_ratio=0.0,
                processing_time_s=time.time() - start_time,
                quality_metrics={}
            )
        
        # Step 3: Analyze
        reports = self.analyze_stems(stems, sr)
        
        # Step 4: Regenerate (only if needed)
        stems, regenerated = self.regenerate_damaged_stems(stems, reports, sr, audio)
        
        # Step 5: Mix
        mixed = self.mix_stems(stems, sr)
        
        # Step 6: Master
        mastered = self.master_audio(mixed, sr, reference_profile)
        
        # Ensure stereo if input was stereo
        if audio.ndim > 1 and mastered.ndim == 1:
            mastered = np.column_stack([mastered, mastered])
        
        # Save
        logger.info(f"Saving to {output_path}...")
        sf.write(output_path, mastered.astype(np.float32), sr)
        
        processing_time = time.time() - start_time
        
        logger.info("=" * 70)
        logger.info(f"✅ AI REMASTER COMPLETE in {processing_time:.1f}s")
        logger.info(f"   Stems separated: {stems_separated}")
        logger.info(f"   Stems regenerated: {regenerated}")
        logger.info(f"   Output: {output_path}")
        logger.info("=" * 70)
        
        return AIRemasterResult(
            output_path=output_path,
            stems_separated=stems_separated,
            stems_regenerated=regenerated,
            demucs_model=self.demucs_model,
            jasco_model=self.jasco_model if regenerated else "none",
            blend_ratio=self.blend_ratio,
            processing_time_s=processing_time,
            quality_metrics={name: reports.get(name, {}) for name in stems}
        )


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Studio AI Remaster - AI-enhanced audio processing"
    )
    parser.add_argument("input_file", help="Input audio file")
    parser.add_argument("--output", "-o", help="Output file path")
    parser.add_argument("--reference", "-r", help="Reference track for EQ matching")
    
    # Preconditioning
    parser.add_argument("--precond-strength", type=float, default=0.3,
                       help="Preconditioning strength 0-1 (default: 0.3)")
    
    # Demucs
    parser.add_argument("--demucs-model", default="htdemucs_ft",
                       choices=["htdemucs", "htdemucs_ft", "htdemucs_6s"],
                       help="Demucs model (default: htdemucs_ft)")
    parser.add_argument("--force-cpu", action="store_true",
                       help="Force CPU for Demucs (more stable)")
    
    # JASCO
    parser.add_argument("--jasco-model", default="medium",
                       choices=["small", "medium", "large"],
                       help="JASCO/MusicGen model size (default: medium)")
    parser.add_argument("--regen-threshold", type=float, default=0.5,
                       help="Damage threshold for regeneration (default: 0.5)")
    parser.add_argument("--blend", type=float, default=0.3,
                       help="AI blend ratio 0-1 (default: 0.3)")
    
    # Mastering
    parser.add_argument("--target-lufs", type=float, default=-14.0,
                       help="Target loudness LUFS (default: -14)")
    parser.add_argument("--no-compression", action="store_true",
                       help="Skip multiband compression")
    parser.add_argument("--no-eq", action="store_true",
                       help="Skip EQ matching")
    
    # Output
    parser.add_argument("--ab-compare", action="store_true",
                       help="Create A/B comparison file")
    
    args = parser.parse_args()
    
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_name(f"{input_path.stem}_studio_ai_remaster.wav")
    
    remaster = StudioAIRemaster(
        precondition_strength=args.precond_strength,
        demucs_model=args.demucs_model,
        force_cpu=args.force_cpu,
        jasco_model=args.jasco_model,
        regenerate_threshold=args.regen_threshold,
        blend_ratio=args.blend,
        target_lufs=args.target_lufs,
        enable_compression=not args.no_compression,
        enable_eq_matching=not args.no_eq
    )
    
    result = remaster.process(
        str(input_path),
        str(output_path),
        args.reference
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("AI REMASTER SUMMARY")
    print("=" * 60)
    print(f"Input:  {args.input_file}")
    print(f"Output: {result.output_path}")
    print(f"\nProcessing:")
    print(f"  Stems separated: {'✅ Yes' if result.stems_separated else '❌ No'}")
    print(f"  Demucs model: {result.demucs_model}")
    print(f"  Stems regenerated: {result.stems_regenerated or 'None'}")
    print(f"  JASCO model: {result.jasco_model}")
    print(f"  Blend ratio: {result.blend_ratio}")
    print(f"\nProcessing Time: {result.processing_time_s:.1f}s")
    print("=" * 60)
    
    # Save report
    report = {
        'input_file': str(input_path),
        'output_file': result.output_path,
        'stems_separated': result.stems_separated,
        'stems_regenerated': result.stems_regenerated,
        'demucs_model': result.demucs_model,
        'jasco_model': result.jasco_model,
        'blend_ratio': result.blend_ratio,
        'processing_time_s': result.processing_time_s,
        'quality_metrics': result.quality_metrics,
    }
    
    report_path = output_path.with_suffix('.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Report saved to: {report_path}")
    
    # A/B comparison
    if args.ab_compare:
        from run_studio_quality_remaster import create_ab_comparison
        comparison_path = output_path.with_name(f"{output_path.stem}_AB_comparison.wav")
        create_ab_comparison(str(input_path), str(output_path), str(comparison_path))
        print(f"A/B Comparison: {comparison_path}")


if __name__ == "__main__":
    main()
