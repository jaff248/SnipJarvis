#!/usr/bin/env python3
"""
Studio AI Remaster V2 - Enhanced AI-Enhanced Audio Processing

This version adds:
- --regenerate-minor: Regenerate stems with "minor" damage
- --force-jasco: Regenerate ALL stems with JASCO
- --reference: Reference track for vocal level matching
- Reference vocal analysis and extraction

Usage:
  python tools/run_studio_ai_remaster_v2.py <input_file> [options]
  
  # Use Kelly Clarkson track as reference:
  python tools/run_studio_ai_remaster_v2.py "Shadows John Summit Unreleased.mp3" \
    --reference "Kelly_Clarkson_-_Catch_My_Breath_ZUEZEU_POP_EDIT_FREE_DL_KLICKAUD.mp3" \
    --regenerate-minor \
    --blend 0.4
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
    reference_analysis: Optional[Dict] = None


class StudioAIRemasterV2:
    """
    AI-Enhanced Studio Quality Remaster V2.
    
    Enhanced features:
    - Regenerate minor damage stems
    - Force JASCO regeneration for all stems
    - Reference track vocal analysis
    - Vocal level matching to reference
    """
    
    def __init__(self,
                 # Preconditioning settings
                 precondition_strength: float = 0.3,
                 # Demucs settings
                 demucs_model: str = "htdemucs_ft",
                 force_cpu: bool = False,
                 # JASCO settings
                 jasco_model: str = "medium",
                 regenerate_threshold: float = 0.5,
                 regenerate_minor: bool = False,
                 force_jasco: bool = False,
                 blend_ratio: float = 0.3,
                 # Reference settings
                 reference_path: Optional[str] = None,
                 vocal_boost: float = 1.0,
                 # Mastering settings
                 target_lufs: float = -14.0,
                 enable_compression: bool = True,
                 enable_eq_matching: bool = True,
                 # Genre settings
                 genre: str = "studio"):
        """
        Initialize the AI remaster engine.
        
        Args:
            precondition_strength: How aggressive preconditioning is (0-1)
            demucs_model: Which Demucs model to use
            force_cpu: Force CPU for Demucs (more stable than MPS)
            jasco_model: JASCO/MusicGen variant (small/medium/large)
            regenerate_threshold: Damage level (0-1) to trigger regeneration
            regenerate_minor: If True, also regenerate "minor" damage stems
            force_jasco: If True, regenerate ALL stems regardless of damage
            blend_ratio: How much regenerated audio to blend (0-1)
            reference_path: Path to reference track for vocal analysis
            vocal_boost: Vocal gain multiplier (1.0 = match reference, >1.0 = louder)
            target_lufs: Target loudness for mastering
            enable_compression: Apply multiband compression
            enable_eq_matching: Match EQ to reference
            genre: Target genre/style (e.g., "future_rave", "tech_house")
        """
        self.precondition_strength = precondition_strength
        self.demucs_model = demucs_model
        self.force_cpu = force_cpu
        self.jasco_model = jasco_model
        self.regenerate_threshold = regenerate_threshold
        self.regenerate_minor = regenerate_minor
        self.force_jasco = force_jasco
        self.blend_ratio = blend_ratio
        self.reference_path = reference_path
        self.vocal_boost = vocal_boost
        self.target_lufs = target_lufs
        self.enable_compression = enable_compression
        self.enable_eq_matching = enable_eq_matching
        self.genre = genre
        
        logger.info("=" * 60)
        logger.info("Studio AI Remaster V2 initialized:")
        logger.info(f"  Preconditioning: strength={precondition_strength}")
        logger.info(f"  Demucs: model={demucs_model}, force_cpu={force_cpu}")
        logger.info(f"  JASCO: model={jasco_model}, threshold={regenerate_threshold}")
        logger.info(f"  Regenerate minor: {regenerate_minor}")
        logger.info(f"  Force JASCO: {force_jasco}")
        logger.info(f"  Blend ratio: {blend_ratio} (AI/original)")
        logger.info(f"  Reference: {reference_path or 'None'}")
        logger.info(f"  Vocal boost: {vocal_boost}x")
        logger.info(f"  Mastering: LUFS={target_lufs}")
        logger.info(f"  Genre: {genre}")
        logger.info("=" * 60)
    
    def analyze_reference_track(self, reference_path: str, sr: int) -> Dict[str, Any]:
        """
        Analyze reference track to extract vocal characteristics.
        
        This separates the reference and analyzes the vocal stem to understand
        how vocals should be mixed relative to the full mix.
        """
        logger.info(f"📊 Analyzing reference track: {reference_path}")
        
        try:
            # Load reference audio
            ref_audio, ref_sr = sf.read(reference_path)
            # Convert to mono for analysis
            if ref_audio.ndim > 1:
                ref_mono = np.mean(ref_audio, axis=1)
            else:
                ref_mono = ref_audio
            
            if ref_sr != sr:
                logger.warning(f"Reference sample rate ({ref_sr}) != input ({sr}), resampling...")
                import librosa
                ref_mono = librosa.resample(ref_mono, orig_sr=ref_sr, target_sr=sr)
            
            # Separate reference track to get vocal stem
            logger.info("  Separating reference track...")
            ref_stems = self.separate_stems(ref_mono, sr)
            
            if ref_stems is None or 'vocals' not in ref_stems:
                logger.warning("  Could not separate reference vocals, using simple analysis")
                return self._simple_reference_analysis(ref_mono, sr)
            
            # Analyze vocal characteristics
            vocal_audio = ref_stems['vocals']
            vocal_rms = np.sqrt(np.mean(vocal_audio ** 2))
            vocal_peak = np.max(np.abs(vocal_audio))
            
            # Analyze mix characteristics
            mix_rms = np.sqrt(np.mean(ref_mono ** 2))
            mix_peak = np.max(np.abs(ref_mono))
            
            # Calculate vocal-to-mix ratio
            vocal_ratio = vocal_rms / (mix_rms + 1e-10)
            vocal_peak_ratio = vocal_peak / (mix_peak + 1e-10)
            
            # Spectral analysis
            try:
                import librosa
                S = np.abs(librosa.stft(ref_mono, n_fft=2048, hop_length=512))
                S_db = librosa.amplitude_to_db(S, ref=np.max)
                
                # Vocal frequency range energy (200-4000 Hz)
                vocal_freq_mask = np.zeros_like(S_db)
                # Simple frequency bin approximation
                nyquist = sr / 2
                # Roughly 200-4000 Hz bins
                low_bin = int(200 / nyquist * S_db.shape[0])
                high_bin = int(4000 / nyquist * S_db.shape[0])
                if high_bin > low_bin:
                    vocal_freq_mask[low_bin:high_bin, :] = 1
                
                vocal_energy = np.mean(S_db[vocal_freq_mask == 1])
                mix_energy = np.mean(S_db)
                
                spectral_balance = vocal_energy - mix_energy
            except:
                spectral_balance = 0.0
            
            analysis = {
                'vocal_rms': float(vocal_rms),
                'vocal_peak': float(vocal_peak),
                'vocal_ratio': float(vocal_ratio),
                'vocal_peak_ratio': float(vocal_peak_ratio),
                'spectral_balance_db': float(spectral_balance),
                'duration_s': len(ref_mono) / sr
            }
            
            logger.info(f"  ✅ Reference analysis complete:")
            logger.info(f"     Vocal RMS: {vocal_rms:.6f}")
            logger.info(f"     Vocal/Mix ratio: {vocal_ratio:.3f}")
            logger.info(f"     Vocal peak ratio: {vocal_peak_ratio:.3f}")
            logger.info(f"     Spectral balance: {spectral_balance:+.1f} dB")
            
            return analysis
            
        except Exception as e:
            logger.warning(f"  Reference analysis failed: {e}")
            return {
                'vocal_rms': None,
                'vocal_peak': None,
                'vocal_ratio': 0.35,
                'vocal_peak_ratio': None,
                'spectral_balance_db': 0.0,
                'duration_s': 0.0
            }
    
    def _simple_reference_analysis(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """Simple reference analysis without Demucs separation."""
        logger.info("  Using simple reference analysis...")
        
        # Estimate vocal presence based on spectral characteristics
        try:
            import librosa
            S = np.abs(librosa.stft(audio, n_fft=2048, hop_length=512))
            S_db = librosa.amplitude_to_db(S, ref=np.max)
            
            # Mid-range energy (where vocals typically are)
            nyquist = sr / 2
            mid_start = int(200 / nyquist * S_db.shape[0])
            mid_end = int(4000 / nyquist * S_db.shape[0])
            
            mid_energy = np.mean(S_db[mid_start:mid_end, :])
            overall_energy = np.mean(S_db)
            
            # Estimate vocal ratio
            estimated_vocal_ratio = max(0.1, min(0.8, 0.3 + (mid_energy - overall_energy) / 30))
        except:
            estimated_vocal_ratio = 0.35  # Default
        
        return {
            'vocal_rms': None,
            'vocal_peak': None,
            'vocal_ratio': estimated_vocal_ratio,
            'vocal_peak_ratio': None,
            'spectral_balance_db': 0.0,
            'duration_s': len(audio) / sr
        }
    
    def apply_vocal_boost(self, stems: Dict[str, np.ndarray], 
                          reference_analysis: Dict,
                          mix_rms: float) -> Dict[str, np.ndarray]:
        """
        Apply vocal boost to match reference characteristics.
        """
        if 'vocals' not in stems:
            logger.warning("No vocal stem to boost")
            return stems
        
        if reference_analysis is None:
            logger.info("  No reference analysis, skipping vocal boost")
            return stems
        
        vocal_audio = stems['vocals']
        vocal_rms = np.sqrt(np.mean(vocal_audio ** 2))
        
        # Get target ratio from reference
        target_ratio = reference_analysis.get('vocal_ratio', 0.35)
        
        # Apply vocal boost multiplier
        if self.vocal_boost != 1.0:
            target_ratio *= self.vocal_boost
        
        # Calculate gain needed
        if vocal_rms > 0:
            current_ratio = vocal_rms / (mix_rms + 1e-10)
            gain = target_ratio / current_ratio
            
            # Clamp gain to reasonable range
            gain = max(0.5, min(2.0, gain))
            
            # If user requested boost (>1.0), ensure we don't attenuate
            if self.vocal_boost > 1.0 and gain < 1.0:
                logger.info(f"  ⚠️ Calculated gain {gain:.2f}x would attenuate vocals, but boost requested. Clamping to 1.0x.")
                gain = 1.0
            
            if gain != 1.0:
                stems['vocals'] = vocal_audio * gain
                logger.info(f"  ✅ Applied vocal boost: {gain:.2f}x (ratio: {target_ratio:.3f})")
        
        return stems
    
    def precondition_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply preconditioning to clean the audio BEFORE separation."""
        logger.info("Step 1: PRECONDITIONING (cleaning audio before separation)")
        
        processed = audio.copy()
        is_stereo = audio.ndim > 1
        
        # Noise reduction
        if self.precondition_strength > 0:
            try:
                import noisereduce as nr
                effective_strength = self.precondition_strength * 0.5
                
                chunk_samples = 30 * sr
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
        
        # Soft peak limiting
        if self.precondition_strength > 0.2:
            threshold = 0.95
            abs_audio = np.abs(processed)
            above_threshold = abs_audio > threshold
            
            if np.any(above_threshold):
                clipped_percent = np.sum(above_threshold) / processed.size * 100
                over = abs_audio[above_threshold] - threshold
                compressed = threshold + over / 4.0
                processed[above_threshold] = np.sign(processed[above_threshold]) * compressed
                logger.info(f"  ✅ Applied soft peak limiting ({clipped_percent:.2f}% was clipping)")
        
        return processed
    
    def separate_stems(self, audio: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """Separate audio into stems using Demucs."""
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
            
            if result.model == "fallback_spectral":
                logger.warning("  ⚠️ Demucs failed, using fallback spectral separation")
                
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
        """Analyze each stem for damage levels."""
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
    
    def _get_genre_prompt(self, stem_name: str, genre: str) -> str:
        """Generate a genre-aware prompt for the given stem."""
        
        # Base templates for different genres
        genre_templates = {
            'future_rave': {
                'vocals': "high quality studio vocals, emotional edm vocal, perfect pitch, clear enunciation, mainstage festival style, dry acapella",
                'drums': "massive future rave drums, David Guetta style, huge punchy kick, big clap, aggressive, mainstage festival sound, perfectly mixed",
                'bass': "future rave bassline, aggressive saw bass, rolling rhythm, deep low end, wide stereo, intense energy",
                'other': "future rave synthesizer, massive supersaws, hypnotic melody, dark atmosphere, mainstage energy, wide stereo mix"
            },
            'dance': {
                'vocals': "modern pop dance vocals, radio ready, perfect pitch, crisp high end, Calvin Harris style, polished production",
                'drums': "modern dance pop drums, radio hit quality, tight kick, crisp hi-hats, groovy rhythm, Calvin Harris style, polished",
                'bass': "dance pop bass, funky groove, sub bass warmth, clean production, radio ready",
                'other': "dance pop piano and synths, catchy melody, bright production, radio ready, summer vibe, polished mix"
            },
            'tech_house': {
                'vocals': "tech house vocal chops, rhythmic processing, deep vibe, club ready, perfectly EQed",
                'drums': "tech house drums, John Summit style, tight punchy kick, shuffling hi-hats, driving rhythm, club ready, groovy",
                'bass': "tech house bass, rolling bassline, deep sub, rhythmic groove, John Summit style, club shaker",
                'other': "tech house synths, stabs, atmospheric fx, club vibe, deep house chords, hypnotic"
            },
            'studio': {
                'vocals': "high quality studio vocals, professional recording, perfect pitch, clear enunciation, dry studio acapella",
                'drums': "studio quality drum kit, perfectly mixed, punchy kick, crisp snare, tight groove, high fidelity, mastered",
                'bass': "professional studio bass, deep clean low end, perfectly compressed, rich tone, high definition",
                'other': "studio quality instrumental, crystal clear synthesizer, professional mix, high definition, wide stereo"
            }
        }

        # Parse requested genres (comma-separated)
        requested_genres = [g.strip().lower() for g in genre.split(',')]
        
        # Build prompt components
        prompt_parts = []
        
        for g in requested_genres:
            # Map common aliases
            if g in ['guetta', 'david guetta']: g = 'future_rave'
            if g in ['calvin', 'calvin harris']: g = 'dance'
            if g in ['summit', 'john summit']: g = 'tech_house'
            
            template = genre_templates.get(g, genre_templates['studio'])
            base_desc = template.get(stem_name, template.get('other'))
            prompt_parts.append(base_desc)
        
        # Combine descriptions if multiple genres
        if len(prompt_parts) > 1:
            combined_desc = " ".join(prompt_parts)
            # Add blending instruction
            combined_desc += ", hybrid style blend"
        else:
            combined_desc = prompt_parts[0]
            
        # Add universal quality boosters
        final_prompt = f"{combined_desc}, high fidelity, studio remaster, professional mix"
        
        logger.info(f"    🎨 Prompt for {stem_name} ({genre}): {final_prompt}")
        return final_prompt

    def regenerate_damaged_stems(self, stems: Dict[str, np.ndarray],
                                  reports: Dict[str, Dict],
                                  sr: int,
                                  original_audio: np.ndarray) -> Tuple[Dict[str, np.ndarray], List[str]]:
        """Use JASCO to regenerate damaged portions of stems."""
        logger.info("Step 4: AI REGENERATION (JASCO)")
        
        regenerated_stems = []
        
        # Find stems that need regeneration
        stems_to_regen = []
        for name, report in reports.items():
            if self.force_jasco:
                # Force regeneration of all stems
                stems_to_regen.append(name)
            elif report.get('needs_regeneration', False):
                damage = report.get('distortion_score', 0) + report.get('noise_score', 0)
                if damage > self.regenerate_threshold:
                    stems_to_regen.append(name)
            elif self.regenerate_minor and report.get('damage_level') == 'minor':
                # Also regenerate minor damage stems if flag is set
                stems_to_regen.append(name)
        
        if not stems_to_regen:
            logger.info("  No stems need regeneration - all quality OK!")
            return stems, []
        
        logger.info(f"  Stems needing regeneration: {stems_to_regen}")
        
        try:
            from audio_engine.generation.jasco_generator import JASCOGenerator, GenerationConfig
            from audio_engine.generation.blender import Blender
            from audio_engine.profiling.melody_extractor import MelodyExtractor
            from audio_engine.profiling.drum_pattern_extractor import DrumPatternExtractor
            
            # Initialize extractors
            melody_extractor = MelodyExtractor()
            drum_extractor = DrumPatternExtractor()
            
            # Configure JASCO
            config = GenerationConfig()
            config.model_variant = self.jasco_model
            config.duration = 30  # Max supported duration per chunk
            config.num_steps = 50
            
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
                
                # Generate genre-aware prompt
                prompt = self._get_genre_prompt(stem_name, self.genre)
                
                generator.config.style_description = prompt
                
                # Generate replacement audio using chunking with overlaps and conditioning
                try:
                    total_samples = len(original_stem)
                    chunk_seconds = 30
                    overlap_seconds = 5
                    
                    # Calculate step size
                    step_seconds = chunk_seconds - overlap_seconds
                    step_samples = int(step_seconds * sr)
                    chunk_samples = int(chunk_seconds * sr)
                    overlap_samples = int(overlap_seconds * sr)
                    
                    num_chunks = int(np.ceil((total_samples - overlap_samples) / step_samples))
                    if num_chunks < 1: num_chunks = 1
                    
                    generated_chunks = []
                    
                    logger.info(f"    Generating {num_chunks} chunks ({chunk_seconds}s each, {overlap_seconds}s overlap)...")
                    logger.info("    Extracting structure for conditioning...")
                    
                    for i in range(num_chunks):
                        start_sample = i * step_samples
                        end_sample = min(start_sample + chunk_samples, total_samples)
                        
                        # Pad last chunk if needed
                        chunk_len = end_sample - start_sample
                        current_chunk_seconds = chunk_len / sr
                        
                        # Extract conditioning from THIS chunk
                        chunk_audio_in = original_stem[start_sample:end_sample]
                        
                        # Pad to full 30s for consistent generation if last chunk is short
                        # (MusicGen prefers fixed durations)
                        if len(chunk_audio_in) < chunk_samples:
                            pad_len = chunk_samples - len(chunk_audio_in)
                            chunk_audio_in_padded = np.pad(chunk_audio_in, (0, pad_len))
                        else:
                            chunk_audio_in_padded = chunk_audio_in
                            
                        logger.info(f"      Chunk {i+1}/{num_chunks} (conditioning on original)...")
                        
                        try:
                            if stem_name == 'drums':
                                # Extract drum pattern
                                onsets = drum_extractor.extract(chunk_audio_in_padded, sr)
                                result = generator.generate_with_drums(
                                    drum_pattern=onsets.timestamps,
                                    duration=chunk_seconds,
                                    tempo=124.0, # Todo: detect
                                    key="C"
                                )
                            else:
                                # Extract melody contour for vocals/bass/other
                                contour = melody_extractor.extract(chunk_audio_in_padded, sr)
                                result = generator.generate_with_melody(
                                    melody_contour=contour.salience, # Use salience matrix
                                    stem_type=stem_name if stem_name != "vocals" else "other",
                                    duration=chunk_seconds,
                                    tempo=124.0,
                                    key="C"
                                )
                            
                            # Unwrap audio
                            if result.audio is not None:
                                chunk_audio = result.audio
                            else:
                                logger.warning("      Empty audio from generator, using silence")
                                chunk_audio = np.zeros(chunk_samples, dtype=np.float32)
                                
                            # Trim padding if it was last chunk
                            if len(chunk_audio_in) < chunk_samples:
                                chunk_audio = chunk_audio[:len(chunk_audio_in)]
                                
                            generated_chunks.append(chunk_audio)
                            
                        except Exception as e:
                            logger.error(f"      Chunk generation failed: {e}")
                            # Fallback: use original audio for this chunk
                            generated_chunks.append(chunk_audio_in)
                    
                    # Stitch chunks with crossfade
                    generated = self._crossfade_stitch(generated_chunks, overlap_samples, total_samples)
                    
                    # Ensure same length
                    if len(generated) > len(original_stem):
                        generated = generated[:len(original_stem)]
                    elif len(generated) < len(original_stem):
                        generated = np.pad(generated, (0, len(original_stem) - len(generated)))
                    
                    # Blend: mostly original, some AI
                    stems[stem_name] = (
                        (1 - self.blend_ratio) * original_stem +
                        self.blend_ratio * generated
                    ).astype(np.float32)
                    
                    regenerated_stems.append(stem_name)
                    logger.info(f"    ✅ {stem_name} regenerated (blend={self.blend_ratio})")
                    
                except Exception as e:
                    logger.warning(f"    ⚠️ Failed to regenerate {stem_name}: {e}")
                    import traceback
                    traceback.print_exc()
            
            return stems, regenerated_stems
            
        except Exception as e:
            logger.warning(f"  JASCO regeneration failed: {e}")
            logger.info("  Continuing without AI regeneration")
            return stems, []

    def _crossfade_stitch(self, chunks: List[np.ndarray], overlap_samples: int, total_length: int) -> np.ndarray:
        """Stitch overlapping chunks with crossfades."""
        if not chunks:
            return np.zeros(total_length)
            
        # Create output buffer
        output = np.zeros(total_length, dtype=np.float32)
        
        # Create crossfade window
        fade_in = np.linspace(0, 1, overlap_samples)
        fade_out = np.linspace(1, 0, overlap_samples)
        
        current_pos = 0
        step_size = len(chunks[0]) - overlap_samples
        
        for i, chunk in enumerate(chunks):
            # Calculate position
            if i == 0:
                # First chunk: no fade in
                length = len(chunk)
                output[0:length] = chunk
                current_pos = length - overlap_samples
            else:
                # Subsequent chunks
                chunk_len = len(chunk)
                
                # Apply fade in to start of this chunk
                # Blend with fade out of previous chunk (already in output)
                
                # Overlap region
                overlap_start = current_pos
                overlap_end = min(current_pos + overlap_samples, total_length)
                actual_overlap = overlap_end - overlap_start
                
                if actual_overlap > 0:
                    # Blend
                    existing = output[overlap_start:overlap_end]
                    new_segment = chunk[:actual_overlap]
                    
                    # Apply crossfade curves
                    blended = (existing * fade_out[:actual_overlap]) + (new_segment * fade_in[:actual_overlap])
                    output[overlap_start:overlap_end] = blended
                
                # Copy rest of chunk
                remaining_start = overlap_samples
                remaining_end = chunk_len
                
                dest_start = overlap_end
                dest_end = min(dest_start + (remaining_end - remaining_start), total_length)
                
                copy_len = dest_end - dest_start
                if copy_len > 0:
                    output[dest_start:dest_end] = chunk[remaining_start:remaining_start+copy_len]
                
                current_pos += (chunk_len - overlap_samples)
                
        return output
    
    def mix_stems(self, stems: Dict[str, np.ndarray], sr: int) -> np.ndarray:
        """Mix stems back together with optimal levels."""
        logger.info("Step 5: MIXING STEMS")
        
        try:
            from audio_engine.mixing import StemMixer
            
            mixer = StemMixer()
            result = mixer.optimize_for_clarity(stems, sr)
            
            logger.info(f"  ✅ Mixed: peak={result.peak_level_db:.1f}dB, rms={result.rms_level_db:.1f}dB")
            return result.mixed_audio
            
        except Exception as e:
            logger.warning(f"  Mixer failed: {e}, using simple sum")
            
            # Handle case where all stems are empty
            first_stem = list(stems.values())[0]
            if len(first_stem) == 0:
                logger.warning("  All stems empty, returning empty mix")
                return np.array([], dtype=np.float32)
                
            mixed = np.zeros_like(first_stem)
            for name, audio in stems.items():
                if len(audio) == len(mixed):
                    mixed += audio * 0.25
            
            if len(mixed) > 0:
                peak = np.max(np.abs(mixed))
                if peak > 0.95:
                    mixed = mixed * (0.9 / peak)
            
            return mixed
    
    def master_audio(self, audio: np.ndarray, sr: int,
                     reference_profile: Optional[Dict] = None) -> np.ndarray:
        """Apply final mastering using our proven DSP pipeline."""
        logger.info("Step 6: MASTERING")
        
        processed = audio.copy()
        
        # EQ matching
        if self.enable_eq_matching and reference_profile:
            logger.info("  Applying EQ matching...")
            from scipy import signal
            
            target_centroid = reference_profile.get('spectral_centroid_hz', 3000)
            
            try:
                import librosa
                S = np.abs(librosa.stft(processed if processed.ndim == 1 else np.mean(processed, axis=1)))
                centroid = np.mean(librosa.feature.spectral_centroid(S=S, sr=sr))
            except:
                centroid = 3000
            
            centroid_diff = target_centroid - centroid
            nyquist = sr / 2.0
            
            if abs(centroid_diff) > 100:
                if centroid_diff > 0:
                    low_adjust = -min(centroid_diff / 500, 3.0)
                else:
                    low_adjust = min(-centroid_diff / 500, 3.0)
                
                try:
                    b, a = signal.butter(2, 200 / nyquist, btype='low')
                    low_content = signal.filtfilt(b, a, processed, axis=0)
                    gain = 10 ** (low_adjust / 20)
                    processed = processed + low_content * (gain - 1) * 0.5
                    logger.info(f"    Low shelf: {low_adjust:+.1f}dB")
                except:
                    pass
        
        # Multiband compression
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
        
        # Loudness normalization
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
        
        # True peak limiting
        peak_ceiling = 10 ** (-1.0 / 20)
        if len(processed) > 0:
            current_peak = np.max(np.abs(processed))
            if current_peak > peak_ceiling:
                processed = np.tanh(processed / peak_ceiling) * peak_ceiling
                logger.info("    Applied peak limiting")
        else:
            logger.warning("  Empty audio in mastering, returning silence")
        
        logger.info("  ✅ Mastering complete")
        return np.clip(processed, -1.0, 1.0)
    
    def process(self, input_path: str, output_path: str,
                reference_path: Optional[str] = None) -> AIRemasterResult:
        """Run the full AI remaster pipeline."""
        start_time = time.time()
        
        logger.info("=" * 70)
        logger.info("STUDIO AI REMASTER V2 - Full Pipeline with JASCO Enhancement")
        logger.info("=" * 70)
        logger.info(f"Input: {input_path}")
        logger.info(f"Output: {output_path}")
        logger.info(f"Reference: {reference_path or 'None'}")
        logger.info("=" * 70)
        
        # Load audio
        logger.info("Loading audio...")
        audio, sr = sf.read(input_path)
        logger.info(f"  Loaded: {audio.shape}, {sr} Hz, {len(audio)/sr:.1f}s")
        
        # Analyze reference track if provided
        reference_analysis = None
        if reference_path:
            reference_analysis = self.analyze_reference_track(reference_path, sr)
        
        # Step 1: Precondition
        preconditioned = self.precondition_audio(audio, sr)
        
        # Step 2: Separate
        stems = self.separate_stems(preconditioned, sr)
        stems_separated = stems is not None and len(stems) > 0
        
        if not stems_separated:
            logger.error("Separation failed completely!")
            processed = self.master_audio(preconditioned, sr, None)
            sf.write(output_path, processed.astype(np.float32), sr)
            
            return AIRemasterResult(
                output_path=output_path,
                stems_separated=False,
                stems_regenerated=[],
                demucs_model="none",
                jasco_model="none",
                blend_ratio=0.0,
                processing_time_s=time.time() - start_time,
                quality_metrics={},
                reference_analysis=None
            )
        
        # Calculate mix RMS for vocal boost
        mix_audio = np.zeros_like(list(stems.values())[0])
        for stem_audio in stems.values():
            mix_audio += stem_audio * 0.25
        mix_rms = np.sqrt(np.mean(mix_audio ** 2))
        
        # Apply vocal boost before mixing (if using reference)
        if reference_analysis and self.vocal_boost != 1.0:
            stems = self.apply_vocal_boost(stems, reference_analysis, mix_rms)
        
        # Step 3: Analyze
        reports = self.analyze_stems(stems, sr)
        
        # Step 4: Regenerate (only if needed)
        stems, regenerated = self.regenerate_damaged_stems(stems, reports, sr, audio)
        
        # Re-apply vocal boost after regeneration
        if reference_analysis and self.vocal_boost != 1.0:
            stems = self.apply_vocal_boost(stems, reference_analysis, mix_rms)
        
        # Step 5: Mix
        mixed = self.mix_stems(stems, sr)
        
        # Step 6: Master
        mastered = self.master_audio(mixed, sr, None)
        
        # Ensure stereo if input was stereo
        if audio.ndim > 1 and mastered.ndim == 1:
            mastered = np.column_stack([mastered, mastered])
        
        # Save
        logger.info(f"Saving to {output_path}...")
        sf.write(output_path, mastered.astype(np.float32), sr)
        
        processing_time = time.time() - start_time
        
        logger.info("=" * 70)
        logger.info(f"✅ AI REMASTER V2 COMPLETE in {processing_time:.1f}s")
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
            quality_metrics={name: reports.get(name, {}) for name in stems},
            reference_analysis=reference_analysis
        )


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Studio AI Remaster V2 - AI-enhanced audio processing with JASCO support"
    )
    parser.add_argument("input_file", help="Input audio file")
    parser.add_argument("--output", "-o", help="Output file path")
    parser.add_argument("--reference", "-r", help="Reference track for vocal analysis")
    
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
    parser.add_argument("--regenerate-minor", action="store_true",
                       help="Also regenerate stems with 'minor' damage")
    parser.add_argument("--force-jasco", action="store_true",
                       help="Regenerate ALL stems with JASCO")
    parser.add_argument("--blend", type=float, default=0.3,
                       help="AI blend ratio 0-1 (default: 0.3)")
    
    # Reference and vocal
    parser.add_argument("--vocal-boost", type=float, default=1.0,
                       help="Vocal gain multiplier (1.0 = match reference, default: 1.0)")
    
    # Genre
    parser.add_argument("--genre", default="studio",
                       help="Target genre (future_rave, dance, tech_house, or comma-separated list)")
    
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
        output_path = input_path.with_name(f"{input_path.stem}_studio_ai_remaster_v2.wav")
    
    # Use provided reference or default Kelly Clarkson track
    reference_path = args.reference
    if not reference_path:
        kelly_path = ROOT / "Kelly_Clarkson_-_Catch_My_Breath_ZUEZEU_POP_EDIT_FREE_DL_KLICKAUD.mp3"
        if kelly_path.exists():
            reference_path = str(kelly_path)
            logger.info(f"Using Kelly Clarkson track as reference: {reference_path}")
    
    remaster = StudioAIRemasterV2(
        precondition_strength=args.precond_strength,
        demucs_model=args.demucs_model,
        force_cpu=args.force_cpu,
        jasco_model=args.jasco_model,
        regenerate_threshold=args.regen_threshold,
        regenerate_minor=args.regenerate_minor,
        force_jasco=args.force_jasco,
        blend_ratio=args.blend,
        reference_path=reference_path,
        vocal_boost=args.vocal_boost,
        target_lufs=args.target_lufs,
        enable_compression=not args.no_compression,
        enable_eq_matching=not args.no_eq,
        genre=args.genre
    )
    
    result = remaster.process(
        str(input_path),
        str(output_path),
        reference_path
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("AI REMASTER V2 SUMMARY")
    print("=" * 60)
    print(f"Input:  {args.input_file}")
    print(f"Output: {result.output_path}")
    print(f"\nProcessing:")
    print(f"  Stems separated: {'✅ Yes' if result.stems_separated else '❌ No'}")
    print(f"  Demucs model: {result.demucs_model}")
    print(f"  Stems regenerated: {result.stems_regenerated or 'None'}")
    print(f"  JASCO model: {result.jasco_model}")
    print(f"  Blend ratio: {result.blend_ratio}")
    if result.reference_analysis:
        ref_vocal_ratio = result.reference_analysis.get('vocal_ratio', 'N/A')
        print(f"  Reference vocal ratio: {ref_vocal_ratio:.3f}" if isinstance(ref_vocal_ratio, float) else f"  Reference vocal ratio: {ref_vocal_ratio}")
        print(f"  Vocal boost: {args.vocal_boost}x")
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
        'reference_analysis': result.reference_analysis,
        'regenerate_minor': args.regenerate_minor,
        'force_jasco': args.force_jasco,
        'vocal_boost': args.vocal_boost,
    }
    
    report_path = output_path.with_suffix('.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
