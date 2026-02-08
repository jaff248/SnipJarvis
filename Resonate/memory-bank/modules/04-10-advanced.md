# Modules 04-10: Advanced Processing & Infrastructure

## Module 04: Frequency Restoration

### Purpose
Extend limited phone bandwidth (~100Hz-8kHz) to full spectrum (20Hz-20kHz) using harmonic synthesis and spectral modeling.

---

## ✨ NEW: Module 11 - JASCO Creative Reconstruction (Experimental v2)

### Overview
**JASCO** (Joint Audio and Symbolic Conditioning for Temporally Controlled Text-to-Music Generation) enables a fundamentally different approach: instead of *restoring* the original, we *regenerate* studio-quality versions based on extracted characteristics.

**⚠️ CRITICAL DISTINCTION:**
- **Restoration Mode** (Modules 01-10): "Reveal what was played" → Authentic but limited by input quality
- **Enhancement Mode** (MBD): "Polish what was restored" → Higher fidelity, still authentic
- **Generation Mode** (JASCO): "Recreate what we think was played" → High quality but AI-generated

---

## 🔥 Module 12 - MultiBandDiffusion Enhancement (v1.5 Feature)

### Overview
**MultiBandDiffusion (MBD)** is NOT a generative model - it's a **high-fidelity decoder** that reduces artifacts and improves audio quality. This is HIGHLY relevant for restoration!

### What MBD Does
From the [AudioCraft MBD docs](https://github.com/facebookresearch/audiocraft/blob/main/docs/MBD.md):

> "MultiBand diffusion is a collection of 4 models that can decode tokens from EnCodec tokenizer into waveform audio... improves the perceived quality and **reduces the artifacts** coming from adversarial decoders."

### Key Insight: The `regenerate()` Method

```python
from audiocraft.models import MultiBandDiffusion

mbd = MultiBandDiffusion.get_mbd_24khz(bw=6.0)  # 6 kbps = highest quality

# THIS IS THE MAGIC:
# Takes ANY audio → EnCodec compress → MBD diffusion decode → Higher quality output
enhanced = mbd.regenerate(restored_audio, sample_rate=24000)
```

**What happens inside `regenerate()`:**
1. Audio → EnCodec encoder → Discrete tokens (compression)
2. Tokens → EnCodec latent space (continuous embeddings)
3. Embeddings → 4 parallel diffusion models (one per frequency band)
4. Diffusion output → Summed → High-fidelity waveform

### Why MBD is Perfect for Restoration

| Problem | MBD Solution |
|---------|--------------|
| GAN decoder artifacts (metallic, harsh) | Diffusion produces smoother output |
| Missing high frequencies | EnCodec trained on full bandwidth |
| Quantization noise | Diffusion denoising removes it |
| Harsh transients | Multi-band processing handles each range |

### Architecture: How It Fits Our Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      ENHANCED RESTORATION PIPELINE                       │
│                                                                          │
│  ┌────────────┐   ┌────────────┐   ┌────────────┐   ┌────────────────┐  │
│  │ SEPARATION │ → │ ENHANCE    │ → │ MIX/MASTER │ → │ MBD POLISH     │  │
│  │ (Demucs)   │   │ per-stem   │   │            │   │ (NEW!)         │  │
│  └────────────┘   └────────────┘   └────────────┘   └────────────────┘  │
│                                                              │          │
│                                                              ▼          │
│                                              ┌──────────────────────┐   │
│                                              │ EnCodec → MBD Decode │   │
│                                              │ (Artifact reduction) │   │
│                                              │ (Frequency polish)   │   │
│                                              └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

### Implementation

```python
import torch
from audiocraft.models import MultiBandDiffusion
import julius

class MBDEnhancer:
    """
    MultiBandDiffusion-based audio enhancement.
    Uses neural codec + diffusion decoding for artifact reduction.
    """
    
    def __init__(self, bandwidth: float = 6.0, device: str = 'mps'):
        """
        Args:
            bandwidth: EnCodec bandwidth (1.5, 3.0, or 6.0 kbps)
                      Higher = better quality but more processing
            device: Compute device
        """
        # MBD at 24kHz with specified bandwidth
        self.mbd = MultiBandDiffusion.get_mbd_24khz(bw=bandwidth)
        self.device = device
        self.target_sr = 24000  # MBD operates at 24kHz
        
    def enhance(
        self,
        audio: torch.Tensor,
        sample_rate: int,
        n_bands: int = 32,
        strictness: float = 0.8
    ) -> torch.Tensor:
        """
        Enhance audio through MBD regeneration.
        
        Args:
            audio: Input audio tensor [B, C, T]
            sample_rate: Input sample rate
            n_bands: Number of EQ bands for matching
            strictness: How strictly to match original EQ (0-1)
                       Lower = more MBD character, Higher = closer to original
        
        Returns:
            Enhanced audio at input sample rate
        """
        # Resample to 24kHz if needed
        if sample_rate != self.target_sr:
            audio_24k = julius.resample_frac(audio, sample_rate, self.target_sr)
        else:
            audio_24k = audio
            
        # The magic: regenerate through MBD
        with torch.no_grad():
            # Get EnCodec embeddings
            emb = self.mbd.get_condition(audio_24k, sample_rate=self.target_sr)
            
            # Generate via diffusion
            size = audio_24k.size()
            enhanced = self.mbd.generate(emb, size=size)
            
            # Match EQ to original (prevents over-processing)
            enhanced = self.mbd.re_eq(
                wav=enhanced,
                ref=audio_24k,
                n_bands=n_bands,
                strictness=strictness
            )
        
        # Resample back to original rate
        if sample_rate != self.target_sr:
            enhanced = julius.resample_frac(enhanced, self.target_sr, sample_rate)
            
        return enhanced
    
    def enhance_stems(
        self,
        stems: dict,
        sample_rate: int,
        strictness: float = 0.8
    ) -> dict:
        """
        Enhance each stem individually through MBD.
        Better results than enhancing the mix.
        
        Args:
            stems: Dict of stem_name -> audio tensor
            sample_rate: Sample rate of stems
            
        Returns:
            Dict of enhanced stems
        """
        enhanced_stems = {}
        for name, audio in stems.items():
            enhanced_stems[name] = self.enhance(
                audio,
                sample_rate,
                # Vocals get gentler treatment
                strictness=0.9 if name == 'vocals' else strictness
            )
        return enhanced_stems
```

### When to Use MBD vs Skip It

| Scenario | Use MBD? | Reason |
|----------|----------|--------|
| Phone recording has harsh artifacts | ✅ Yes | MBD reduces harshness |
| Audio already sounds clean | ❌ Skip | May add unnecessary processing |
| Need to preserve exact transients | ⚠️ Careful | Use high strictness (0.9+) |
| Drums/percussion | ⚠️ Careful | Test first, transients may soften |
| Vocals | ✅ Yes | MBD handles vocals well |
| Want maximum quality | ✅ Yes | Final polish step |

### Bandwidth Selection

```python
# Lower bandwidth = more compression = more "interpretation" by MBD
# Higher bandwidth = less compression = closer to original

BANDWIDTH_GUIDE = {
    1.5: "Aggressive enhancement, may change character",
    3.0: "Balanced, good for most restoration",
    6.0: "Conservative, minimal character change"  # Recommended
}
```

### The Complete AudioCraft Integration Map

```
┌───────────────────────────────────────────────────────────────────────────┐
│                     AUDIOCRAFT FOR LIVE MUSIC RESTORATION                  │
│                                                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                        RESTORATION MODE (v1)                         │  │
│  │                                                                      │  │
│  │  Phone Recording → Demucs → DSP Enhancement → Mix/Master            │  │
│  │                                     │                                │  │
│  │                                     ▼                                │  │
│  │                              MBD Polish (v1.5)                       │  │
│  │                        (EnCodec + Diffusion decode)                  │  │
│  │                                     │                                │  │
│  │                                     ▼                                │  │
│  │                           ✅ RESTORED OUTPUT                         │  │
│  │                         (Authentic, enhanced)                        │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                        GENERATION MODE (v2)                          │  │
│  │                                                                      │  │
│  │  Phone Recording → Demucs → Profile (chords/melody/drums/style)     │  │
│  │                                     │                                │  │
│  │                                     ▼                                │  │
│  │                              JASCO Generate                          │  │
│  │                    (Text + chords + drums + melody)                  │  │
│  │                                     │                                │  │
│  │                                     ▼                                │  │
│  │                           ✨ GENERATED OUTPUT                        │  │
│  │                         (High quality, AI-created)                   │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                    NOT USED (Violates principles)                    │  │
│  │                                                                      │  │
│  │  ❌ MusicGen - Would hallucinate music content                       │  │
│  │  ❌ AudioGen - For sound effects, not music restoration              │  │
│  │  ❌ MAGNeT - Non-autoregressive generation, same issue               │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────────┘
```

### Dependencies (Additional for MBD)

```txt
# MBD requires same deps as AudioCraft
audiocraft>=1.3.0
# EnCodec for the compression model
encodec>=0.1.1
```

### Implementation Priority

**Phase 1 (v1)**: Core restoration (Demucs + DSP)
**Phase 1.5**: Add MBD as optional post-processing polish
**Phase 2 (v2)**: JASCO generation mode

### Key Difference: MBD vs JASCO

| Aspect | MBD | JASCO |
|--------|-----|-------|
| **What it does** | Decodes tokens to high-fidelity audio | Generates new audio from conditions |
| **Input** | Encoded audio tokens | Text + chords + drums + melody |
| **Output** | Enhanced version of input | New audio matching conditions |
| **Authentic?** | ✅ Yes, same content | ❌ No, AI interpretation |
| **Use case** | Polish restoration | Recreate when restoration fails |
| **Philosophy fit** | "Reveal" (perfect) | "Recreate" (opt-in) |

### What JASCO Can Do
From analysis of the [AudioCraft JASCO docs](https://github.com/facebookresearch/audiocraft/blob/main/docs/JASCO.md):

| Conditioning Type | Input | Description |
|------------------|-------|-------------|
| **Text** | "Pop vocals like Tate McRae, energetic synths" | Global style/mood |
| **Chords** | `[('C', 0.0), ('D', 2.0), ('F', 4.0)]` | Harmonic progression (timing-locked) |
| **Drums** | Extracted drum stem WAV | Rhythmic pattern (beat-locked) |
| **Melody** | Salience matrix (pitch over time) | Melodic contour |

### Proposed Workflow: Profile → Generate → Blend

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CREATIVE RECONSTRUCTION PIPELINE                  │
│                                                                      │
│  ┌────────────────┐   ┌────────────────┐   ┌─────────────────┐     │
│  │  1. SEPARATE   │ → │  2. PROFILE    │ → │  3. GENERATE    │     │
│  │  (Demucs)      │   │  Each Stem     │   │  (JASCO)        │     │
│  └────────────────┘   └────────────────┘   └─────────────────┘     │
│        │                     │                     │                │
│        ▼                     ▼                     ▼                │
│   [Vocals]            [Style: "Tate McRae"]   [New Vocals @        │
│   [Drums]             [Tempo: 120 BPM]         Studio Quality]     │
│   [Bass]              [Key: C major]          [New Drums]          │
│   [Other]             [Chords: extracted]     [New Bass]           │
│                       [Melody: contour]       [New Instruments]    │
│                                                                      │
│  ┌────────────────┐   ┌────────────────────────────────────────┐   │
│  │  4. COMPARE    │ → │  5. BLEND / CHOOSE                     │   │
│  │  Side-by-side  │   │  User picks restored vs generated      │   │
│  └────────────────┘   └────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### Implementation Strategy

```python
from audiocraft.models import JASCO
import torch

class JASCOGenerator:
    """Generate studio-quality stems from profiled characteristics"""
    
    def __init__(self):
        # Load JASCO model (1B params for best quality)
        self.model = JASCO.get_pretrained(
            'facebook/jasco-chords-drums-melody-1B',
            chords_mapping_path='assets/chord_to_index_mapping.pkl'
        )
        self.model.set_generation_params(
            cfg_coef_all=1.75,  # Temporal control strength
            cfg_coef_txt=3.5   # Text conditioning strength
        )
        
    def generate_from_profile(
        self,
        profile: StemProfile,
        segment_duration: float = 10.0
    ) -> torch.Tensor:
        """
        Generate new audio from extracted profile
        
        Args:
            profile: Extracted characteristics from phone recording
            segment_duration: JASCO generates 10s chunks
            
        Returns:
            Generated audio tensor
        """
        return self.model.generate_music(
            descriptions=[profile.text_description],
            chords=profile.chord_progression,
            drums_wav=profile.drums_stem,  # Use actual extracted drums!
            melody_salience_matrix=profile.melody_contour,
            segment_duration=segment_duration,
            progress=True
        )
```

### Profiling Pipeline

```python
@dataclass
class StemProfile:
    """Extracted characteristics for JASCO generation"""
    
    # Text description (user + auto-detected)
    text_description: str  # "Female pop vocals, energetic synths, electronic drums"
    
    # Extracted from audio
    chord_progression: List[Tuple[str, float]]  # [('C', 0.0), ('Am', 2.5), ...]
    melody_contour: torch.Tensor  # Salience matrix (pitch x time)
    drums_stem: torch.Tensor  # Actual drums from Demucs separation
    
    # Detected characteristics
    tempo_bpm: float
    key: str
    time_signature: str
    vocal_style: str  # Auto-detected or user-specified

class StemProfiler:
    """Extract characteristics from separated stems"""
    
    def __init__(self):
        # Chord extraction (using Chordino)
        self.chord_extractor = ChordinoExtractor()
        
        # Melody extraction (salience maps)
        self.melody_extractor = MelodySalienceExtractor()
        
        # Tempo/key detection
        self.rhythm_analyzer = RhythmAnalyzer()
        
    def profile(self, stems: SeparatedStems, user_hints: dict = None) -> StemProfile:
        """
        Extract JASCO-compatible profile from stems
        
        Args:
            stems: Demucs-separated stems
            user_hints: Optional user guidance ("vocals sound like Tate McRae")
        """
        
        # Extract chords from full mix or 'other' stem
        chords = self.chord_extractor.extract(stems.to_mix())
        
        # Extract melody contour from vocals
        melody = self.melody_extractor.extract(
            stems.vocals,
            stems.sample_rate
        )
        
        # Detect tempo/key
        rhythm = self.rhythm_analyzer.analyze(stems.drums)
        
        # Build text description
        description = self._build_description(
            user_hints=user_hints,
            detected_key=rhythm.key,
            detected_tempo=rhythm.bpm
        )
        
        return StemProfile(
            text_description=description,
            chord_progression=chords,
            melody_contour=melody,
            drums_stem=stems.drums,
            tempo_bpm=rhythm.bpm,
            key=rhythm.key,
            time_signature=rhythm.time_sig,
            vocal_style=user_hints.get('vocal_style', 'pop vocals')
        )
        
    def _build_description(self, user_hints, detected_key, detected_tempo):
        """Build JASCO text prompt from profile"""
        
        parts = []
        
        # User-provided style hints
        if user_hints:
            if 'vocal_style' in user_hints:
                parts.append(user_hints['vocal_style'])  # "Female vocals like Tate McRae"
            if 'instrument_style' in user_hints:
                parts.append(user_hints['instrument_style'])  # "Electronic synths, punchy bass"
            if 'mood' in user_hints:
                parts.append(user_hints['mood'])  # "Energetic, danceable"
                
        # Auto-detected characteristics
        parts.append(f"in {detected_key}")
        parts.append(f"at {detected_tempo} BPM")
        
        return ", ".join(parts)
```

### Use Cases for JASCO Generation

| Scenario | Restoration Mode | Generation Mode (JASCO) |
|----------|-----------------|-------------------------|
| **Phone recording very degraded** | Limited improvement | Generate clean version |
| **Lost forever, need recreation** | Can't bring back | Recreate approximation |
| **Creative reinterpretation** | Not applicable | Generate "what if" versions |
| **Missing instruments** | Can't add | Generate matching instruments |
| **Vocals too damaged** | Best effort cleanup | Generate similar vocals |

### UI Integration

```python
# In Streamlit UI
st.header("🎵 Processing Mode")

mode = st.radio(
    "Choose approach:",
    [
        "🔧 Restoration (Authentic)",
        "✨ Generation (Creative)"
    ]
)

if mode == "✨ Generation (Creative)":
    st.warning(
        "⚠️ **Generation Mode** creates NEW audio that sounds similar to your recording. "
        "This is NOT the original performance - it's an AI interpretation based on "
        "extracted characteristics. Use when restoration isn't enough."
    )
    
    # User hints for generation
    col1, col2 = st.columns(2)
    with col1:
        vocal_style = st.text_input(
            "Vocal style hint",
            placeholder="e.g., Female pop vocals like Tate McRae"
        )
    with col2:
        instrument_style = st.text_input(
            "Instrument style",
            placeholder="e.g., Electronic synths, punchy drums"
        )
        
    # After processing, show both options
    st.subheader("Compare Results")
    
    col1, col2 = st.columns(2)
    with col1:
        st.caption("🔧 Restored (Authentic)")
        st.audio(restored_audio)
        st.metric("SNR Improvement", "+18 dB")
        
    with col2:
        st.caption("✨ Generated (Creative)")
        st.audio(generated_audio)
        st.metric("Quality", "Studio Grade")
        
    # Let user blend or choose
    blend = st.slider(
        "Blend (Restored ↔ Generated)",
        0.0, 1.0, 0.5,
        help="Mix between authentic restoration and AI generation"
    )
```

### Key Constraints & Limitations

1. **10 second limit**: JASCO generates 10s chunks (need stitching for longer)
2. **Not the original**: Generated audio is a NEW creation, not restoration
3. **Vocal synthesis**: JASCO doesn't clone specific voices (generates similar style)
4. **Legal/ethical**: Generated content may have different copyright status
5. **Memory**: 1B model requires significant VRAM (~8GB+)

### When to Use JASCO vs Restoration

```
                        Phone Recording Quality
                    ┌─────────────────────────────────┐
                    │                                 │
         Bad       │  ← JASCO Generation Best →     │  Good
      (SNR < 5dB)  │                                 │ (SNR > 15dB)
                    │                                 │
                    └─────────────────────────────────┘
                              
                    ← Restoration Best →
```

**Decision Tree:**
1. Is recording recoverable with restoration? → Use Restoration
2. Is authentic performance critical? → Use Restoration (even if degraded)
3. Need "studio quality" regardless of authenticity? → Use JASCO Generation
4. Want to explore "what if" variations? → Use JASCO Generation

### Dependencies (Additional)

```txt
# For JASCO support
audiocraft>=1.3.0          # Includes JASCO models
xformers>=0.0.22           # Required for JASCO v1 models
chord-extractor            # Chordino for chord extraction
```

### Implementation Priority

**Phase 1 (v1)**: Restoration only (no JASCO)
**Phase 2 (v1.1)**: Add profiling infrastructure
**Phase 3 (v2)**: JASCO generation as experimental feature
**Phase 4 (v2.1)**: Blending/comparison tools

### Philosophy: Two Modes, Clear Labels

```
┌────────────────────────────────────────────────────────────────────┐
│                                                                     │
│   RESTORATION MODE          │          GENERATION MODE              │
│   "Reveal what was played"  │    "Recreate based on profile"        │
│                             │                                       │
│   ✅ Authentic              │    ✅ High quality                    │
│   ✅ Original performance   │    ✅ Can recover "unrecoverable"     │
│   ❌ Limited by input       │    ❌ Not the original                │
│   ❌ Can't add missing      │    ❌ AI interpretation               │
│                             │                                       │
│   Output: "Restored"        │    Output: "AI-Generated"             │
│                             │                                       │
└────────────────────────────────────────────────────────────────────┘
```

This way users can choose: **authenticity** vs **quality**, with full transparency about what they're getting.

### Core Strategy
```python
class FrequencyRestorer:
    """Restore frequency content lost in phone recording"""
    
    def restore_sub_bass(self, audio, fundamental_freq=40):
        """Generate sub-bass from bass harmonics (80-200 Hz → 20-80 Hz)"""
        # Octave-down synthesis: fundamental / 2
        
    def restore_highs(self, audio, cutoff=8000):
        """Extend high frequencies (8kHz+ using harmonic extrapolation)"""
        # Method 1: Harmonic synthesis (safe, predictable)
        # Method 2: Spectral envelope matching (experimental)
```

### Techniques
1. **Sub-Bass Generation**: Synthesize octave-below from bass fundamental
2. **High-Frequency Extension**: Harmonic exciter + spectral modeling
3. **Bandwidth Detection**: Auto-detect phone mic limitations
4. **Natural Blending**: Crossfade generated content with original

### Key Decision
Start with **harmonic synthesis** (v1), research ML-based extension (v2).

---

## Module 05: De-reverberation

### Purpose
Remove venue acoustics (reverb/echo) from live recording to reveal dry, direct sound.

### Core Strategy
```python
class Dereverberator:
    """Remove venue reverb from live recording"""
    
    def __init__(self, method="wiener"):  # "wiener" or "wpe" or "spectral"
        self.method = method
        
    def estimate_reverb(self, audio):
        """Estimate room impulse response"""
        # Detect reverb tail length (RT60)
        # Estimate early reflections vs late reverb
        
    def remove_reverb(self, audio, strength=0.5):
        """Apply de-reverberation"""
        # Wiener filtering with IR estimation
        # Or: Weighted Prediction Error (WPE) algorithm
```

### Algorithms
1. **Wiener Filtering**: Statistical approach, needs IR estimation
2. **WPE (Weighted Prediction Error)**: Blind dereverberation (no IR needed)
3. **Spectral Suppression**: Frequency-domain reverb reduction
4. **Dry/Wet Blend**: User control for reverb amount

### Key Decision
Research **WPE** or **DNN-based** methods. If none suitable, make this a manual/optional step.

---

## Module 06: Mixing

### Purpose
Recombine enhanced stems with proper gain staging and spatial placement.

### Core Strategy
```python
class MixEngine:
    """Intelligent stem mixing"""
    
    def __init__(self):
        self.target_lufs = -14  # Streaming loudness standard
        
    def auto_mix(self, stems: SeparatedStems) -> np.ndarray:
        """
        Automatic mixing with sensible defaults
        
        Steps:
        1. Level matching (balance stems)
        2. Panning (optional stereo width)
        3. Master bus EQ (gentle glue)
        4. Sum to stereo/mono
        """
        
        # Level targets (relative dB)
        levels = {
            "vocals": 0,      # Reference (loudest)
            "drums": -3,      # Slightly under vocals
            "bass": -6,       # Foundation, not overpowering
            "other": -8       # Background
        }
        
        # Mix with gains
        mixed = (
            stems.vocals * db_to_gain(levels["vocals"]) +
            stems.drums * db_to_gain(levels["drums"]) +
            stems.bass * db_to_gain(levels["bass"]) +
            stems.other * db_to_gain(levels["other"])
        )
        
        return mixed
```

### Features
- **Auto-leveling**: Based on RMS/LUFS per stem
- **Reference Matching**: If clean version available, match its balance
- **User Controls**: Per-stem volume sliders in UI
- **Stem Export**: Allow exporting individual stems for manual mixing in DAW

---

## Module 07: Mastering

### Purpose
Final polish: loudness normalization, limiting, dithering, and export.

### Core Strategy
```python
class MasteringEngine:
    """Final stage: loudness, limiting, export"""
    
    def master(
        self,
        mixed_audio: np.ndarray,
        target_lufs: float = -14.0,
        true_peak_db: float = -1.0,
        output_format: str = "wav",
        bit_depth: int = 16
    ) -> np.ndarray:
        """
        Master mixed audio to streaming standards
        
        Steps:
        1. Loudness normalize to -14 LUFS (Spotify/Apple standard)
        2. True peak limiting to -1 dB (prevent clipping)
        3. Dither to output bit depth (16-bit for distribution)
        """
        
        # 1. Measure loudness
        current_lufs = self._measure_lufs(mixed_audio)
        
        # 2. Apply gain to reach target
        gain_db = target_lufs - current_lufs
        normalized = mixed_audio * db_to_gain(gain_db)
        
        # 3. True peak limiter
        limited = self._true_peak_limit(normalized, true_peak_db)
        
        # 4. Dither (if reducing bit depth)
        if bit_depth < 24:
            limited = self._dither(limited, bit_depth)
            
        return limited
        
    def _measure_lufs(self, audio: np.ndarray) -> float:
        """Measure integrated loudness (LUFS)"""
        import pyloudnorm as pyln
        meter = pyln.Meter(self.sample_rate)
        return meter.integrated_loudness(audio)
        
    def _true_peak_limit(self, audio: np.ndarray, ceiling_db: float) -> np.ndarray:
        """True peak limiter (catches inter-sample peaks)"""
        from pedalboard import Limiter
        limiter = Limiter(threshold_db=ceiling_db, release_ms=50)
        return limiter(audio.flatten(), self.sample_rate).reshape(-1, 1)
        
    def _dither(self, audio: np.ndarray, target_bits: int) -> np.ndarray:
        """TPDF dithering for bit depth reduction"""
        # Triangular PDF dither (industry standard)
        dither_amount = 1.0 / (2 ** target_bits)
        dither = np.random.triangular(-dither_amount, 0, dither_amount, audio.shape)
        return audio + dither
```

### Export Options
- **Formats**: WAV (lossless), FLAC (compressed lossless), MP3 (lossy)
- **Bit Depths**: 16-bit (CD), 24-bit (archival), 32-bit float (DAW import)
- **Sample Rates**: 44.1 kHz (music standard), 48 kHz (video), keep original

---

## Module 08: UI (Streamlit)

### Purpose
User interface for upload, processing, comparison, and export.

### Page Structure
```python
import streamlit as st
from audio_engine import AudioPipeline

def main():
    st.title("🎵 Resonate: Live Music Reconstruction")
    st.caption("Transform phone recordings into studio quality")
    
    # Step 1: Upload
    uploaded_file = st.file_uploader(
        "Upload phone recording",
        type=["mp3", "m4a", "wav", "flac"]
    )
    
    if uploaded_file:
        # Step 2: Analyze
        with st.spinner("Analyzing audio..."):
            buffer = pipeline.ingest.load(uploaded_file)
            
        st.audio(buffer.to_bytes(), format="audio/wav")
        
        # Show metadata
        col1, col2, col3 = st.columns(3)
        col1.metric("Duration", f"{buffer.metadata.duration:.1f}s")
        col2.metric("SNR", f"{buffer.metadata.estimated_snr:.1f} dB")
        col3.metric("Brightness", f"{buffer.metadata.spectral_centroid:.0f} Hz")
        
        # Step 3: Processing controls
        st.divider()
        st.subheader("Processing Settings")
        
        quality = st.radio("Quality", ["Preview (fast)", "Render (best)"])
        
        with st.expander("Advanced Settings"):
            noise_reduction = st.slider("Noise Reduction", 0.0, 1.0, 0.5)
            eq_intensity = st.slider("EQ Intensity", 0.0, 1.0, 0.7)
            # ... more controls
            
        # Step 4: Process
        if st.button("Process Audio", type="primary"):
            with st.spinner("Processing... (this may take 5-10 minutes)"):
                result = pipeline.process(
                    buffer,
                    quality=quality,
                    settings=EnhancementSettings(
                        noise_reduction_strength=noise_reduction,
                        eq_intensity=eq_intensity
                    )
                )
                
            # Step 5: Results
            st.success("✅ Processing complete!")
            
            st.subheader("Results")
            
            # Before/After comparison
            col1, col2 = st.columns(2)
            with col1:
                st.caption("Before")
                st.audio(buffer.to_bytes())
            with col2:
                st.caption("After")
                st.audio(result.to_bytes())
                
            # Waveform visualization
            st.subheader("Waveform Comparison")
            fig = plot_waveform_comparison(buffer.data, result.data)
            st.pyplot(fig)
            
            # Spectrogram
            st.subheader("Frequency Content")
            fig = plot_spectrogram_comparison(buffer.data, result.data)
            st.pyplot(fig)
            
            # Step 6: Export
            st.divider()
            st.subheader("Export")
            
            export_format = st.selectbox("Format", ["WAV", "FLAC", "MP3"])
            bit_depth = st.selectbox("Bit Depth", [16, 24]) if export_format == "WAV" else 16
            
            # Download button
            st.download_button(
                "Download Processed Audio",
                data=result.to_bytes(format=export_format, bit_depth=bit_depth),
                file_name=f"resonate_{uploaded_file.name}",
                mime=get_mime_type(export_format)
            )
            
            # Per-stem export
            with st.expander("Export Individual Stems"):
                st.download_button("Vocals", stems.vocals.to_bytes())
                st.download_button("Drums", stems.drums.to_bytes())
                st.download_button("Bass", stems.bass.to_bytes())
                st.download_button("Other", stems.other.to_bytes())
```

### Key UI Features
- **Progress Indicators**: Show processing stages (separating, enhancing, mixing)
- **Real-time Preview**: Play audio at any stage
- **Visual Feedback**: Waveforms, spectrograms, before/after
- **Parameter Persistence**: Remember settings between sessions
- **Error Handling**: Clear messages for file errors, processing failures

---

## Module 09: Caching System

### Purpose
Cache expensive operations (separation, enhancement) for instant re-processing with different settings.

### Strategy
```python
class CacheManager:
    """Manage processing cache"""
    
    def __init__(self, cache_dir: Path = Path("cache")):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)
        
    def get_cache_key(self, audio_hash: str, operation: str, params: dict) -> str:
        """Generate unique cache key"""
        # Hash: audio + operation + parameters
        
    def cache_stems(self, key: str, stems: SeparatedStems):
        """Save separated stems (most expensive operation)"""
        
    def cache_enhanced_stems(self, key: str, stems: SeparatedStems):
        """Save enhanced stems (medium expensive)"""
        
    def get_cached_stems(self, key: str) -> Optional[SeparatedStems]:
        """Retrieve cached stems"""
        
    def clear_old_cache(self, max_age_days: int = 7):
        """Clean up old cache files"""
        
    def get_cache_size(self) -> int:
        """Total cache size in MB"""
```

### Cache Strategy
- **Separation**: Always cache (most expensive, ~5 min)
- **Enhancement**: Cache with parameter hash (medium expensive, ~30 sec)
- **Mixing/Mastering**: Don't cache (cheap, <5 sec)
- **Auto-Cleanup**: Delete cache >7 days old

---

## Module 10: Metrics & Quality Analysis

### Purpose
Measure processing quality, detect artifacts, guide enhancement decisions.

### Metrics
```python
class AudioMetrics:
    """Comprehensive audio quality analysis"""
    
    def analyze(self, audio: np.ndarray, sample_rate: int) -> dict:
        """Full audio analysis"""
        return {
            "snr": self.estimate_snr(audio),
            "lufs": self.measure_lufs(audio, sample_rate),
            "dynamic_range": self.dynamic_range(audio),
            "spectral_centroid": self.spectral_centroid(audio, sample_rate),
            "bandwidth": self.estimate_bandwidth(audio, sample_rate),
            "clipping": self.detect_clipping(audio),
            "artifacts": self.detect_artifacts(audio, sample_rate),
            "quality_score": self.overall_quality(audio, sample_rate)
        }
        
    def estimate_snr(self, audio) -> float:
        """Signal-to-noise ratio (dB)"""
        # Frame-based energy variance
        
    def dynamic_range(self, audio) -> float:
        """Dynamic range in dB (RMS_max / RMS_min)"""
        
    def estimate_bandwidth(self, audio, sr) -> Tuple[float, float]:
        """Estimate frequency range (low_hz, high_hz)"""
        # Spectral rolloff detection
        
    def detect_artifacts(self, audio, sr) -> dict:
        """Detect processing artifacts"""
        return {
            "metallic": self._detect_metallic(audio, sr),
            "robotic": self._detect_robotic(audio, sr),
            "clipping": self._detect_clipping(audio)
        }
        
    def overall_quality(self, audio, sr) -> float:
        """Composite quality score (0-100)"""
        # Weighted combination of metrics
```

### Usage
- **Pre-Processing**: Assess input quality → guide enhancement strategy
- **Post-Processing**: Validate improvement → detect failures
- **User Feedback**: Show quality scores in UI
- **A/B Testing**: Compare before/after objectively

---

## Integration: Complete Pipeline

```python
class AudioPipeline:
    """Main orchestrator for complete processing"""
    
    def __init__(self):
        self.ingest = AudioIngest()
        self.separator = DemucsWrapper()
        self.enhancer = StemEnhancementPipeline()
        self.restorer = FrequencyRestorer()
        self.dereverb = Dereverberator()
        self.mixer = MixEngine()
        self.master = MasteringEngine()
        self.cache = CacheManager()
        self.metrics = AudioMetrics()
        
    def process(
        self,
        file_path: Path,
        quality: SeparationQuality,
        settings: EnhancementSettings,
        enable_frequency_restoration: bool = True,
        enable_dereverberation: bool = False  # Experimental
    ) -> AudioBuffer:
        """
        Complete processing pipeline
        
        Stages:
        1. Ingest → load and validate
        2. Separate → isolate stems (cached)
        3. Enhance → per-stem processing
        4. Restore → bandwidth extension (optional)
        5. De-reverb → remove venue acoustics (optional)
        6. Mix → recombine stems
        7. Master → final polish and export
        """
        
        # Stage 1: Ingest
        buffer = self.ingest.load(file_path)
        print(f"✓ Loaded: {buffer.metadata.duration:.1f}s, SNR: {buffer.metadata.estimated_snr:.1f} dB")
        
        # Stage 2: Separation
        stems = self.separator.separate(buffer, use_cache=True)
        quality_score = self.separator.assess_separation_quality(stems)
        print(f"✓ Separated into stems (quality: {quality_score:.2f})")
        
        # Stage 3: Enhancement
        enhanced_stems = self.enhancer.process(stems)
        print("✓ Enhanced all stems")
        
        # Stage 4: Frequency Restoration (optional)
        if enable_frequency_restoration:
            enhanced_stems = self.restorer.restore_all_stems(enhanced_stems)
            print("✓ Restored frequency content")
            
        # Stage 5: De-reverberation (experimental)
        if enable_dereverberation:
            enhanced_stems = self.dereverb.process_stems(enhanced_stems)
            print("✓ Removed reverb")
            
        # Stage 6: Mixing
        mixed = self.mixer.auto_mix(enhanced_stems)
        print("✓ Mixed stems")
        
        # Stage 7: Mastering
        mastered = self.master.master(mixed)
        print("✓ Mastered audio")
        
        # Final quality check
        final_metrics = self.metrics.analyze(mastered, buffer.sample_rate)
        print(f"✓ Final SNR: {final_metrics['snr']:.1f} dB (+{final_metrics['snr'] - buffer.metadata.estimated_snr:.1f} dB improvement)")
        
        return AudioBuffer(
            data=mastered,
            sample_rate=buffer.sample_rate,
            metadata=buffer.metadata
        )
```

---

## Repository Structure (Advanced)

```
resonate/
├── README.md                       # Project overview, quick start
├── requirements.txt                # Python dependencies
├── pyproject.toml                  # Modern Python packaging
├── .gitignore                      # Cache/, *.pyc, etc.
│
├── app.py                          # Streamlit entry point
│
├── audio_engine/                   # Core audio processing
│   ├── __init__.py
│   ├── pipeline.py                 # AudioPipeline orchestrator
│   ├── ingest.py                   # Module 01
│   ├── separator.py                # Module 02
│   │
│   ├── enhancers/                  # Module 03
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── vocal.py
│   │   ├── drums.py
│   │   ├── bass.py
│   │   ├── instruments.py
│   │   └── pipeline.py
│   │
│   ├── restoration/                # Module 04 & 05
│   │   ├── __init__.py
│   │   ├── frequency.py            # FrequencyRestorer
│   │   └── dereverberation.py      # Dereverberator
│   │
│   ├── mixing.py                   # Module 06
│   ├── mastering.py                # Module 07
│   ├── metrics.py                  # Module 10
│   ├── cache.py                    # Module 09
│   └── device.py                   # MPS/CPU device management
│
├── ui/                             # Module 08
│   ├── __init__.py
│   ├── app.py                      # Main Streamlit UI
│   ├── components.py               # Reusable UI widgets
│   └── visualizations.py           # Waveforms, spectrograms
│
├── config/                         # Configuration
│   ├── defaults.yaml               # Default processing settings
│   └── presets.yaml                # User presets (conservative/balanced/aggressive)
│
├── tests/                          # Test suite
│   ├── __init__.py
│   ├── test_ingest.py
│   ├── test_separator.py
│   ├── test_enhancers.py
│   ├── test_pipeline.py
│   └── fixtures/                   # Test audio files
│       ├── phone_recording.mp3
│       ├── clean_reference.wav
│       └── synthetic_noisy.wav
│
├── cache/                          # Generated caches (gitignored)
│   ├── stems/
│   └── enhanced/
│
├── examples/                       # Example usage
│   ├── basic_usage.py
│   ├── batch_processing.py
│   └── custom_settings.py
│
└── memory-bank/                    # Project documentation
    ├── projectbrief.md
    ├── productContext.md
    ├── techContext.md
    ├── systemPatterns.md
    ├── activeContext.md
    ├── progress.md
    └── modules/
        ├── 01-ingest.md
        ├── 02-separation.md
        ├── 03-enhancement.md
        └── 04-10-advanced.md       # This file
```

---

## Dependencies (Complete)

```txt
# Core
python>=3.10

# Audio Processing
demucs>=4.0.0              # Source separation
torch>=2.0.0               # PyTorch (with MPS for M1)
torchaudio>=2.0.0
pedalboard>=0.7.0          # DSP effects
noisereduce>=2.0.0         # Noise reduction
librosa>=0.10.0            # Audio analysis
soundfile>=0.12.0          # Audio I/O
pyloudnorm>=0.4.0          # Loudness normalization

# Scientific Computing
numpy>=1.24.0
scipy>=1.10.0

# UI
streamlit>=1.28.0          # Web interface
matplotlib>=3.7.0          # Plotting
plotly>=5.14.0             # Interactive plots

# Utilities
pyyaml>=6.0                # Config files
psutil>=5.9.0              # System monitoring
tqdm>=4.65.0               # Progress bars

# Development
pytest>=7.3.0              # Testing
black>=23.0.0              # Code formatting
flake8>=6.0.0              # Linting
```

---

## Next Steps

1. **Validate approach** with user
2. **Set up development environment**
3. **Implement core pipeline** (modules 01-03, 06-07)
4. **Test with real phone recording**
5. **Iterate based on results**
6. **Add advanced features** (modules 04-05) once core works
