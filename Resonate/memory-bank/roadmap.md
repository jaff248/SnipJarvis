# Implementation Roadmap - Resonate

## Quick Reference: What We're Building

**Mission**: Transform phone-captured live music recordings into studio-quality audio

**Core Technology**: Demucs (source separation) + Per-stem DSP enhancement + Advanced restoration

**Hardware**: Optimized for Apple M1 Max (64GB RAM) with MPS acceleration

---

## Memory Bank Status: ✅ COMPLETE

### Documentation Created

#### Core Planning Files
- ✅ **projectbrief.md** - Mission, success criteria, technical challenges
- ✅ **productContext.md** - User journey, competitive analysis, philosophy
- ✅ **techContext.md** - Technology stack, dependencies, hardware config
- ✅ **systemPatterns.md** - Architecture, design patterns, repository structure
- ✅ **activeContext.md** - Current decisions, AudioCraft evaluation, open questions
- ✅ **progress.md** - Comprehensive progress tracker with phases

#### Module Implementation Guides
- ✅ **modules/01-ingest.md** - Audio loading, validation, preprocessing (15+ pages)
- ✅ **modules/02-separation.md** - Demucs integration, caching, MPS optimization (20+ pages)
- ✅ **modules/03-enhancement.md** - Per-stem enhancement (vocals, drums, bass, instruments) (25+ pages)
- ✅ **modules/04-10-advanced.md** - Frequency restoration, de-reverb, mixing, mastering, UI, caching, metrics (15+ pages)

**Total Documentation**: 75+ pages of detailed implementation guides

---

## AudioCraft Evaluation: ANSWERED ✅

### Question: Is AudioCraft useful for live music reconstruction?

**TL;DR**: YES - Three components are useful:
1. **MultiBandDiffusion** ✅ - Quality polish (v1.5) - reduces artifacts without changing content
2. **JASCO** ✅ - Creative reconstruction (v2) - when restoration fails
3. **MusicGen/AudioGen** ❌ - NEVER (would hallucinate music)

### Detailed Analysis

#### ✅ USEFUL - MultiBandDiffusion (MBD)
**The Key Finding**: MBD is NOT a generative model - it's a high-fidelity decoder!

```python
# The magic method:
mbd = MultiBandDiffusion.get_mbd_24khz(bw=6.0)
polished = mbd.regenerate(restored_audio, sample_rate=24000)
```

**What happens inside `regenerate()`**:
1. Audio → EnCodec encoder → Discrete tokens (compression)
2. Tokens → Continuous latent embeddings
3. Embeddings → 4 parallel diffusion models (one per frequency band)
4. Output → Summed → High-fidelity waveform

**Why it's perfect for us**:
- Reduces artifacts (harsh/metallic sounds from processing)
- Improves perceptual quality without changing content
- EnCodec trained on full bandwidth (helps with frequencies)
- Optional polish step - doesn't change the audio, just cleans it up

#### ✅ USEFUL - JASCO (v2)
**Creative Reconstruction Mode** - for heavily degraded recordings

- Conditions: text description + extracted chords + drums stem + melody
- Generates studio-quality 10s segments
- Clearly labeled as "AI-Generated"
- Opt-in feature, not default

#### ❌ NOT Useful (Would Break Project Goals):
1. **MusicGen / AudioGen** (text-to-music generation)
   - **Problem**: These are GENERATIVE models - they hallucinate musical content
   - **Violates**: "Reveal, don't fabricate" principle
   - **Decision**: NEVER use for this project

2. **AudioSeal** (watermarking)
   - Not relevant to restoration

### Component Categorization

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       AUDIOCRAFT FOR RESONATE                                │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ RESTORATION (Authentic)                                                 │ │
│  │                                                                         │ │
│  │   v1.0: Demucs + DSP → "Reveal what was played"                        │ │
│  │   v1.5: + MBD polish → "Enhance what was revealed"                     │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ GENERATION (AI-Created)                                                 │ │
│  │                                                                         │ │
│  │   v2.0: JASCO → "Recreate what we think was played"                    │ │
│  │   Opt-in, clearly labeled                                               │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ ❌ NEVER USE                                                            │ │
│  │                                                                         │ │
│  │   MusicGen, AudioGen, MAGNeT → Would hallucinate music                 │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Phases

### Phase 1: Core Pipeline (MVP) - ✅ Complete
**Goal**: Basic working pipeline that improves phone recordings

**Modules to Implement**:
1. ✅ `audio_engine/device.py` - MPS device setup (1 hour)
2. ✅ `audio_engine/utils.py` - Helper functions (1 hour)
3. ✅ `audio_engine/ingest.py` - Audio loading (1 day)
4. ✅ `audio_engine/separator.py` - Demucs integration (2 days)
5. ✅ `audio_engine/cache.py` - Caching system (1 day)
6. ✅ `audio_engine/enhancers/` - All enhancers (3-4 days)
   - base.py, vocal.py, drums.py, bass.py, instruments.py, pipeline.py
7. ✅ `audio_engine/mixing.py` - Stem mixing (1 day)
8. ✅ `audio_engine/mastering.py` - Final polish (1 day)
9. ✅ `audio_engine/pipeline.py` - Orchestrator (1 day)
10. ✅ `ui/app.py` - Streamlit interface (2-3 days)

**Success Criteria**:
- Pipeline implemented and tests pass (7/7)
- Ready for Phase 2–3 features

**Testing**:
- Use friend's actual phone recording
- Compare before/after
- Iterate on enhancement parameters

### Phase 2: Quality & Polish - 1-2 Weeks
**Goal**: Measure results, add validation, improve quality

**Modules to Implement**:
1. ✅ `audio_engine/metrics.py` - Quality analysis (2 days)
2. ✅ `tests/` - Comprehensive test suite (3 days)
3. ✅ Optimize performance - MPS memory, caching (1 day)
4. ✅ UI improvements - Visualizations, comparison (2 days)
5. ✅ Error handling - Graceful degradation (1 day)

**Success Criteria**:
- Quantifiable SNR improvement (≥20 dB target)
- No artifacts introduced
- Processing time acceptable (<5 min preview, <30 min render)

### Phase 2.5: MBD Neural Polish (v1.5) - 3-4 Days
**Goal**: Add optional MultiBandDiffusion enhancement for artifact reduction

**Modules to Implement**:
1. ✅ `audio_engine/polish/mbd_enhancer.py` - MBD integration (2 days)
   ```python
   from audiocraft.models import MultiBandDiffusion
   mbd = MultiBandDiffusion.get_mbd_24khz(bw=6.0)
   polished = mbd.regenerate(audio, sample_rate=24000)
   ```
2. ✅ UI toggle for "Neural Polishing" option (1 day)
3. ✅ A/B comparison: DSP only vs. DSP + MBD (1 day)

**What MBD Does**:
- Reduces artifacts from compression/processing
- Improves perceptual quality (smoother transients)
- Does NOT change content - just cleans it up
- Works via: audio → EnCodec compress → diffusion decode → cleaner audio

**When to Use**:
- Enable for final render (adds ~30s processing)
- Skip for preview (saves time)
- User toggleable: "Enable neural polishing"

**Success Criteria**:
- Audibly cleaner output (blind listening test)
- No content changes (phase correlation >0.95)
- Processing time acceptable (~30s for 3 min track)

### Phase 3: Advanced Features (v1.1) - 2-3 Weeks
**Goal**: Frequency restoration, de-reverberation

**Modules to Implement**:
1. ✅ `audio_engine/restoration/frequency.py` - Bandwidth extension (3-4 days)
   - Research harmonic synthesis methods
   - Implement sub-bass generation (40-80 Hz)
   - Implement high-frequency extension (8kHz+ → 20kHz)
   - Test naturalness (avoid artifacts)

2. ✅ `audio_engine/restoration/dereverberation.py` - Venue acoustics removal (4-5 days)
   - Research WPE or DNN-based methods
   - Implement room impulse response estimation
   - Implement reverb removal
   - Add dry/wet blend control

**Success Criteria**:
- Full 20Hz-20kHz frequency response
- Reverb reduced without artifacts
- User control over intensity

### Phase 4: JASCO Creative Reconstruction (v2) - 2-3 Weeks
**Goal**: Add AI generation mode for heavily degraded recordings

**Modules to Implement**:
1. ✅ `audio_engine/profiling/chord_extractor.py` - Extract chord progressions (2 days)
   - Use Chordino library for chord detection
   - Convert to JASCO format: `[('C', 0.0), ('Am', 2.5), ...]`

2. ✅ `audio_engine/profiling/melody_extractor.py` - Extract melody contour (2 days)
   - Compute salience matrix from vocals
   - Format for JASCO conditioning

3. ✅ `audio_engine/profiling/style_analyzer.py` - Describe audio style (2 days)
   - Auto-detect tempo, key, time signature
   - Genre/mood classification
   - Build text prompts for JASCO

4. ✅ `audio_engine/generation/jasco_generator.py` - JASCO integration (3 days)
   - Load JASCO 1B model
   - Generate from profile (text + chords + drums + melody)
   - Handle 10s chunk stitching for longer tracks

5. ✅ `ui/mode_selector.py` - Dual mode UI (2 days)
   - Restoration vs Generation mode toggle
   - Clear labeling ("AI-Generated" watermark)
   - Side-by-side comparison
   - Blend slider option

**User Workflow**:
```
1. Upload phone recording
2. App analyzes and profiles content
3. User sees suggested profile:
   - "Detected: Pop, 120 BPM, C major"
   - "Vocals: Female, similar to [detected style]"
4. User can edit profile (e.g., "sounds like Tate McRae")
5. User chooses: Restore | Generate | Both
6. App processes and shows comparison
7. User can blend or export either/both
```

**Success Criteria**:
- JASCO generation produces studio-quality 10s segments
- Extracted chords/melody preserve song structure
- Clear UI distinction between restored vs generated
- User can A/B compare and choose

---

## Key Technical Decisions (Locked In)

### 1. Demucs over Spleeter/OpenUnmix
**Reason**: HTDemucs is state-of-the-art, hybrid transformer architecture, best separation quality

### 2. Pedalboard for DSP
**Reason**: Spotify's library, professional quality, fast C++ backend, Pythonic API

### 3. Streamlit for UI (v1)
**Reason**: Fastest prototyping, good enough for single-user, easy to extend

### 4. MPS (Metal) Acceleration
**Reason**: Native M1 Max support, ~10x faster than CPU for Demucs

### 5. Mono Processing
**Reason**: Phone recordings are pseudo-stereo at best, mono is 2x faster, can add stereo width at mastering

### 6. Conservative Enhancement by Default
**Reason**: Better to under-process than over-process, users can increase intensity

### 7. Cache Separated Stems
**Reason**: Separation is 80% of processing time, caching enables rapid experimentation

### 8. MultiBandDiffusion for Polish (v1.5) - NEW!
**Reason**: MBD reduces artifacts without changing content
- Uses: `mbd.regenerate()` → EnCodec compress → diffusion decode
- NOT generative - just a high-fidelity decoder
- Optional step: user can skip to save ~30s processing time
- Handles frequency restoration implicitly (EnCodec trained on full bandwidth)

### 9. JASCO for Generation Mode (v2)
**Reason**: When restoration isn't enough, JASCO can regenerate from extracted profile
- Uses: chords, drums, melody contour, text description
- Clearly labeled as "AI-Generated" not "Restored"
- Opt-in experimental feature, not default

### 10. Two-Mode Architecture
**Reason**: "Reveal" (restoration) vs "Recreate" (generation) serve different needs
- Restoration: authenticity paramount
- Generation: quality paramount
- User chooses based on their priority

---

## Open Questions (To Resolve During Implementation)

### Resolved ✅

#### Frequency Restoration Method
**Options Considered**:
- A) Harmonic synthesis (safe, predictable)
- B) ML-based (EnCodec decoder)
- C) Hybrid approach

**Decision**: MBD handles this! 
- `mbd.regenerate()` implicitly extends bandwidth
- EnCodec trained on full 24kHz bandwidth
- No need for separate harmonic synthesis module
- Simpler architecture, better results

### Still Open

### 1. De-reverberation Algorithm
**Options**:
- A) Wiener filtering with room IR estimation
- B) WPE (Weighted Prediction Error) - blind method ← **Leaning this way**
- C) DNN-based dereverberation (if pre-trained model available)
- D) Manual spectral editing tools (user-driven)

**Decision Point**: Phase 3, research available implementations

### 2. Crowd Noise Handling
**Question**: Will Demucs separate crowd noise from music, or does it need pre-conditioning?

**Hypothesis**: Demucs trained on clean audio may struggle with crowd noise, but worth testing first

**Decision Point**: Phase 1 testing with real phone recording

### 3. Stereo Enhancement
**Question**: Add stereo width at mastering stage, or keep mono?

**Options**:
- A) Keep mono (simple, honest)
- B) Pseudo-stereo (stereo widener on instruments/vocals)
- C) Per-stem panning (drums center, guitars left, keys right)

**Decision Point**: User preference after Phase 1 testing

---

## Success Metrics (from projectbrief.md)

### Must Have (v1)
- ✅ **SNR Improvement**: ≥20 dB improvement over phone recording
- ✅ **Frequency Restoration**: 20Hz-20kHz from ~100Hz-8kHz input
- ✅ **Separation Quality**: Isolated stems with minimal bleed
- ✅ **Processing Time**: <5 min preview, <30 min render (M1 Max)
- ✅ **No Artifacts**: No metallic/robotic sounds introduced

### Should Have (v1.1)
- ✅ **Crowd Noise Reduction**: Remove ambient crowd without affecting music
- ✅ **De-reverberation**: Remove venue acoustics for dry sound
- ✅ **User Controls**: Adjustable enhancement intensity per stage
- ✅ **Stem Export**: Export individual stems for DAW import

### Could Have (v2)
- ⏳ **Reference-Guided Enhancement**: If clean version available, match its characteristics
- ⏳ **Batch Processing**: Process multiple recordings
- ⏳ **Real-Time Preview**: Instant preview of parameter changes

---

## Risk Mitigation

### Risk 1: Demucs Fails on Degraded Audio
**Probability**: Medium  
**Impact**: High (core functionality)  
**Mitigation**:
- Test early with real phone recording
- Implement pre-conditioning if needed
- Have fallback to simpler separation (Spleeter)
- Conservative enhancement even on poor separation

### Risk 2: MPS Memory Issues
**Probability**: Low  
**Impact**: Medium (performance)  
**Mitigation**:
- Implemented memory fraction control (75%)
- Automatic CPU fallback
- Chunked processing for long files

### Risk 3: Frequency Restoration Sounds Unnatural
**Probability**: Medium  
**Impact**: Medium (quality)  
**Mitigation**:
- Conservative defaults (subtle enhancement)
- User control to dial back
- A/B comparison in UI
- Start with harmonic synthesis (safer than ML)

### Risk 4: Expectations Exceed Reality
**Probability**: High  
**Impact**: High (user satisfaction)  
**Mitigation**:
- Clear documentation of limitations
- Show realistic demo in README
- "Reveal, don't fabricate" messaging
- Explicit: "This is restoration, not magic"

---

## Development Environment Setup

### Prerequisites
```bash
# Python 3.10+
python --version  # Should be 3.10 or higher

# Check MPS availability
python -c "import torch; print(torch.backends.mps.is_available())"  # Should print True
```

### Installation
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download Demucs models (automatic on first run)
python -c "from demucs.api import Separator; Separator(model='htdemucs')"

# Verify installation
pytest tests/  # Should pass all tests
```

### Running
```bash
# Start Streamlit app
streamlit run app.py

# Or use command-line script
python examples/basic_usage.py path/to/phone_recording.mp3
```

---

## Next Immediate Steps

### Right Now (Post‑Phase 1)
1. ✅ Phase 1 pipeline + tests complete
2. ⏳ Implement Phase 2–3 features (restoration + dereverb + metrics)
3. ⏳ Validate against real phone recording

### After Approval (Implementation Begins)
1. Set up repository structure
2. Create `requirements.txt` with all dependencies
3. Implement `audio_engine/device.py` (MPS setup)
4. Implement `audio_engine/ingest.py` (foundation)
5. Implement `audio_engine/separator.py` (Demucs)
6. **Test separation** with real phone recording
7. Continue based on results...

---

## File Summary

### What We Have Now
```
memory-bank/
├── projectbrief.md              # ✅ COMPLETE - Mission & goals
├── productContext.md            # ✅ COMPLETE - User journey
├── techContext.md               # ✅ COMPLETE - Tech stack
├── systemPatterns.md            # ✅ COMPLETE - Architecture
├── activeContext.md             # ✅ COMPLETE - Current decisions
├── progress.md                  # ✅ COMPLETE - Progress tracker
├── modules/
│   ├── 01-ingest.md             # ✅ COMPLETE - 15+ pages
│   ├── 02-separation.md         # ✅ COMPLETE - 20+ pages
│   ├── 03-enhancement.md        # ✅ COMPLETE - 25+ pages
│   └── 04-10-advanced.md        # ✅ COMPLETE - 15+ pages
└── roadmap.md                   # ✅ THIS FILE - Implementation guide
```

### What We'll Create Next (After Approval)
```
resonate/
├── README.md                    # Project overview
├── requirements.txt             # Dependencies
├── app.py                       # Streamlit entry
├── audio_engine/               
│   ├── __init__.py
│   ├── device.py               # MPS/CPU handling
│   ├── utils.py                # Helper functions
│   ├── ingest.py               # Audio loading
│   ├── separator.py            # Demucs wrapper
│   ├── cache.py                # Caching system
│   ├── pipeline.py             # Main orchestrator
│   ├── mixing.py               # Stem mixing
│   ├── mastering.py            # Final polish
│   ├── metrics.py              # Quality analysis
│   ├── enhancers/              
│   │   ├── __init__.py
│   │   ├── base.py             # Base enhancer class
│   │   ├── vocal.py            # Vocal enhancement
│   │   ├── drums.py            # Drums enhancement
│   │   ├── bass.py             # Bass enhancement
│   │   ├── instruments.py      # Other instruments
│   │   └── pipeline.py         # Enhancement orchestrator
│   ├── restoration/            # v1.1
│   │   ├── __init__.py
│   │   ├── frequency.py        # Bandwidth extension
│   │   └── dereverberation.py  # Reverb removal
│   ├── polish/                 # v1.5 (MBD)
│   │   ├── __init__.py
│   │   └── mbd_enhancer.py     # MultiBandDiffusion polish
│   ├── profiling/              # v2 (JASCO)
│   │   ├── __init__.py
│   │   ├── chord_extractor.py  # Chord detection
│   │   ├── melody_extractor.py # Melody salience
│   │   └── style_analyzer.py   # Tempo/key/style
│   └── generation/             # v2 (JASCO)
│       ├── __init__.py
│       └── jasco_generator.py  # JASCO integration
├── ui/                         
│   ├── __init__.py
│   ├── components.py           # Reusable UI pieces
│   ├── visualizations.py       # Waveforms, spectrograms
│   └── mode_selector.py        # v2: Restore vs Generate
├── tests/                      # Test suite (10+ files)
├── examples/                   # Usage examples (4 files)
├── assets/                     # JASCO chord mapping
│   └── chord_to_index_mapping.pkl
└── cache/                      # Auto-generated
```

---

## Questions for User Before Implementation

1. **Do you have the actual phone recording available for testing?**
   - This will be critical for validating our approach early

2. **Do you have the original studio recording for A/B comparison?**
   - Would help measure success objectively
   - Could enable reference-guided enhancement (v2 feature)

3. **What's the priority: quality or speed?**
   - Affects: model choice, parameter defaults, optimization focus

4. **Any specific artifacts in the phone recording we should know about?**
   - Clipping, wind noise, handling noise, crowd chatter, etc.
   - Helps us prepare appropriate pre-conditioning

5. **Preferred output format?**
   - WAV (lossless), FLAC (compressed lossless), MP3 (lossy)
   - We'll support all, but what's your target?

---

## Conclusion

**Memory bank is complete and comprehensive**. We have:
- ✅ Clear mission and success criteria
- ✅ Detailed technical architecture
- ✅ 75+ pages of implementation guides for every module
- ✅ Locked-in technology decisions with rationale
- ✅ Risk mitigation strategies
- ✅ Clear implementation phases
- ✅ AudioCraft evaluation (conclusion: skip generative models)

**Ready to begin implementation immediately upon approval.**

Philosophy: **"Reveal, don't fabricate"** - We reconstruct what was played, we don't hallucinate what wasn't.

Goal: **Transform phone-captured live music into studio-quality audio** using proven ML (Demucs) + professional DSP techniques.
