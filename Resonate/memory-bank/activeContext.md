# Active Context - Current Focus

**Last Updated**: January 24, 2026
**Status**: Phase 4 - JASCO Stem Regeneration Implementation

## Current Objective
Implement JASCO-based stem regeneration to reconstruct heavily damaged/cooked bass and drums stems. This is the sole focus until complete.

## Implementation Plan
Full plan saved at: `plans/jasco-stem-regeneration.md`

## Recent Changes (Phase 2 & 3)

### Phase 2: Quality & Polish ✅
- ✅ Created `audio_engine/metrics.py` with comprehensive `QualityMetrics` class
  - SNR estimation using spectral analysis
  - LUFS loudness measurement (pyloudnorm)
  - Spectral centroid calculation
  - Clipping detection
  - Artifact detection (metallic, ringing, clicks, phase)
  - Before/after comparison
- ✅ Optimized `audio_engine/cache.py` with segment caching
  - Added version tracking for cache invalidation
  - Segment caching for 30-second chunks
  - `cache_segment()`, `get_segment()`, `cache_segments()`, `get_segments()`
  - `invalidate_by_version()`, `invalidate_segments()`
- ✅ Enhanced `ui/app.py` with visualizations
  - Interactive waveform plots using Plotly
  - A/B toggle button for original vs processed comparison
  - Quality metrics display (SNR, LUFS, spectral centroid, artifacts)
  - Progress indicator with time estimates
- ✅ Added error handling to `audio_engine/separator.py`
  - Fallback spectral separation when Demucs fails
  - Graceful degradation with band-pass filtering
  - User-friendly error messages

### Phase 3: Advanced Features ✅
- ✅ Created `audio_engine/restoration/frequency.py` with `FrequencyRestorer`
  - Bass extension (40-100 Hz) using high-pass filtering
  - High-frequency extension (8-20 kHz) using shelving EQ
  - Intensity-based blending
  - Clamp parameters to prevent instability
- ✅ Created `audio_engine/restoration/dereverberation.py` with `Dereverberator`
  - Early reflection suppression (adaptive filtering)
  - Late reverb suppression (spectral subtraction)
  - Dry/wet mix control
  - Fallback smoothing for robustness
- ✅ Integrated restoration into `audio_engine/pipeline.py`
  - Added restoration config to `PipelineConfig`
  - Added restoration stage between mixing and mastering
  - Frequency restoration → Dereverberation → Mixing → Mastering
- ✅ Updated `ui/app.py` with restoration controls
  - Toggle for frequency restoration
  - Intensity slider for frequency restoration
  - Toggle for dereverberation
  - Intensity slider for dereverberation

## Phase 4: JASCO Stem Regeneration (CURRENT FOCUS)

### Problem Statement
Bass and drums stems are "cooked" (heavily distorted/clipped) in phone recordings. Current DSP enhancement cannot fix fundamentally damaged audio. Need AI-powered regeneration.

### Solution
Use JASCO (Joint Audio and Symbolic Conditioning) from AudioCraft to:
1. Detect damaged stems automatically
2. Extract musical profile (chords, tempo, key, melody, drums)
3. Generate plausible replacement audio conditioned on profile
4. Blend regenerated audio with original (preserve good portions)

### Implementation Phases

**Phase A: Foundation**
- [ ] Create profiling module structure
- [ ] Implement stem quality detector

**Phase B: Musical Profile Extraction**
- [ ] Chord extraction (Chordino/librosa)
- [ ] Tempo/key detection
- [ ] Melody contour extraction
- [ ] Drum pattern extraction
- [ ] MusicalProfile dataclass

**Phase C: JASCO Integration**
- [ ] JASCO model wrapper
- [ ] MPS/CPU fallback for M1 Max

**Phase D: Selective Regeneration**
- [ ] Stem regenerator logic
- [ ] Crossfade blender
- [ ] Time-region damage detection

**Phase E: UI Integration**
- [ ] Per-stem quality indicators
- [ ] Regenerate buttons
- [ ] Style description input
- [ ] Blend slider
- [ ] A/B comparison

**Phase F: Bug Fixes**
- [ ] Fix MBD regenerate() unpacking error
- [ ] Update requirements.txt

### Key Files to Create
```
audio_engine/
├── profiling/
│   ├── __init__.py
│   ├── quality_detector.py
│   ├── chord_extractor.py
│   ├── tempo_key_analyzer.py
│   ├── melody_extractor.py
│   └── drum_pattern_extractor.py
├── generation/
│   ├── __init__.py
│   ├── jasco_generator.py
│   ├── stem_regenerator.py
│   └── blender.py
```

### Known Issues to Fix
1. **MBD regenerate() failing**: `not enough values to unpack (expected 3, got 1)`
2. **FLVAC typo**: Fixed in mastering.py (was FLVAC, now FLAC)

## Implementation Learnings (from Phase 2-3)
1. **Import dependencies**: Always import librosa at module level where used
2. **Filter parameter clamping**: scipy filters require 0 < w0 < 1, apply gains externally
3. **Cache directory structure**: Create stems_dir, enhanced_dir, metadata_dir on init
4. **Fallback strategies**: Always have simple fallback when ML models fail
5. **Audio normalization**: Prevent clipping by normalizing output

## Recommended Model for Implementation
For Code mode implementation, use **Claude Sonnet 4** or **GPT-4.1** - good balance of capability and cost for Python/audio code.
