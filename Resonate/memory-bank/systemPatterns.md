# Resonate: System Patterns

## Architecture Overview: Live Music Reconstruction Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                      STREAMLIT WEB INTERFACE                         │
│  ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐    │
│  │ Upload  │→ │ Analyze  │→ │ Process  │→ │ Compare & Export │    │
│  │ Phone   │  │ Quality  │  │ Settings │  │ Before/After     │    │
│  └─────────┘  └──────────┘  └──────────┘  └──────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│              LIVE MUSIC RECONSTRUCTION PIPELINE                      │
│                                                                      │
│  ┌──────────────┐   ┌─────────────────┐   ┌─────────────────┐     │
│  │   INGEST     │ → │    SEPARATE     │ → │    ENHANCE      │     │
│  │ Load/Analyze │   │ Demucs HTDemucs │   │  Per-Stem DSP   │     │
│  └──────────────┘   └─────────────────┘   └─────────────────┘     │
│       │                    │                      │                 │
│       ▼                    ▼                      ▼                 │
│  [Validate]         [Vocals Stem]          [Noise Reduce]          │
│  [Estimate SNR]     [Drums Stem]           [EQ Clarity]            │
│  [Detect Phone]     [Bass Stem]            [Compression]           │
│  [Mono Convert]     [Other Stem]           [Harmonic Excite]       │
│                     [Cache Stems]                                   │
│                                                                      │
│  ┌─────────────────┐   ┌─────────────┐   ┌──────────────────┐     │
│  │    RESTORE      │ → │   DE-REVERB │ → │   MIX & MASTER   │     │
│  │ Freq Extension  │   │ Venue Remove│   │  Final Polish    │     │
│  └─────────────────┘   └─────────────┘   └──────────────────┘     │
│       │                    │                      │                 │
│       ▼                    ▼                      ▼                 │
│  [Sub-bass Gen]     [Estimate IR]          [Gain Stage]            │
│  [High-freq Ext]    [WPE/Wiener]           [LUFS Normalize]        │
│  [20Hz-20kHz]       [Dry/Wet Blend]        [True Peak Limit]       │
│  [Natural Blend]    [Optional]             [Dither & Export]       │
│                                                                      │
│                     [Quality Metrics: SNR, LUFS, Artifacts]         │
└─────────────────────────────────────────────────────────────────────┘
```

### Critical Path (Minimum Viable Pipeline)
**Phase 1 (v1):** Ingest → Separate → Enhance → Mix → Master
**Phase 2 (v1.1):** Add Frequency Restoration
**Phase 3 (v2):** Add De-reverberation (experimental)

## Processing Pipeline

### Stage 1: Ingest
```python
class AudioIngest:
    def process(self, file_path: str) -> AudioBuffer:
        # 1. Load with soundfile (preserve native format)
        # 2. Validate: sample_rate, channels, duration
        # 3. Normalize to float32 range [-1, 1]
        # 4. Store metadata for reconstruction
```

### Stage 2: Separation (Demucs)
- **Model**: `htdemucs_ft` for quality, `htdemucs` for speed
- **Output**: 4 stems (vocals, drums, bass, other)
- **Key Pattern**: Use official API, not direct model access
```python
from demucs.api import Separator

separator = Separator(model="htdemucs_ft", segment=10, device="mps")
_, stems = separator.separate_audio_file(path)
```

### Stage 3: Per-Stem Enhancement
Each stem gets tailored processing:

| Stem | Primary Enhancement | Secondary |
|------|---------------------|-----------|
| Vocals | Noise reduction, Presence EQ | Compression |
| Drums | Transient shaping | None (preserve dynamics) |
| Bass | Low-end clarity | Subtle compression |
| Other | Harmonic exciter, EQ | De-mudding |

### Stage 4: Mix & Master
```python
class MixMaster:
    def process(self, stems: Dict[str, AudioBuffer]) -> AudioBuffer:
        # 1. Recombine stems with gain staging
        # 2. Apply gentle master EQ (if reference provided)
        # 3. Loudness normalize to -14 LUFS
        # 4. True peak limit to -1 dB
        # 5. Dither to output bit depth
```

## Key Design Patterns

### 1. Preview/Render Mode
```python
class ProcessingMode(Enum):
    PREVIEW = "preview"   # 30 sec sample, fast settings
    RENDER = "render"     # Full audio, high quality

# Preview: segment=5, shifts=0
# Render: segment=10, shifts=5
```

### 2. Safe Processing with Rollback
```python
def safe_enhance(audio, enhancer, threshold=0.1):
    """Apply enhancement, rollback if artifacts detected"""
    enhanced = enhancer.process(audio)
    artifact_score = detect_artifacts(enhanced, audio)
    
    if artifact_score > threshold:
        return audio  # Return original if too many artifacts
    return enhanced
```

### 3. Metric-Driven Decisions
```python
class AudioMetrics:
    def should_denoise(self, audio) -> bool:
        snr = estimate_snr(audio)
        return snr < 20  # Only denoise if SNR is poor
    
    def should_enhance_highs(self, audio) -> bool:
        spectral_centroid = compute_centroid(audio)
        return spectral_centroid < 2000  # Muffled audio
```

### 4. Caching Strategy
```
cache/
├── {hash}_stems/           # Separated stems (expensive)
│   ├── vocals.npy
│   ├── drums.npy
│   ├── bass.npy
│   └── other.npy
├── {hash}_enhanced/        # Enhanced stems  
└── {hash}_metrics.json     # Analysis results
```

## Error Handling Patterns

### Graceful Degradation
```python
try:
    device = torch.device('mps')
    model.to(device)
except Exception:
    warnings.warn("MPS unavailable, using CPU")
    device = torch.device('cpu')
    model.to(device)
```

### Processing Timeouts
```python
from concurrent.futures import ThreadPoolExecutor, TimeoutError

def process_with_timeout(func, audio, timeout=300):
    with ThreadPoolExecutor() as executor:
        future = executor.submit(func, audio)
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            raise ProcessingTimeout(f"Processing exceeded {timeout}s limit")
```

## File Organization (Complete Repository Structure)

```
resonate/
├── README.md                       # Project overview, installation
├── requirements.txt                # Python dependencies
├── pyproject.toml                  # Modern Python packaging
├── .gitignore                      # Ignore cache/, *.pyc, models/
├── LICENSE                         # Open source license (MIT)
│
├── app.py                          # Streamlit entry point (Module 08)
│
├── audio_engine/                   # Core audio processing library
│   ├── __init__.py
│   ├── pipeline.py                 # AudioPipeline orchestrator (integrates all modules)
│   │
│   ├── ingest.py                   # Module 01: Audio loading & validation
│   ├── separator.py                # Module 02: Demucs source separation
│   │
│   ├── enhancers/                  # Module 03: Per-stem enhancement
│   │   ├── __init__.py
│   │   ├── base.py                 # StemEnhancer base class
│   │   ├── vocal.py                # VocalEnhancer
│   │   ├── drums.py                # DrumEnhancer
│   │   ├── bass.py                 # BassEnhancer
│   │   ├── instruments.py          # InstrumentEnhancer (other)
│   │   └── pipeline.py             # StemEnhancementPipeline
│   │
│   ├── restoration/                # Modules 04 & 05: Advanced restoration
│   │   ├── __init__.py
│   │   ├── frequency.py            # FrequencyRestorer (bandwidth extension)
│   │   └── dereverberation.py      # Dereverberator (venue acoustics removal)
│   │
│   ├── mixing.py                   # Module 06: Stem mixing & gain staging
│   ├── mastering.py                # Module 07: Loudness, limiting, export
│   │
│   ├── metrics.py                  # Module 10: Quality analysis (SNR, artifacts)
│   ├── cache.py                    # Module 09: Caching system
│   ├── device.py                   # MPS/CPU device management
│   └── utils.py                    # Helper functions (db_to_gain, etc.)
│
├── ui/                             # Module 08: Streamlit interface
│   ├── __init__.py
│   ├── app.py                      # Main UI layout
│   ├── components.py               # Reusable widgets (sliders, buttons)
│   └── visualizations.py           # Waveforms, spectrograms, comparisons
│
├── config/                         # Configuration files
│   ├── defaults.yaml               # Default processing parameters
│   └── presets.yaml                # User presets (conservative/balanced/aggressive)
│
├── tests/                          # Test suite (pytest)
│   ├── __init__.py
│   ├── conftest.py                 # Shared fixtures
│   ├── test_ingest.py
│   ├── test_separator.py
│   ├── test_enhancers.py
│   ├── test_restoration.py
│   ├── test_mixing.py
│   ├── test_mastering.py
│   ├── test_pipeline.py            # Integration tests
│   └── fixtures/                   # Test audio files
│       ├── phone_recording.mp3     # Real phone capture
│       ├── clean_reference.wav     # Studio version (if available)
│       ├── synthetic_noisy.wav     # Generated test audio
│       └── short_clip.mp3          # 10sec clip for fast tests
│
├── examples/                       # Usage examples
│   ├── basic_usage.py              # Simple script example
│   ├── batch_processing.py         # Process multiple files
│   ├── custom_settings.py          # Advanced parameter tuning
│   └── export_stems.py             # Export stems for DAW
│
├── docs/                           # Documentation (optional)
│   ├── installation.md
│   ├── usage.md
│   ├── api_reference.md
│   └── troubleshooting.md
│
├── cache/                          # Generated caches (gitignored)
│   ├── stems/                      # Separated stems (expensive)
│   │   └── {hash}/
│   │       ├── vocals.npy
│   │       ├── drums.npy
│   │       ├── bass.npy
│   │       ├── other.npy
│   │       └── meta.json
│   └── enhanced/                   # Enhanced stems (medium expensive)
│
├── models/                         # Downloaded ML models (gitignored)
│   └── demucs/                     # Demucs models auto-downloaded here
│
└── memory-bank/                    # Project memory & documentation
    ├── projectbrief.md             # Core mission & success criteria
    ├── productContext.md           # User journey, competitive landscape
    ├── techContext.md              # Technology stack details
    ├── systemPatterns.md           # This file: Architecture & patterns
    ├── activeContext.md            # Current decisions & open questions
    ├── progress.md                 # Progress tracker
    └── modules/                    # Detailed implementation guides
        ├── 01-ingest.md            # Audio loading & validation
        ├── 02-separation.md        # Demucs integration
        ├── 03-enhancement.md       # Per-stem enhancement
        └── 04-10-advanced.md       # Frequency restoration → UI
```

### Module Dependencies (Import Graph)

```
app.py
 └─> ui/app.py
      └─> audio_engine/pipeline.py
           ├─> audio_engine/ingest.py
           ├─> audio_engine/separator.py
           │    └─> audio_engine/device.py
           ├─> audio_engine/enhancers/pipeline.py
           │    ├─> audio_engine/enhancers/vocal.py
           │    ├─> audio_engine/enhancers/drums.py
           │    ├─> audio_engine/enhancers/bass.py
           │    └─> audio_engine/enhancers/instruments.py
           ├─> audio_engine/restoration/frequency.py
           ├─> audio_engine/restoration/dereverberation.py
           ├─> audio_engine/mixing.py
           ├─> audio_engine/mastering.py
           ├─> audio_engine/metrics.py
           └─> audio_engine/cache.py
```

### Implementation Order (Priority)

**Phase 1: Core Pipeline (MVP)**
1. `audio_engine/ingest.py` - Foundation
2. `audio_engine/device.py` - MPS setup
3. `audio_engine/separator.py` - Demucs integration
4. `audio_engine/enhancers/` - All enhancers
5. `audio_engine/mixing.py` - Stem recombination
6. `audio_engine/mastering.py` - Final polish
7. `audio_engine/pipeline.py` - Orchestrator
8. `audio_engine/cache.py` - Performance
9. `ui/app.py` - User interface

**Phase 2: Advanced Features**
10. `audio_engine/metrics.py` - Quality validation
11. `audio_engine/restoration/frequency.py` - Bandwidth extension
12. `audio_engine/restoration/dereverberation.py` - Reverb removal

**Phase 3: Polish**
13. `tests/` - Comprehensive testing
14. `examples/` - Usage demonstrations
15. `docs/` - User documentation
