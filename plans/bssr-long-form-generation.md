# Beat-Synchronous Stem-Sequential Regeneration (BSSR)

## Research-Grade Long-Form AI Music Generation

### Problem Statement

MusicGen/JASCO models have a hard 30-second generation limit. For a typical 3-5 minute song in "Taylor's Version" mode (100% AI regeneration), we need a novel approach to generate coherent, full-length audio without audible discontinuities.

**Current Failure Mode:**
```
Song: 4 minutes (240 seconds)
↓
JASCOGenerator: "duration > 30s, truncating to 30s"  ← FAILS
↓
Output: Only first 30 seconds regenerated
```

### Solution: BSSR Pipeline

BSSR combines five novel techniques:
1. **Bar-Aligned Chunking** - Chunk boundaries at musical bar lines
2. **Autoregressive Continuation** - Each chunk conditioned on previous chunk's tail
3. **Stem-Sequential Dependencies** - Generate drums→bass→other→vocals in order
4. **Optimal Cut-Point Selection** - Spectral distance minimization in overlap regions
5. **Beat-Phase Coherent Stitching** - Zero-crossing aligned crossfades at beat boundaries

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              BSSR PIPELINE                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │ MusicalStructure│───▶│  BarAligned     │───▶│  Stem-Sequential│         │
│  │    Analyzer     │    │    Chunker      │    │   Orchestrator  │         │
│  └─────────────────┘    └─────────────────┘    └────────┬────────┘         │
│         ▲                                               │                   │
│         │                                               ▼                   │
│         │                                    ┌─────────────────┐            │
│         │                                    │  Autoregressive │            │
│         │                                    │    Generator    │            │
│         │                                    └────────┬────────┘            │
│         │                                             │                     │
│  ┌──────┴──────────┐                                  ▼                     │
│  │   Original      │                      ┌─────────────────┐               │
│  │   Audio Input   │                      │ OptimalCut      │               │
│  └─────────────────┘                      │    Finder       │               │
│                                           └────────┬────────┘               │
│                                                    │                        │
│                                                    ▼                        │
│                                         ┌─────────────────┐                 │
│                                         │ BeatAligned     │                 │
│                                         │   Stitcher      │                 │
│                                         └────────┬────────┘                 │
│                                                  │                          │
│                                                  ▼                          │
│                                         ┌─────────────────┐                 │
│                                         │  Final Output   │                 │
│                                         │ (Full-length)   │                 │
│                                         └─────────────────┘                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Specifications

### Component 1: MusicalStructureAnalyzer

**File:** `Resonate/audio_engine/profiling/musical_structure.py`

**Purpose:** Extract complete musical structure including beat grid, bar boundaries, and section markers.

```python
@dataclass
class MusicalStructure:
    # Beat information
    beat_times: np.ndarray        # Timestamps of all beats
    beat_frames: np.ndarray       # Frame indices of beats
    tempo: float                  # BPM
    
    # Bar information
    bar_times: np.ndarray         # Timestamps of bar starts
    bar_frames: np.ndarray        # Frame indices of bar starts
    beats_per_bar: int            # Typically 4 for 4/4
    
    # Section information (optional but valuable)
    section_boundaries: List[float]  # Verse/chorus/bridge boundaries
    section_labels: List[str]        # Labels for each section
    
    # Derived
    total_bars: int
    total_beats: int
    duration: float


class MusicalStructureAnalyzer:
    def __init__(self, hop_length: int = 512):
        self.hop_length = hop_length
    
    def analyze(self, audio: np.ndarray, sample_rate: int) -> MusicalStructure:
        """Extract complete musical structure from audio."""
        # 1. Beat tracking
        # 2. Bar inference
        # 3. Section segmentation (optional, uses MSAF or spectral novelty)
        pass
    
    def get_beat_times(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Extract beat timestamps using librosa."""
        import librosa
        tempo, beats = librosa.beat.beat_track(y=audio, sr=sample_rate, 
                                                hop_length=self.hop_length)
        return librosa.frames_to_time(beats, sr=sample_rate, 
                                       hop_length=self.hop_length)
    
    def get_bar_times(self, beat_times: np.ndarray, 
                      beats_per_bar: int = 4) -> np.ndarray:
        """Convert beat times to bar times."""
        return beat_times[::beats_per_bar]
    
    def find_section_boundaries(self, audio: np.ndarray, 
                                 sample_rate: int) -> List[float]:
        """Detect section boundaries using spectral novelty."""
        # Uses librosa.segment or custom spectral contrast analysis
        pass
```

**Key Methods:**
| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `analyze()` | audio, sr | MusicalStructure | Full structure extraction |
| `get_beat_times()` | audio, sr | np.ndarray | Beat timestamp array |
| `get_bar_times()` | beat_times, bpb | np.ndarray | Bar timestamp array |
| `find_section_boundaries()` | audio, sr | List[float] | Section start times |

---

### Component 2: BarAlignedChunker

**File:** `Resonate/audio_engine/generation/bar_aligned_chunker.py`

**Purpose:** Create chunks with boundaries at musically optimal points (bar lines).

```python
@dataclass
class BarAlignedChunk:
    chunk_index: int
    start_time: float
    end_time: float
    start_bar: int
    end_bar: int
    overlap_start: float  # Overlap with previous chunk
    overlap_end: float    # Overlap with next chunk
    is_section_boundary: bool  # True if chunk starts at section boundary


@dataclass  
class ChunkingConfig:
    target_duration: float = 28.0       # Target ~28s (under 30s limit)
    max_duration: float = 30.0          # Hard max
    min_duration: float = 20.0          # Min duration
    overlap_bars: int = 4               # 4 bars overlap
    prefer_section_boundaries: bool = True


class BarAlignedChunker:
    def __init__(self, config: ChunkingConfig = None):
        self.config = config or ChunkingConfig()
    
    def create_chunks(self, structure: MusicalStructure) -> List[BarAlignedChunk]:
        """Create bar-aligned chunks for the song."""
        pass
    
    def find_optimal_boundaries(self, bar_times: np.ndarray,
                                 section_boundaries: List[float],
                                 target_duration: float) -> List[int]:
        """Find bar indices that create ~target_duration chunks."""
        # Algorithm:
        # 1. Start at bar 0
        # 2. Find bar closest to current_position + target_duration
        # 3. If a section boundary is within ±2 bars, prefer it
        # 4. Add overlap region
        # 5. Repeat until end
        pass
```

**Chunking Algorithm Pseudocode:**
```
function create_chunks(bar_times, section_boundaries):
    chunks = []
    current_bar = 0
    chunk_idx = 0
    
    while current_bar < total_bars:
        # Find target end bar
        target_time = bar_times[current_bar] + target_duration
        target_bar = find_nearest_bar(bar_times, target_time)
        
        # Check for nearby section boundary
        if prefer_section_boundaries:
            nearby_section = find_section_near_bar(target_bar, ±2 bars)
            if nearby_section:
                target_bar = nearby_section
        
        # Create chunk with overlap
        chunk = BarAlignedChunk(
            start_bar=current_bar,
            end_bar=target_bar + overlap_bars,  # Overlap into next chunk
            overlap_start=bar_times[current_bar] if chunk_idx > 0 else 0,
            ...
        )
        chunks.append(chunk)
        
        # Move to next chunk (start at target_bar, not target_bar + overlap)
        current_bar = target_bar
        chunk_idx += 1
    
    return chunks
```

---

### Component 3: AutoregressiveGenerator

**File:** `Resonate/audio_engine/generation/autoregressive_generator.py`

**Purpose:** Wrap JASCOGenerator to support continuation-based generation.

```python
@dataclass
class ContinuationContext:
    previous_audio: np.ndarray      # Last N seconds of previous chunk
    context_duration: float          # How much context to use (default 2s)
    tempo: float
    key: str
    chords: List[Tuple[str, float]]


class AutoregressiveGenerator:
    def __init__(self, base_generator: JASCOGenerator):
        self.generator = base_generator
        self.context_duration = 2.0  # 2 seconds of context
    
    def generate_chunk(self, 
                       chunk: BarAlignedChunk,
                       context: Optional[ContinuationContext],
                       musical_profile: Dict,
                       stem_type: str) -> GenerationResult:
        """Generate a single chunk with optional continuation context."""
        
        if context is not None:
            # Use continuation mode
            return self._generate_with_continuation(
                chunk, context, musical_profile, stem_type
            )
        else:
            # First chunk - no context
            return self._generate_fresh(chunk, musical_profile, stem_type)
    
    def _generate_with_continuation(self, chunk, context, profile, stem_type):
        """Generate audio that continues from previous chunk."""
        # MusicGen supports continuation via audio conditioning
        # We feed the tail of previous chunk as "melody" conditioning
        
        # Build description
        description = self._build_description(profile, stem_type)
        
        # Generate with continuation
        # Note: This may require custom MusicGen integration
        result = self.generator._model.generate_continuation(
            prompt=context.previous_audio,
            prompt_sample_rate=44100,
            descriptions=[description],
            duration=chunk.end_time - chunk.start_time,
        )
        
        return result
    
    def _generate_fresh(self, chunk, profile, stem_type):
        """Generate first chunk without continuation."""
        return self.generator.generate(
            chords_timeline=profile.get('chords'),
            tempo=profile.get('tempo'),
            key=profile.get('key'),
            stem_type=stem_type,
            duration=int(chunk.end_time - chunk.start_time)
        )
```

**MusicGen Continuation API:**
```python
# MusicGen has a generate_continuation method:
model.generate_continuation(
    prompt=audio_tensor,           # Audio to continue from
    prompt_sample_rate=44100,      # Sample rate of prompt
    descriptions=['description'],  # Text description
    duration=10.0,                 # Duration to generate
)
```

---

### Component 4: StemSequentialOrchestrator

**File:** `Resonate/audio_engine/generation/stem_orchestrator.py`

**Purpose:** Manage stem generation order and inter-stem dependencies.

```python
@dataclass
class StemDependency:
    stem_type: str
    depends_on: List[str]  # Stems that must be generated first
    conditioning_weight: float  # How much to condition on dependencies


# Default dependency order
STEM_ORDER = [
    StemDependency('drums', [], 1.0),           # First - sets rhythm
    StemDependency('bass', ['drums'], 0.8),     # Follows drums + chords
    StemDependency('other', ['drums', 'bass'], 0.6),  # Follows all
    StemDependency('vocals', ['drums', 'bass', 'other'], 0.4),  # Last
]


class StemSequentialOrchestrator:
    def __init__(self, 
                 chunker: BarAlignedChunker,
                 generator: AutoregressiveGenerator):
        self.chunker = chunker
        self.generator = generator
        self.stem_order = STEM_ORDER
        
    def regenerate_all_stems(self,
                             original_stems: Dict[str, np.ndarray],
                             musical_profile: Dict,
                             structure: MusicalStructure,
                             callbacks: Optional[Any] = None) -> Dict[str, np.ndarray]:
        """Regenerate all stems in dependency order."""
        
        # Create chunks
        chunks = self.chunker.create_chunks(structure)
        
        regenerated_stems = {}
        
        for stem_dep in self.stem_order:
            stem_type = stem_dep.stem_type
            
            # Skip if not in original stems
            if stem_type not in original_stems:
                continue
            
            # Get conditioning from already-generated stems
            conditioning_audio = self._get_conditioning_audio(
                regenerated_stems, stem_dep.depends_on
            )
            
            # Regenerate this stem
            regenerated = self._regenerate_stem(
                chunks=chunks,
                stem_type=stem_type,
                original_audio=original_stems[stem_type],
                musical_profile=musical_profile,
                conditioning_audio=conditioning_audio,
                callbacks=callbacks,
            )
            
            regenerated_stems[stem_type] = regenerated
        
        return regenerated_stems
    
    def _regenerate_stem(self, chunks, stem_type, original_audio, 
                          musical_profile, conditioning_audio, callbacks):
        """Regenerate a single stem across all chunks."""
        
        generated_chunks = []
        context = None
        
        for i, chunk in enumerate(chunks):
            # Generate this chunk
            result = self.generator.generate_chunk(
                chunk=chunk,
                context=context,
                musical_profile=musical_profile,
                stem_type=stem_type,
            )
            
            generated_chunks.append(result.audio)
            
            # Update context for next chunk
            context = ContinuationContext(
                previous_audio=result.audio[-2*44100:],  # Last 2 seconds
                context_duration=2.0,
                tempo=musical_profile.get('tempo'),
                key=musical_profile.get('key'),
                chords=musical_profile.get('chords'),
            )
            
            if callbacks:
                progress = (i + 1) / len(chunks) * 100
                callbacks.report_progress(progress, f"Chunk {i+1}/{len(chunks)}")
        
        # Stitch chunks together
        return self._stitch_chunks(generated_chunks, chunks)
```

---

### Component 5: OptimalCutFinder

**File:** `Resonate/audio_engine/generation/optimal_cut_finder.py`

**Purpose:** Find the best frame to cut between overlapping chunks.

```python
@dataclass
class CutPoint:
    time: float            # Cut time in seconds
    frame: int             # Cut frame index
    spectral_distance: float  # How similar the chunks are at this point
    is_beat_aligned: bool  # Whether cut is on a beat
    confidence: float      # Overall confidence in this cut point


class OptimalCutFinder:
    def __init__(self, sample_rate: int = 44100, n_fft: int = 2048):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = n_fft // 4
    
    def find_optimal_cut(self,
                         chunk_a: np.ndarray,
                         chunk_b: np.ndarray,
                         overlap_duration: float,
                         beat_times: np.ndarray) -> CutPoint:
        """Find the optimal cut point in the overlap region."""
        
        # Extract overlap regions
        overlap_samples = int(overlap_duration * self.sample_rate)
        a_overlap = chunk_a[-overlap_samples:]
        b_overlap = chunk_b[:overlap_samples]
        
        # Compute spectral representations
        stft_a = np.abs(librosa.stft(a_overlap, n_fft=self.n_fft, 
                                      hop_length=self.hop_length))
        stft_b = np.abs(librosa.stft(b_overlap, n_fft=self.n_fft,
                                      hop_length=self.hop_length))
        
        # Compute frame-by-frame spectral distance
        distances = self._compute_spectral_distances(stft_a, stft_b)
        
        # Find minimum distance frame
        min_frame = np.argmin(distances)
        min_time = librosa.frames_to_time(min_frame, sr=self.sample_rate,
                                           hop_length=self.hop_length)
        
        # Snap to nearest beat
        cut_time, is_beat = self._snap_to_beat(min_time, beat_times)
        cut_frame = librosa.time_to_frames(cut_time, sr=self.sample_rate,
                                            hop_length=self.hop_length)
        
        return CutPoint(
            time=cut_time,
            frame=cut_frame,
            spectral_distance=distances[min_frame],
            is_beat_aligned=is_beat,
            confidence=1.0 - distances[min_frame],
        )
    
    def _compute_spectral_distances(self, stft_a, stft_b) -> np.ndarray:
        """Compute spectral distance between corresponding frames."""
        # Cosine distance between magnitude spectra
        distances = []
        for i in range(min(stft_a.shape[1], stft_b.shape[1])):
            a_norm = stft_a[:, i] / (np.linalg.norm(stft_a[:, i]) + 1e-10)
            b_norm = stft_b[:, i] / (np.linalg.norm(stft_b[:, i]) + 1e-10)
            distance = 1.0 - np.dot(a_norm, b_norm)
            distances.append(distance)
        return np.array(distances)
    
    def _snap_to_beat(self, time: float, beat_times: np.ndarray, 
                       tolerance: float = 0.1) -> Tuple[float, bool]:
        """Snap time to nearest beat if within tolerance."""
        if len(beat_times) == 0:
            return time, False
        
        nearest_idx = np.argmin(np.abs(beat_times - time))
        nearest_beat = beat_times[nearest_idx]
        
        if abs(nearest_beat - time) <= tolerance:
            return nearest_beat, True
        return time, False
```

**Spectral Distance Visualization:**
```
Chunk A overlap:  ────────────────────────▶
                        ╔════════════════╗
Spectral Distance:      ║▓▓▓▓░░▓▓▓░░░▓▓▓║  (lower = more similar)
                        ╚════════════════╝
Chunk B overlap:  ────────────────────────▶
                              ▲
                         Optimal cut
                    (minimum distance)
```

---

### Component 6: BeatAlignedStitcher

**File:** `Resonate/audio_engine/generation/beat_aligned_stitcher.py`

**Purpose:** Stitch chunks together with phase-coherent crossfades at beat boundaries.

```python
@dataclass
class StitchConfig:
    crossfade_bars: int = 2        # Crossfade duration in bars
    use_equal_power: bool = True   # Equal-power crossfade
    phase_correction: bool = True  # Apply phase correction


class BeatAlignedStitcher:
    def __init__(self, sample_rate: int = 44100, config: StitchConfig = None):
        self.sample_rate = sample_rate
        self.config = config or StitchConfig()
    
    def stitch_chunks(self,
                      chunks: List[np.ndarray],
                      cut_points: List[CutPoint],
                      structure: MusicalStructure) -> np.ndarray:
        """Stitch all chunks together using optimal cut points."""
        
        if len(chunks) == 1:
            return chunks[0]
        
        # Start with first chunk (up to first cut point)
        cut_sample = int(cut_points[0].time * self.sample_rate)
        result = chunks[0][:cut_sample].copy()
        
        for i in range(1, len(chunks)):
            cut_point = cut_points[i-1]
            next_cut = cut_points[i] if i < len(cut_points) else None
            
            # Get chunk segment
            start_sample = int(cut_point.time * self.sample_rate)
            if next_cut:
                end_sample = int(next_cut.time * self.sample_rate)
                segment = chunks[i][start_sample:end_sample]
            else:
                segment = chunks[i][start_sample:]
            
            # Apply crossfade at boundary
            crossfade_duration = self._get_crossfade_duration(structure)
            crossfade_samples = int(crossfade_duration * self.sample_rate)
            
            # Phase correction
            if self.config.phase_correction:
                segment = self._phase_correct(result[-crossfade_samples:], 
                                               segment[:crossfade_samples],
                                               segment)
            
            # Apply crossfade
            result = self._apply_crossfade(result, segment, crossfade_samples)
        
        return result
    
    def _apply_crossfade(self, a: np.ndarray, b: np.ndarray, 
                          crossfade_samples: int) -> np.ndarray:
        """Apply equal-power crossfade between a and b."""
        
        # Create crossfade curves
        fade_out = np.cos(np.linspace(0, np.pi/2, crossfade_samples)) ** 2
        fade_in = np.sin(np.linspace(0, np.pi/2, crossfade_samples)) ** 2
        
        # Apply crossfade
        a_end = a[-crossfade_samples:]
        b_start = b[:crossfade_samples]
        
        crossfade = a_end * fade_out + b_start * fade_in
        
        # Concatenate
        return np.concatenate([a[:-crossfade_samples], crossfade, b[crossfade_samples:]])
    
    def _phase_correct(self, a_tail: np.ndarray, b_head: np.ndarray,
                        full_b: np.ndarray) -> np.ndarray:
        """Apply phase correction to minimize discontinuity."""
        # Cross-correlate to find optimal alignment
        correlation = np.correlate(a_tail, b_head, mode='full')
        offset = np.argmax(correlation) - len(b_head) + 1
        
        # Shift b if needed (small adjustments only)
        if abs(offset) < 100:  # Max 100 samples shift
            if offset > 0:
                return np.concatenate([np.zeros(offset), full_b[:-offset]])
            elif offset < 0:
                return full_b[-offset:]
        
        return full_b
    
    def _get_crossfade_duration(self, structure: MusicalStructure) -> float:
        """Get crossfade duration in seconds based on bar length."""
        bar_duration = 60.0 / structure.tempo * structure.beats_per_bar
        return bar_duration * self.config.crossfade_bars
```

---

## Integration Plan

### Step 1: Integrate into StemRegenerator

**File:** `Resonate/audio_engine/generation/stem_regenerator.py`

**Changes:**
1. Add `use_bssr: bool = True` to StemRegenerator config
2. Add `_regenerate_with_bssr()` method
3. Route long audio (>30s) through BSSR pipeline

```python
class StemRegenerator:
    # ... existing code ...
    
    def regenerate_regions(self, audio, regions, musical_profile, 
                           stem_type, sample_rate, callbacks=None):
        
        duration = len(audio) / sample_rate
        
        # Use BSSR for long audio
        if duration > 30.0 and self.use_bssr:
            return self._regenerate_with_bssr(
                audio, musical_profile, stem_type, sample_rate, callbacks
            )
        
        # Use existing method for short audio
        return self._regenerate_standard(audio, regions, musical_profile,
                                          stem_type, sample_rate, callbacks)
    
    def _regenerate_with_bssr(self, audio, profile, stem_type, sr, callbacks):
        """Use BSSR pipeline for long-form regeneration."""
        from .musical_structure import MusicalStructureAnalyzer
        from .bar_aligned_chunker import BarAlignedChunker
        from .stem_orchestrator import StemSequentialOrchestrator
        
        # Analyze structure
        analyzer = MusicalStructureAnalyzer()
        structure = analyzer.analyze(audio, sr)
        
        # Create orchestrator
        orchestrator = StemSequentialOrchestrator(
            chunker=BarAlignedChunker(),
            generator=AutoregressiveGenerator(self.generator)
        )
        
        # Regenerate
        return orchestrator.regenerate_stem(
            audio, stem_type, profile, structure, callbacks
        )
```

### Step 2: Update UI

**File:** `Resonate/ui/app.py`

Add BSSR controls in the sidebar:

```python
st.subheader("🎵 Long-Form Generation (BSSR)")
st.caption("For songs > 30 seconds")

use_bssr = st.toggle(
    "Enable BSSR Pipeline",
    value=True,
    help="Beat-Synchronous Stem-Sequential Regeneration for full-length songs"
)

if use_bssr:
    overlap_bars = st.slider(
        "Overlap (bars)",
        min_value=2,
        max_value=8,
        value=4,
        help="More overlap = smoother transitions, but slower"
    )
    
    prefer_sections = st.checkbox(
        "Prefer section boundaries",
        value=True,
        help="Place chunk boundaries at verse/chorus transitions"
    )
```

---

## Testing Strategy

### Unit Tests

**File:** `Resonate/tests/test_bssr.py`

```python
def test_musical_structure_analyzer_extracts_beats():
    """Verify beat extraction works correctly."""
    
def test_musical_structure_analyzer_extracts_bars():
    """Verify bar boundary detection."""
    
def test_bar_aligned_chunker_creates_valid_chunks():
    """Verify chunks are within 30s limit."""
    
def test_bar_aligned_chunker_respects_section_boundaries():
    """Verify section boundaries are preferred."""
    
def test_optimal_cut_finder_finds_minimum():
    """Verify spectral distance minimization."""
    
def test_beat_aligned_stitcher_crossfades_smoothly():
    """Verify no clicks/pops at boundaries."""
    
def test_stem_orchestrator_respects_dependencies():
    """Verify drums generated before bass, etc."""
    
def test_full_bssr_pipeline_4_minute_song():
    """Integration test with real 4-minute song."""
```

### Integration Tests

```python
def test_taylors_version_with_bssr():
    """Full Taylor's Version regeneration using BSSR."""
    audio, sr = load_test_song("4_minute_song.mp3")
    
    # Should not truncate
    result = taylors_version_regenerate(audio, sr)
    
    # Verify full length
    assert len(result) / sr >= 4 * 60 - 1  # Within 1 second of original
    
    # Verify no audible discontinuities (spectral smoothness)
    discontinuities = detect_discontinuities(result)
    assert len(discontinuities) == 0
```

---

## File Structure After Implementation

```
Resonate/audio_engine/
├── generation/
│   ├── __init__.py
│   ├── jasco_generator.py        # Existing - no changes
│   ├── stem_regenerator.py       # Modified - add BSSR routing
│   ├── blender.py                # Existing - no changes
│   │
│   └── bssr/                     # NEW MODULE
│       ├── __init__.py
│       ├── musical_structure.py      # MusicalStructureAnalyzer
│       ├── bar_aligned_chunker.py    # BarAlignedChunker
│       ├── autoregressive_generator.py  # AutoregressiveGenerator
│       ├── stem_orchestrator.py      # StemSequentialOrchestrator
│       ├── optimal_cut_finder.py     # OptimalCutFinder
│       └── beat_aligned_stitcher.py  # BeatAlignedStitcher
│
└── profiling/
    └── musical_structure.py      # Could also go here
```

---

## Implementation Order

| Task | Component | Dependencies | Priority |
|------|-----------|--------------|----------|
| 1 | MusicalStructureAnalyzer | tempo_key_analyzer | High |
| 2 | BarAlignedChunker | MusicalStructureAnalyzer | High |
| 3 | OptimalCutFinder | None | Medium |
| 4 | BeatAlignedStitcher | OptimalCutFinder | Medium |
| 5 | AutoregressiveGenerator | JASCOGenerator | High |
| 6 | StemSequentialOrchestrator | All above | High |
| 7 | Integration | All above | High |
| 8 | UI Controls | Integration | Low |
| 9 | Tests | All above | Medium |
| 10 | Documentation | All above | Low |

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Duration Coverage** | 100% | Output length matches input |
| **Chunk Boundaries** | All bar-aligned | Manual verification |
| **Audible Discontinuities** | 0 | Spectral analysis + listening test |
| **Spectral Similarity** | >0.8 at cuts | OptimalCutFinder distance |
| **Phase Coherence** | No clicks/pops | Zero-crossing analysis |
| **Processing Speed** | <5x realtime | Benchmark on 4-min song |

---

## Research Contributions

This implementation makes several novel contributions:

1. **Bar-Aligned Chunking for AI Music Generation**
   - First application of musical structure-aware chunking for long-form generation
   - Preserves musical coherence at chunk boundaries

2. **Stem-Sequential Dependencies**
   - Novel approach where stems are generated in order of musical dependency
   - Each stem is conditioned on previously generated stems

3. **Optimal Cut-Point Selection**
   - Spectral distance minimization combined with beat-snapping
   - Reduces perceptual discontinuities at chunk boundaries

4. **Autoregressive Continuation for Multi-Chunk Generation**
   - Uses generated audio as context for next chunk
   - Maintains temporal coherence across long audio

These techniques could be adapted for other long-form AI generation tasks beyond music.
