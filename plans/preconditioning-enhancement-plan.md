# Pre-Conditioning Enhancement Plan

## Overview

Transform phone recordings to studio quality by adding a **Pre-Conditioning Pipeline** that runs BEFORE Demucs separation.

**Problem**: The current pipeline feeds raw, degraded phone audio directly to Demucs → **Garbage in = Garbage out**

**Solution**: Pre-condition the audio (noise reduction, de-clipping, dynamics restoration) BEFORE separation.

---

## Current Pipeline Flow

```
Phone Recording
      ↓
   INGEST
      ↓
  SEPARATE (Demucs)  ← Struggles with noisy/clipped input
      ↓
  ENHANCE (per-stem)
      ↓
    MIX
      ↓
  RESTORE (freq + dereverb)
      ↓
  POLISH (MBD)
      ↓
   MASTER
```

## Target Pipeline Flow

```
Phone Recording
      ↓
   INGEST
      ↓
★ PRE-CONDITION ★  ← NEW STAGE
   - Noise Reduction (crowd, hiss, rumble)
   - De-clipping (repair AGC distortion)
   - Dynamics Restoration (undo AGC compression)
      ↓
  SEPARATE (Demucs)  ← Now gets cleaner input!
      ↓
  ENHANCE (per-stem)
      ↓
    MIX
      ↓
  RESTORE (freq + dereverb)
      ↓
  POLISH (MBD)
      ↓
★ STEM REGENERATION (JASCO) ★  ← Already implemented
      ↓
   MASTER
```

---

## Implementation Tasks

### Phase A: Complete Preconditioning Module (2-3 hours)

#### Task A1: Noise Reducer ✅ DONE
- File: `audio_engine/preconditioning/noise_reducer.py`
- Features:
  - Global noise reduction using noisereduce library
  - Auto-detection of quiet sections for noise profile
  - Frequency-specific reduction (more on rumble, preserve highs)

#### Task A2: De-clipper (1 hour)
- File: `audio_engine/preconditioning/declip.py`
- Features:
  ```python
  class DeclipConfig:
      detection_threshold: float = 0.99  # Consider clipped above this
      interpolation_method: str = "cubic"  # cubic, sinc, ML
      max_clip_duration_ms: float = 5.0  # Max clip length to repair
  
  class Declipper:
      def detect_clipping(self, audio) -> List[ClipRegion]
      def repair_clip(self, audio, region) -> np.ndarray
      def process(self, audio) -> np.ndarray
  ```
- Implementation:
  1. Detect clipped regions (samples at ±0.99-1.0)
  2. Mark clip boundaries
  3. Interpolate using cubic spline or sinc
  4. Blend repaired sections

#### Task A3: Dynamics Restorer (1 hour)
- File: `audio_engine/preconditioning/dynamics.py`
- Features:
  ```python
  class DynamicsConfig:
      expansion_threshold_db: float = -40.0  # Expand below this
      expansion_ratio: float = 1.5  # Upward expansion ratio
      attack_ms: float = 10.0
      release_ms: float = 100.0
  
  class DynamicsRestorer:
      def analyze_dynamics(self, audio) -> DynamicsProfile
      def apply_upward_expansion(self, audio) -> np.ndarray
  ```
- Implementation:
  1. Analyze dynamic range of input
  2. Apply upward expansion to restore dynamics killed by phone AGC
  3. Preserve transients

#### Task A4: Preconditioning Pipeline (30 min)
- File: `audio_engine/preconditioning/pipeline.py`
- Features:
  ```python
  class PreConditioningConfig:
      enable_noise_reduction: bool = True
      noise_reduction_strength: float = 0.5
      enable_declipping: bool = True
      enable_dynamics_restoration: bool = True
      dynamics_expansion_ratio: float = 1.5
  
  class PreConditioningPipeline:
      def process(self, audio, sample_rate) -> PreConditioningResult:
          # 1. Analyze input quality
          # 2. Apply noise reduction
          # 3. De-clip if needed
          # 4. Restore dynamics
          # 5. Return with metrics
  ```

---

### Phase B: Integrate into Main Pipeline (1 hour)

#### Task B1: Update PipelineConfig
- File: `audio_engine/pipeline.py`
- Add:
  ```python
  @dataclass
  class PipelineConfig:
      # Existing settings...
      
      # Pre-conditioning (NEW)
      enable_preconditioning: bool = True
      precondition_noise_reduction: bool = True
      precondition_noise_strength: float = 0.5
      precondition_declip: bool = True
      precondition_dynamics: bool = True
  ```

#### Task B2: Add Preconditioning Stage
- In `AudioPipeline.process()`, add after ingest:
  ```python
  # === STAGE 1.5: PRE-CONDITIONING ===
  if self.config.enable_preconditioning:
      stage_start = time.time()
      logger.info("Stage 1.5: Pre-conditioning input...")
      
      preconditioner = self._get_preconditioner()
      audio = preconditioner.process(original_buffer.audio, sample_rate)
      
      stage_times['preconditioning'] = time.time() - stage_start
      logger.info(f"  ✅ Pre-conditioning complete")
  ```

#### Task B3: Create Lazy Loader
- Add `_get_preconditioner()` method to AudioPipeline

---

### Phase C: Update UI (1 hour)

#### Task C1: Add Preconditioning Controls
- File: `ui/app.py`
- In sidebar, add new section:
  ```python
  st.subheader("🎚️ Pre-Conditioning")
  st.caption("Clean input audio BEFORE separation (critical for phone recordings)")
  
  enable_precondition = st.toggle(
      "Enable Pre-Conditioning",
      value=True,
      help="Recommended ON for phone recordings"
  )
  
  if enable_precondition:
      noise_strength = st.slider(
          "Noise Reduction Strength",
          min_value=0.0,
          max_value=1.0,
          value=0.5,
          help="Higher = more noise removed, risk of artifacts"
      )
      
      enable_declip = st.toggle(
          "De-Clipping",
          value=True,
          help="Repair distorted peaks from phone AGC"
      )
      
      enable_dynamics = st.toggle(
          "Dynamics Restoration",
          value=True,
          help="Undo compression from phone AGC"
      )
  ```

#### Task C2: Pass Config to Pipeline
- Update PipelineConfig creation to include new settings

---

### Phase D: Testing (30 min)

#### Task D1: Create Test File
- File: `tests/test_preconditioning.py`
- Tests:
  ```python
  def test_noise_reducer_reduces_noise():
      """Verify SNR improvement."""
  
  def test_declip_repairs_clipping():
      """Verify clipped peaks are interpolated."""
  
  def test_dynamics_restorer_expands_range():
      """Verify dynamic range increase."""
  
  def test_preconditioning_pipeline_integration():
      """Full pipeline test."""
  ```

#### Task D2: Verify with Real Recording
- Use actual phone recording to verify improvement

---

## File Structure After Implementation

```
audio_engine/preconditioning/
├── __init__.py          ✅ Done
├── noise_reducer.py     ✅ Done
├── declip.py            ⏳ To implement
├── dynamics.py          ⏳ To implement
└── pipeline.py          ⏳ To implement
```

---

## Estimated Total Time

| Phase | Task | Time |
|-------|------|------|
| A | Complete preconditioning module | 2-3 hours |
| B | Integrate into main pipeline | 1 hour |
| C | Update UI | 1 hour |
| D | Testing | 30 min |
| **Total** | | **4.5-5.5 hours** |

---

## Priority Order

1. **Task A2**: De-clipper (most impactful for phone AGC distortion)
2. **Task A3**: Dynamics Restorer
3. **Task A4**: Preconditioning Pipeline
4. **Task B1-B3**: Pipeline Integration
5. **Task C1-C2**: UI Updates
6. **Task D1-D2**: Testing

---

## Success Criteria

After implementation:

1. **Noise Floor**: ≥10 dB reduction in noise
2. **Clipping**: 0% hard clips in output (repaired)
3. **Dynamic Range**: ≥3 dB increase
4. **Separation Quality**: Noticeably cleaner stems from Demucs
5. **UI**: Pre-conditioning controls visible and functional

---

## Dependencies

Existing (already in requirements.txt):
- `noisereduce` - For noise reduction
- `scipy` - For signal processing
- `numpy` - For array operations

No new dependencies required.

---

## Code Examples

### De-clipper Core Logic

```python
def detect_clipping(audio: np.ndarray, threshold: float = 0.99) -> List[Tuple[int, int]]:
    """Find clipped regions."""
    is_clipped = np.abs(audio) >= threshold
    
    # Find contiguous regions
    regions = []
    in_clip = False
    start = 0
    
    for i, clipped in enumerate(is_clipped):
        if clipped and not in_clip:
            start = i
            in_clip = True
        elif not clipped and in_clip:
            regions.append((start, i))
            in_clip = False
    
    return regions


def repair_clip_cubic(audio: np.ndarray, start: int, end: int) -> np.ndarray:
    """Repair clipped region using cubic interpolation."""
    from scipy.interpolate import CubicSpline
    
    # Get surrounding samples (not clipped)
    margin = 5
    x_before = np.arange(max(0, start - margin), start)
    x_after = np.arange(end, min(len(audio), end + margin))
    x_known = np.concatenate([x_before, x_after])
    y_known = audio[x_known]
    
    # Interpolate
    cs = CubicSpline(x_known, y_known)
    x_clipped = np.arange(start, end)
    audio[start:end] = cs(x_clipped)
    
    return audio
```

### Dynamics Restorer Core Logic

```python
def upward_expansion(audio: np.ndarray, 
                    threshold_db: float = -40.0,
                    ratio: float = 1.5) -> np.ndarray:
    """Apply upward expansion to restore dynamics."""
    from scipy.signal import hilbert
    
    # Get envelope
    envelope = np.abs(hilbert(audio))
    envelope_db = 20 * np.log10(envelope + 1e-10)
    
    # Calculate gain for samples below threshold
    gain_db = np.zeros_like(envelope_db)
    below_threshold = envelope_db < threshold_db
    
    if np.any(below_threshold):
        # Expansion: make quiet parts quieter (increases dynamic range)
        distance_below = threshold_db - envelope_db[below_threshold]
        gain_db[below_threshold] = -distance_below * (ratio - 1)
    
    # Convert to linear and apply
    gain = 10 ** (gain_db / 20)
    return audio * gain
```

---

## Notes for Orchestrator

- Start with **Phase A** to complete the module
- Each task can be delegated to Code mode
- Test after each phase before moving to next
- The NoiseReducer is already done - start with Declip
