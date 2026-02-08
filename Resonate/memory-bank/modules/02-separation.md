# Module 02: Source Separation (Demucs)

## Purpose
Isolate individual instruments (vocals, drums, bass, other) from mixed phone recording using Meta's Demucs v4 (HTDemucs). This is the critical foundation for per-stem enhancement - we can't enhance what we can't separate.

## Key Responsibilities
1. Initialize Demucs model with MPS/CPU device handling
2. Separate mixed audio into 4 stems (vocals, drums, bass, other)
3. Handle memory management for M1 Max (64GB limit)
4. Implement preview (fast) vs render (quality) modes
5. Cache separated stems for repeated enhancement iterations

---

## Implementation Guide

### Core Class Structure
```python
from enum import Enum
from pathlib import Path
from typing import Dict, Tuple
import torch
import numpy as np
from demucs.api import Separator
from demucs.pretrained import get_model

class SeparationQuality(Enum):
    PREVIEW = "preview"  # Fast, lower quality
    RENDER = "render"    # Slow, highest quality

@dataclass
class SeparatedStems:
    """Output of separation"""
    vocals: np.ndarray    # shape: (samples, 1), float32
    drums: np.ndarray
    bass: np.ndarray
    other: np.ndarray     # Instruments (guitar, keys, etc.)
    sample_rate: int
    quality_mode: SeparationQuality
    
class DemucsWrapper:
    """Wraps Demucs API with our configuration"""
    
    def __init__(
        self,
        device: str = "auto",  # "mps", "cuda", "cpu", or "auto"
        quality: SeparationQuality = SeparationQuality.RENDER,
        cache_dir: Path = None
    ):
        """
        Args:
            device: Processing device (auto-detects MPS availability)
            quality: PREVIEW (fast) or RENDER (high quality)
            cache_dir: Where to cache separated stems
        """
        self.device = self._setup_device(device)
        self.quality = quality
        self.cache_dir = cache_dir or Path("cache/stems")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Model selection based on quality
        if quality == SeparationQuality.PREVIEW:
            self.model_name = "htdemucs"       # 4-source, fast
            self.segment = 5                    # 5 sec segments (less memory)
            self.shifts = 0                     # No test-time augmentation
        else:  # RENDER
            self.model_name = "htdemucs_ft"    # Fine-tuned, best quality
            self.segment = 10                   # 10 sec segments (more context)
            self.shifts = 1                     # 1 shift for better quality
            
        self.separator = None  # Lazy initialization
        
    def separate(
        self, 
        audio_buffer: AudioBuffer,
        use_cache: bool = True
    ) -> SeparatedStems:
        """
        Main entry point: separate audio into stems
        
        Args:
            audio_buffer: Input from ingest module
            use_cache: Check cache before processing
            
        Returns:
            SeparatedStems with 4 isolated instruments
        """
        # Check cache first
        if use_cache:
            cached = self._load_from_cache(audio_buffer)
            if cached is not None:
                return cached
        
        # Initialize model if needed (lazy)
        if self.separator is None:
            self._initialize_model()
            
        # Separate
        stems_dict = self._separate_audio(audio_buffer)
        
        # Package results
        result = SeparatedStems(
            vocals=stems_dict["vocals"],
            drums=stems_dict["drums"],
            bass=stems_dict["bass"],
            other=stems_dict["other"],
            sample_rate=audio_buffer.sample_rate,
            quality_mode=self.quality
        )
        
        # Cache results
        if use_cache:
            self._save_to_cache(audio_buffer, result)
            
        return result
```

---

## Technical Deep Dive

### Model Selection: htdemucs vs htdemucs_ft

**htdemucs (Hybrid Transformer Demucs)**
- Architecture: Hybrid CNN + Transformer
- Training: Original dataset
- Speed: ~2x faster than fine-tuned
- Quality: Very good
- **Use case**: Preview mode

**htdemucs_ft (Fine-Tuned)**
- Architecture: Same as htdemucs
- Training: Fine-tuned on additional high-quality data
- Speed: Slower due to larger segment size
- Quality: Best available
- **Use case**: Final render

### Device Setup (MPS for M1 Max)
```python
def _setup_device(self, device: str) -> torch.device:
    """Configure processing device with fallback"""
    
    if device == "auto":
        # Auto-detect best device
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
            
    try:
        torch_device = torch.device(device)
        
        # MPS-specific memory management
        if device == "mps":
            # Reserve 75% of memory for Demucs
            # (M1 Max 64GB → ~48GB available)
            torch.mps.set_per_process_memory_fraction(0.75)
            print(f"✓ Using MPS acceleration (Metal)")
        elif device == "cuda":
            print(f"✓ Using CUDA acceleration")
        else:
            print(f"⚠ Using CPU (slow - expect 10-20 min per song)")
            
        return torch_device
        
    except Exception as e:
        print(f"✗ Failed to initialize {device}, falling back to CPU")
        return torch.device("cpu")
```

### Model Initialization (Lazy Loading)
```python
def _initialize_model(self):
    """Load Demucs model on first use"""
    print(f"Loading Demucs model: {self.model_name} on {self.device}...")
    
    try:
        self.separator = Separator(
            model=self.model_name,
            segment=self.segment,
            shifts=self.shifts,
            device=str(self.device),
            progress=True  # Show progress bar
        )
        print("✓ Model loaded successfully")
        
    except Exception as e:
        # Fallback: Try CPU if MPS fails
        if str(self.device) == "mps":
            print(f"✗ MPS initialization failed: {e}")
            print("Retrying with CPU...")
            self.device = torch.device("cpu")
            self.separator = Separator(
                model=self.model_name,
                segment=self.segment,
                shifts=self.shifts,
                device="cpu",
                progress=True
            )
        else:
            raise RuntimeError(f"Failed to initialize Demucs: {e}")
```

### Separation Process
```python
def _separate_audio(self, audio_buffer: AudioBuffer) -> Dict[str, np.ndarray]:
    """Run Demucs separation"""
    
    # Demucs expects (channels, samples) shape
    audio_demucs = audio_buffer.data.T  # (samples, 1) → (1, samples)
    
    # Convert to tensor
    audio_tensor = torch.from_numpy(audio_demucs).to(self.device)
    
    try:
        # Run separation
        # Returns: origin (original audio) and separated stems
        origin, stems = self.separator.separate_tensor(
            audio_tensor,
            sample_rate=audio_buffer.sample_rate
        )
        
        # Convert back to numpy
        # stems shape: (sources, channels, samples)
        # sources: [vocals, drums, bass, other]
        stems_np = stems.cpu().numpy()
        
        # Extract individual stems and convert to (samples, channels)
        stems_dict = {
            "vocals": stems_np[0].T,  # (1, samples) → (samples, 1)
            "drums": stems_np[1].T,
            "bass": stems_np[2].T,
            "other": stems_np[3].T
        }
        
        return stems_dict
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            raise MemoryError(
                f"Ran out of memory during separation. "
                f"Try preview mode or process shorter segments."
            )
        else:
            raise RuntimeError(f"Separation failed: {e}")
```

---

## Caching Strategy

### Cache Key Generation
```python
import hashlib

def _get_cache_key(self, audio_buffer: AudioBuffer) -> str:
    """Generate unique key for caching"""
    
    # Hash based on:
    # 1. Audio content (first/last 1000 samples)
    # 2. Duration
    # 3. Quality mode
    # 4. Model name
    
    header = audio_buffer.data[:1000].tobytes()
    footer = audio_buffer.data[-1000:].tobytes()
    
    content = f"{header}{footer}{audio_buffer.metadata.duration}{self.quality.value}{self.model_name}"
    
    return hashlib.sha256(content.encode()).hexdigest()[:16]
```

### Cache Save/Load
```python
def _save_to_cache(self, audio_buffer: AudioBuffer, stems: SeparatedStems):
    """Save separated stems to disk"""
    cache_key = self._get_cache_key(audio_buffer)
    cache_path = self.cache_dir / cache_key
    cache_path.mkdir(exist_ok=True)
    
    # Save each stem as numpy array (efficient)
    np.save(cache_path / "vocals.npy", stems.vocals)
    np.save(cache_path / "drums.npy", stems.drums)
    np.save(cache_path / "bass.npy", stems.bass)
    np.save(cache_path / "other.npy", stems.other)
    
    # Save metadata
    import json
    meta = {
        "sample_rate": stems.sample_rate,
        "quality": stems.quality_mode.value,
        "model": self.model_name,
        "timestamp": datetime.now().isoformat()
    }
    (cache_path / "meta.json").write_text(json.dumps(meta, indent=2))
    
def _load_from_cache(self, audio_buffer: AudioBuffer) -> Optional[SeparatedStems]:
    """Try to load stems from cache"""
    cache_key = self._get_cache_key(audio_buffer)
    cache_path = self.cache_dir / cache_key
    
    if not (cache_path / "vocals.npy").exists():
        return None
        
    try:
        return SeparatedStems(
            vocals=np.load(cache_path / "vocals.npy"),
            drums=np.load(cache_path / "drums.npy"),
            bass=np.load(cache_path / "bass.npy"),
            other=np.load(cache_path / "other.npy"),
            sample_rate=audio_buffer.sample_rate,
            quality_mode=self.quality
        )
    except Exception as e:
        print(f"Cache load failed: {e}")
        return None
```

---

## Design Decisions

### 🎯 Decision 1: Model Choice
**Choice: htdemucs_ft for render, htdemucs for preview**

**Rationale:**
- htdemucs_ft has measurably better separation quality (tested on MUSDB18-HQ)
- Speed difference is acceptable for render mode (~5 min vs ~3 min on M1 Max)
- Preview mode needs fast feedback, htdemucs is sufficient

### 🎯 Decision 2: Segment Size
**Choice: 10 sec for render, 5 sec for preview**

**Rationale:**
- Longer segments = more context = better separation
- But longer segments = more memory
- 10 sec is Demucs default and fits comfortably in 64GB RAM
- 5 sec for preview reduces memory by half, fast preview feedback

### 🎯 Decision 3: Test-Time Augmentation (shifts)
**Choice: 1 shift for render, 0 for preview**

**Rationale:**
- Shifts: Process audio multiple times with slight offsets, average results
- 1 shift: 2x processing time, ~1-2 dB SDR improvement (worth it for render)
- 0 shifts: Fast preview, good enough to judge results

### 🎯 Decision 4: Caching Stems vs Full Audio
**Choice: Cache stems, not final mixed audio**

**Rationale:**
- Separation is expensive (80% of processing time)
- Enhancement parameters change during experimentation
- Caching stems allows re-enhancement without re-separation
- Disk space: ~40 MB per song (acceptable)

---

## Handling Degraded Input (Phone Recordings)

### Challenge: Demucs Trained on Clean Audio
Demucs is trained on high-quality studio recordings (MUSDB18). Phone recordings from crowds present:
- **Low SNR**: Crowd noise, wind, handling noise
- **Limited bandwidth**: ~100Hz-8kHz (vs full 20Hz-20kHz)
- **Reverb**: Venue acoustics mixed with direct sound
- **Distortion**: Phone mic clipping, AGC artifacts

### Pre-Conditioning Strategy (Future Enhancement)
```python
def precondition(self, audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Prepare phone recording for Demucs separation
    
    Steps:
    1. Gentle noise reduction (don't overdo it - Demucs robust to some noise)
    2. De-reverb if severe (helps separation accuracy)
    3. High-pass filter at 30Hz (remove rumble)
    4. Normalize peak to -3dB (prevent clipping in separator)
    
    NOTE: This is experimental - test if it helps or hurts!
    """
    # To be implemented after testing baseline Demucs performance
    pass
```

### Quality Assessment
```python
def assess_separation_quality(self, stems: SeparatedStems) -> float:
    """
    Estimate separation quality (0-1 score)
    
    Heuristics:
    - Check for stem bleeding (vocals in drums, etc.)
    - Measure stem isolation (SNR of each stem)
    - Detect artifacts (metallic/robotic sounds)
    
    Returns quality score: 1.0 = perfect, 0.5 = moderate bleed, 0 = failed
    """
    # Simple heuristic: Measure correlation between stems
    # Well-separated stems should have low correlation
    
    correlations = []
    stem_arrays = [stems.vocals, stems.drums, stems.bass, stems.other]
    
    for i in range(len(stem_arrays)):
        for j in range(i+1, len(stem_arrays)):
            corr = np.corrcoef(stem_arrays[i].flatten(), stem_arrays[j].flatten())[0, 1]
            correlations.append(abs(corr))
    
    avg_correlation = np.mean(correlations)
    
    # Convert to quality score (lower correlation = better)
    quality = 1.0 - avg_correlation
    
    return max(0.0, min(1.0, quality))
```

---

## Error Handling

### Common Issues & Solutions

| Issue | Detection | Solution |
|-------|-----------|----------|
| Out of memory | RuntimeError with "memory" | Fallback to smaller segment or CPU |
| MPS initialization fails | MPS device error | Auto-fallback to CPU |
| Model download fails | Network error | Clear cache, retry, or use local model |
| Audio too long | Estimate memory usage | Chunk processing (future) or reject |
| Separation quality poor | Quality score < 0.5 | Warn user, suggest pre-conditioning |

### Memory Estimation
```python
def estimate_memory_usage(self, audio_buffer: AudioBuffer) -> int:
    """
    Estimate peak memory usage in MB
    
    Rule of thumb:
    - Input audio: ~10 MB/min (float32 mono @ 44.1kHz)
    - Demucs model: ~500 MB
    - Processing buffer: ~3x input size
    - Output stems: 4x input size
    
    Total ≈ 500 + 7x input size
    """
    duration_min = audio_buffer.metadata.duration / 60
    input_size_mb = duration_min * 10
    
    total_mb = 500 + (7 * input_size_mb)
    
    return int(total_mb)
    
def check_memory_available(self, required_mb: int) -> bool:
    """Check if enough memory available"""
    import psutil
    available_mb = psutil.virtual_memory().available / (1024 ** 2)
    
    return available_mb > required_mb * 1.2  # 20% safety margin
```

---

## Testing Strategy

### Unit Tests
```python
def test_device_selection_mps():
    """Test MPS device is selected on M1 Mac"""
    wrapper = DemucsWrapper(device="auto")
    assert str(wrapper.device) == "mps"
    
def test_model_initialization():
    """Test model loads without error"""
    wrapper = DemucsWrapper(quality=SeparationQuality.PREVIEW)
    wrapper._initialize_model()
    assert wrapper.separator is not None
    
def test_cache_key_consistency():
    """Test cache keys are consistent for same audio"""
    buffer = create_test_buffer()
    wrapper = DemucsWrapper()
    
    key1 = wrapper._get_cache_key(buffer)
    key2 = wrapper._get_cache_key(buffer)
    
    assert key1 == key2
```

### Integration Tests
```python
def test_separation_output_shape():
    """Test separation returns correct shapes"""
    buffer = load_test_audio("fixtures/phone_recording.mp3")
    wrapper = DemucsWrapper(quality=SeparationQuality.PREVIEW)
    
    stems = wrapper.separate(buffer, use_cache=False)
    
    # Check all stems have same length
    assert stems.vocals.shape == stems.drums.shape
    assert stems.vocals.shape == stems.bass.shape
    assert stems.vocals.shape == stems.other.shape
    
    # Check mono
    assert stems.vocals.shape[1] == 1
    
def test_caching_works():
    """Test stems are cached and retrieved"""
    buffer = load_test_audio("fixtures/short_clip.mp3")
    wrapper = DemucsWrapper()
    
    # First call: compute
    stems1 = wrapper.separate(buffer, use_cache=True)
    
    # Second call: should be instant (from cache)
    import time
    start = time.time()
    stems2 = wrapper.separate(buffer, use_cache=True)
    elapsed = time.time() - start
    
    assert elapsed < 1.0  # Should be <1 sec if cached
    assert np.allclose(stems1.vocals, stems2.vocals)
```

---

## Performance Benchmarks

### Expected Processing Times (M1 Max, 64GB RAM)

| Audio Length | Quality | Model | Device | Time | Memory |
|--------------|---------|-------|--------|------|--------|
| 3 min | Preview | htdemucs | MPS | ~90 sec | ~4 GB |
| 3 min | Render | htdemucs_ft | MPS | ~5 min | ~6 GB |
| 10 min | Render | htdemucs_ft | MPS | ~15 min | ~12 GB |
| 3 min | Render | htdemucs_ft | CPU | ~30 min | ~4 GB |

### Optimization Opportunities

1. **Batch Processing**: Process multiple files simultaneously (if memory allows)
2. **Dynamic Chunking**: Split very long files into overlapping chunks
3. **Model Quantization**: Use int8 model for faster inference (slight quality loss)
4. **Streaming**: Process in real-time as audio loads (complex)

---

## Integration with Pipeline

### Output Contract
```python
# Next stage (enhancement) expects:
SeparatedStems(
    vocals: np.ndarray,    # (samples, 1), float32, [-1, 1]
    drums: np.ndarray,     # (samples, 1), float32, [-1, 1]
    bass: np.ndarray,      # (samples, 1), float32, [-1, 1]
    other: np.ndarray,     # (samples, 1), float32, [-1, 1]
    sample_rate: int,      # 44100
    quality_mode: SeparationQuality
)
```

### Usage in Pipeline
```python
class AudioPipeline:
    def process(self, file_path: Path, quality: SeparationQuality):
        # Stage 1: Ingest
        buffer = self.ingest.load(file_path)
        
        # Stage 2: Separation
        separator = DemucsWrapper(quality=quality)
        stems = separator.separate(buffer)
        
        # Check quality
        quality_score = separator.assess_separation_quality(stems)
        if quality_score < 0.5:
            print("⚠ Separation quality is poor - results may not be optimal")
        
        # Stage 3: Per-stem enhancement
        enhanced_stems = self.enhance_stems(stems)
        # ...
```

---

## Future Enhancements (v2)

1. **Custom Demucs Fine-Tuning**: Train on phone recordings for better separation
2. **Alternative Models**: Try Spleeter, OpenUnmix, compare quality
3. **Adaptive Segment Size**: Auto-select based on available memory
4. **GPU Support**: Add CUDA path for NVIDIA users
5. **Real-Time Separation**: Stream processing for live audio
6. **Stem Visualization**: Show waveforms/spectrograms per stem in UI

---

## Dependencies
```python
# requirements.txt (for this module)
demucs>=4.0.0          # Source separation
torch>=2.0.0           # PyTorch
torchaudio>=2.0.0      # Audio utilities
psutil>=5.9.0          # Memory monitoring
```

## File Location
```
resonate/
└── audio_engine/
    └── separator.py   # This module
```

---

## References
- **Demucs Paper**: https://arxiv.org/abs/2111.03600
- **Demucs Repository**: https://github.com/facebookresearch/demucs
- **API Documentation**: https://github.com/facebookresearch/demucs/blob/main/docs/training.md
- **MUSDB18 Dataset**: https://sigsep.github.io/datasets/musdb.html
