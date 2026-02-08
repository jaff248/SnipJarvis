# Module 01: Audio Ingest

## Purpose
Load, validate, and prepare raw phone recordings for processing pipeline. This is the entry point for all audio, handling format conversion, quality validation, and metadata extraction.

## Key Responsibilities
1. Load audio files in various formats (MP3, M4A, WAV, FLAC)
2. Validate input quality and detect potential issues
3. Normalize audio to consistent format for downstream processing
4. Extract and preserve metadata for reconstruction
5. Detect input characteristics (sample rate, bit depth, duration, noise floor)

---

## Implementation Guide

### Core Class Structure
```python
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import soundfile as sf
import librosa

@dataclass
class AudioMetadata:
    """Metadata extracted from input audio"""
    sample_rate: int
    channels: int
    duration: float  # seconds
    bit_depth: Optional[int]
    format: str
    file_size: int  # bytes
    estimated_snr: float  # dB
    spectral_centroid: float  # Hz
    rms_level: float  # dB
    clipping_detected: bool
    
@dataclass  
class AudioBuffer:
    """Normalized audio ready for processing"""
    data: np.ndarray  # shape: (samples, channels) or (samples,), dtype: float32, range: [-1, 1]
    sample_rate: int
    metadata: AudioMetadata
    
class AudioIngest:
    def __init__(self, target_sample_rate: int = 44100):
        """
        Args:
            target_sample_rate: Internal processing sample rate
                              44.1kHz is good balance of quality/speed
                              (Demucs can handle 44.1k or 48k)
        """
        self.target_sr = target_sample_rate
        
    def load(self, file_path: Path) -> AudioBuffer:
        """Main entry point: load and prepare audio"""
        # 1. Load with soundfile (preserves native format)
        # 2. Validate
        # 3. Convert to mono if needed (or keep stereo? see decision below)
        # 4. Resample if needed
        # 5. Normalize
        # 6. Extract metadata
        
    def validate(self, audio: np.ndarray, sr: int) -> None:
        """Validate input meets minimum requirements"""
        # Check duration (minimum 5 seconds, maximum 30 minutes)
        # Check for complete silence
        # Check for extreme clipping
        # Warn if sample rate is very low (<16kHz)
        
    def analyze(self, audio: np.ndarray, sr: int) -> AudioMetadata:
        """Extract characteristics for downstream decisions"""
        # Estimate SNR
        # Compute spectral centroid (detect muffled audio)
        # Detect clipping
        # Measure RMS level
```

---

## Technical Specifications

### Supported Input Formats
| Format | Priority | Notes |
|--------|----------|-------|
| MP3 | High | Most common phone recording export |
| M4A/AAC | High | iPhone default format |
| WAV | High | Lossless, no decoding needed |
| FLAC | Medium | Lossless compression |
| OGG | Low | Rare for phone recordings |

### Loading Strategy
```python
def load(self, file_path: Path) -> AudioBuffer:
    """Load audio with format-agnostic handling"""
    try:
        # Try soundfile first (handles most formats via libsndfile)
        data, sr = sf.read(str(file_path), dtype='float32')
    except Exception:
        # Fallback to librosa for tricky formats (MP3, M4A)
        data, sr = librosa.load(str(file_path), sr=None, mono=False)
    
    # Convert to (samples, channels) shape if needed
    if data.ndim == 1:
        data = data[:, np.newaxis]  # mono: (samples,) → (samples, 1)
    elif data.ndim == 2 and data.shape[0] < data.shape[1]:
        data = data.T  # Fix transposed audio
        
    # Validate before processing
    self.validate(data, sr)
    
    # Resample if needed
    if sr != self.target_sr:
        data = librosa.resample(data.T, orig_sr=sr, target_sr=self.target_sr).T
        sr = self.target_sr
        
    # Extract metadata
    metadata = self.analyze(data, sr)
    
    return AudioBuffer(data=data, sample_rate=sr, metadata=metadata)
```

### Validation Rules
```python
class ValidationError(Exception):
    """Raised when input audio fails validation"""
    pass

def validate(self, audio: np.ndarray, sr: int) -> None:
    duration = len(audio) / sr
    
    # Duration checks
    if duration < 5.0:
        raise ValidationError(f"Audio too short: {duration:.1f}s (minimum 5s)")
    if duration > 1800:  # 30 minutes
        raise ValidationError(f"Audio too long: {duration/60:.1f}min (maximum 30min)")
    
    # Silence detection
    rms = np.sqrt(np.mean(audio**2))
    if rms < 1e-5:  # -100 dB
        raise ValidationError("Audio is completely silent")
    
    # Clipping detection
    clipping_ratio = np.sum(np.abs(audio) > 0.99) / audio.size
    if clipping_ratio > 0.05:  # >5% of samples clipped
        warnings.warn(f"Severe clipping detected: {clipping_ratio*100:.1f}% of samples")
    
    # Sample rate warning
    if sr < 16000:
        warnings.warn(f"Very low sample rate: {sr}Hz - quality may be limited")
```

### Metadata Analysis
```python
def analyze(self, audio: np.ndarray, sr: int) -> AudioMetadata:
    """Extract characteristics for pipeline decisions"""
    
    # Basic properties
    duration = len(audio) / sr
    channels = audio.shape[1] if audio.ndim > 1 else 1
    
    # RMS level
    rms = np.sqrt(np.mean(audio**2))
    rms_db = 20 * np.log10(rms + 1e-10)
    
    # Estimate SNR (rough estimate using temporal variance)
    # High variance in energy = likely signal, low variance = likely noise
    frame_size = int(0.1 * sr)  # 100ms frames
    frame_energy = np.array([
        np.mean(audio[i:i+frame_size]**2) 
        for i in range(0, len(audio) - frame_size, frame_size)
    ])
    signal_energy = np.percentile(frame_energy, 90)  # Loud frames
    noise_energy = np.percentile(frame_energy, 10)   # Quiet frames
    estimated_snr = 10 * np.log10(signal_energy / (noise_energy + 1e-10))
    
    # Spectral centroid (brightness measure)
    # Low centroid (<2kHz) indicates muffled/phone recording
    centroid = librosa.feature.spectral_centroid(y=audio.flatten(), sr=sr)[0]
    mean_centroid = np.mean(centroid)
    
    # Clipping detection
    clipping_detected = np.any(np.abs(audio) > 0.99)
    
    return AudioMetadata(
        sample_rate=sr,
        channels=channels,
        duration=duration,
        bit_depth=None,  # Not easily extractable from float data
        format=None,     # Set by caller
        file_size=0,     # Set by caller
        estimated_snr=estimated_snr,
        spectral_centroid=mean_centroid,
        rms_level=rms_db,
        clipping_detected=clipping_detected
    )
```

---

## Design Decisions

### 🎯 Decision 1: Mono vs Stereo Processing
**Options:**
- A) Convert all input to mono immediately
- B) Preserve stereo through entire pipeline
- C) Convert to mono for separation, restore stereo at mix stage

**Choice: A) Convert to mono**

**Rationale:**
- Phone recordings from crowd are pseudo-stereo at best (mostly mono + ambience)
- Demucs trained on stereo but phone recordings don't have meaningful stereo field
- Processing mono is 2x faster and uses less memory
- Stereo width can be added back artificially at mastering stage if desired

**Implementation:**
```python
if audio.shape[1] == 2:  # Stereo input
    audio = np.mean(audio, axis=1, keepdims=True)  # → (samples, 1)
```

### 🎯 Decision 2: Target Sample Rate
**Options:**
- 44.1 kHz (CD quality)
- 48 kHz (video standard)
- Keep native sample rate

**Choice: 44.1 kHz**

**Rationale:**
- Demucs works with both 44.1k and 48k
- 44.1k is standard for music distribution
- Phone recordings are often 44.1k or 48k anyway
- Lower than 48k saves ~10% computation time

### 🎯 Decision 3: Normalization Strategy
**Options:**
- A) Peak normalize to -1 dB
- B) RMS normalize to -20 dB
- C) Don't normalize (preserve dynamic range)

**Choice: C) Don't normalize during ingest**

**Rationale:**
- Preserve original dynamic range for SNR estimation
- Normalization can affect noise floor detection
- Demucs handles various input levels fine
- Normalize at mastering stage instead

---

## Error Handling

### Common Issues & Solutions

| Issue | Detection | Solution |
|-------|-----------|----------|
| Corrupted file | sf.read() exception | Try librosa, then fail with clear message |
| Unsupported format | Both loaders fail | Suggest conversion with ffmpeg |
| Sample rate mismatch | sr != target_sr | Automatic resampling with librosa |
| Clipped audio | >5% samples at ±1.0 | Warn user, proceed with processing |
| Near-silence | RMS < -100 dB | Reject with error |
| Excessive length | duration > 30 min | Reject (memory concerns) or chunk |

### Example Error Messages
```python
# Bad error:
raise Exception("Audio is bad")

# Good error:
raise ValidationError(
    f"Audio is severely clipped ({clipping_ratio*100:.1f}% of samples at maximum level). "
    f"This may indicate recording distortion. Processing may not improve quality."
)
```

---

## Testing Strategy

### Unit Tests
```python
def test_load_mp3():
    """Test loading MP3 file"""
    buffer = ingest.load(Path("fixtures/phone_recording.mp3"))
    assert buffer.sample_rate == 44100
    assert buffer.data.dtype == np.float32
    assert -1.0 <= buffer.data.max() <= 1.0
    
def test_validation_rejects_silent():
    """Test silent audio is rejected"""
    silent = np.zeros((44100, 1), dtype=np.float32)
    with pytest.raises(ValidationError, match="silent"):
        ingest.validate(silent, 44100)
        
def test_stereo_to_mono_conversion():
    """Test stereo is converted to mono"""
    stereo = np.random.randn(44100, 2).astype(np.float32)
    # ... assert becomes mono
```

### Integration Test
```python
def test_full_ingest_pipeline():
    """Test complete ingest workflow with real file"""
    buffer = ingest.load(Path("fixtures/crowd_recording.m4a"))
    
    # Check output format
    assert buffer.data.ndim == 2
    assert buffer.data.shape[1] == 1  # Mono
    assert buffer.sample_rate == 44100
    
    # Check metadata
    assert buffer.metadata.duration > 5.0
    assert buffer.metadata.estimated_snr < 20  # Noisy recording
    assert buffer.metadata.spectral_centroid < 3000  # Phone-quality
```

---

## Performance Considerations

### Memory Usage
- Audio stored as float32: ~10 MB per minute of mono 44.1kHz
- Peak memory during resample: 2x input size
- **Mitigation**: Stream processing for very long files (future)

### Processing Time
- Loading: <1 second for typical 3-minute song
- Resampling: ~2 seconds for 3-minute song (48k→44.1k)
- Metadata extraction: <1 second
- **Total**: ~3-4 seconds for ingest stage

---

## Integration with Pipeline

### Output Contract
```python
# Next stage expects:
AudioBuffer(
    data: np.ndarray,      # shape: (samples, 1), float32, range: [-1, 1]
    sample_rate: int,      # always 44100
    metadata: AudioMetadata # for downstream decisions
)
```

### Usage in Pipeline
```python
class AudioPipeline:
    def __init__(self):
        self.ingest = AudioIngest(target_sample_rate=44100)
        self.separator = Separator(...)
        # ...
        
    def process(self, file_path: Path):
        # Stage 1: Ingest
        buffer = self.ingest.load(file_path)
        
        # Use metadata for decisions
        if buffer.metadata.estimated_snr < 10:
            print("Warning: Very noisy input detected")
            
        if buffer.metadata.spectral_centroid < 2000:
            print("Phone recording detected - frequency restoration will be applied")
            
        # Stage 2: Separation
        stems = self.separator.separate(buffer)
        # ...
```

---

## Future Enhancements (v2)

1. **Streaming for long files**: Process in chunks to handle >30 min recordings
2. **Multi-file batch processing**: Load and queue multiple recordings
3. **Format conversion**: Built-in ffmpeg wrapper for unsupported formats
4. **Detailed spectral analysis**: ML-based audio quality prediction
5. **Pre-conditioning**: Automatic pre-enhancement before separation (if needed)

---

## Dependencies
```python
# requirements.txt (for this module)
soundfile>=0.12.0      # Primary audio loading
librosa>=0.10.0        # Fallback + resampling
numpy>=1.24.0          # Array operations
```

## File Location
```
resonate/
└── audio_engine/
    └── ingest.py      # This module
```
