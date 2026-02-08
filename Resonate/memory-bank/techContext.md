# Resonate: Technology Context

## Core Technology Stack

### ML/Audio Processing
| Component | Library | Version | Purpose |
|-----------|---------|---------|---------|
| Source Separation | **Demucs** (htdemucs_ft) | 4.0+ | Isolate vocals/instruments |
| Noise Reduction | **noisereduce** | 2.0+ | Spectral gating |
| Audio Effects | **pedalboard** | 0.7+ | EQ, compression, effects |
| Audio I/O | **soundfile** | 0.12+ | WAV/FLAC read/write |
| Feature Extraction | **librosa** | 0.10+ | Analysis, spectrograms |
| Loudness | **pyloudnorm** | 0.4+ | LUFS normalization |

### ML Framework
| Component | Library | Notes |
|-----------|---------|-------|
| Deep Learning | **PyTorch** | MPS-enabled for M1/M2 |
| Tensor Audio | **torchaudio** | Audio transforms |

### Application
| Component | Library | Purpose |
|-----------|---------|---------|
| UI Framework | **Streamlit** | Simple web interface |
| Visualization | **plotly** | Interactive waveforms/spectrograms |

## Hardware Optimization (M1 Max 64GB)

### MPS (Metal Performance Shaders) Configuration
```python
import torch

def get_device():
    if torch.backends.mps.is_available():
        # Test MPS with actual operation
        try:
            test = torch.ones(1, device='mps') * 2
            return torch.device('mps')
        except Exception:
            pass
    return torch.device('cpu')

# Memory configuration (correct API)
torch.mps.set_per_process_memory_fraction(0.75)  # Leave room for system
```

### Known MPS Limitations (Critical!)
1. **Complex Numbers**: MPS doesn't fully support complex tensors
   - Demucs handles this internally (falls back to CPU for complex ops)
   - See: demucs issue #435, #432
2. **Memory Spikes**: Large audio files can cause OOM
   - Solution: Process in segments (10-30 second chunks)
3. **Precision**: Some operations need float32 attention

## Key Dependencies Deep Dive

### Demucs v4 (HTDemucs)
- **Architecture**: Hybrid Transformer (spectrogram + waveform branches)
- **Models Available**:
  - `htdemucs`: Default, fast, good quality
  - `htdemucs_ft`: Fine-tuned, 4x slower, best quality
  - `htdemucs_6s`: 6 sources (adds piano, guitar — experimental)
- **API Usage**:
```python
from demucs.api import Separator

separator = Separator(model="htdemucs_ft", device="mps")
origin, separated = separator.separate_audio_file("input.wav")
# separated = {'vocals': tensor, 'drums': tensor, 'bass': tensor, 'other': tensor}
```

### AudioCraft (Meta) - Evaluated but NOT Primary
- **EnCodec**: Neural audio codec — could be useful for bandwidth restoration
- **MultiBandDiffusion**: Decode tokens to high-quality audio
- **NOT Using**: MusicGen/AudioGen (generative, would hallucinate)

### Why NOT VoiceFixer (from Original PRD)
- Unmaintained, dependency conflicts
- Better alternatives exist (resemble-enhance, SpeechBrain)
- Can introduce uncanny artifacts

## Installation Requirements

### macOS (Apple Silicon)
```bash
# PyTorch with MPS
pip install torch torchaudio

# Demucs 
pip install demucs>=4.0.0

# Audio processing
pip install librosa soundfile noisereduce pedalboard pyloudnorm

# App
pip install streamlit plotly
```

### Verified Compatible Versions (M1/M2)
```
torch>=2.1.0
torchaudio>=2.1.0
demucs>=4.0.0
librosa>=0.10.1
soundfile>=0.12.1
noisereduce>=2.0.0
pedalboard>=0.7.0
pyloudnorm>=0.4.0
streamlit>=1.28.0
```

## Technical Constraints

1. **Sample Rate**: Process at native rate, resample only for models that require it
2. **Bit Depth**: Internal processing at float32, export at 24-bit or 32-bit float  
3. **Mono/Stereo**: Preserve original channel count, check mono compatibility
4. **File Formats**: Input: WAV, MP3, FLAC, OGG; Output: WAV, MP3 (via lameenc)
5. **Max Duration**: Warn at > 10 min, block at > 30 min (memory constraints)
