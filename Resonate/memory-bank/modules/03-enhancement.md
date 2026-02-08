# Module 03: Per-Stem Enhancement

## Purpose
Apply tailored audio processing to each isolated stem (vocals, drums, bass, other) to remove noise, clarify frequency content, and restore fidelity. This is where we transform separated-but-still-degraded stems into clean, professional-quality tracks.

## Key Responsibilities
1. Noise reduction per stem (adaptive to content type)
2. Frequency equalization (restore phone-limited bandwidth)
3. Dynamic range processing (compression where appropriate)
4. Harmonic enhancement (add clarity and presence)
5. Artifact detection and mitigation

---

## Design Philosophy

**"Enhance, Don't Transform"**
- Goal: Reveal the true performance, remove degradation
- Avoid: Over-processing that changes musical intent
- Strategy: Conservative defaults, iterative refinement

**Stem-Specific Processing**
Each stem type needs different treatment:
- **Vocals**: Noise reduction + presence boost + de-essing
- **Drums**: Transient preservation + punch restoration
- **Bass**: Low-end clarity + fundamental reinforcement  
- **Other** (instruments): Harmonic exciter + mid-range clarity

---

## Core Architecture

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import numpy as np
from pedalboard import Pedalboard, Compressor, Gain, HighShelfFilter, LowShelfFilter, PeakFilter
import noisereduce as nr

@dataclass
class EnhancementSettings:
    """User-configurable enhancement parameters"""
    noise_reduction_strength: float = 0.5  # 0-1
    eq_intensity: float = 0.7              # 0-1
    compression_amount: float = 0.3        # 0-1
    harmonic_enhancement: float = 0.4      # 0-1
    
class StemEnhancer(ABC):
    """Base class for stem-specific enhancement"""
    
    def __init__(self, sample_rate: int, settings: EnhancementSettings):
        self.sr = sample_rate
        self.settings = settings
        
    @abstractmethod
    def process(self, audio: np.ndarray) -> np.ndarray:
        """Process audio stem, return enhanced version"""
        pass
        
    def detect_artifacts(self, original: np.ndarray, processed: np.ndarray) -> float:
        """
        Detect if processing introduced artifacts
        Returns artifact score: 0 = clean, 1 = severe artifacts
        """
        # Simple heuristic: Check for new high-frequency content
        # (metallic/robotic artifacts often appear at high freqs)
        
        from scipy import signal
        
        # High-pass filter >10kHz
        nyq = self.sr / 2
        b, a = signal.butter(4, 10000 / nyq, 'high')
        
        orig_hf = signal.filtfilt(b, a, original.flatten())
        proc_hf = signal.filtfilt(b, a, processed.flatten())
        
        orig_energy = np.mean(orig_hf**2)
        proc_energy = np.mean(proc_hf**2)
        
        # If high-freq energy increased >10x, likely artifacts
        if proc_energy > orig_energy * 10:
            return 1.0
        elif proc_energy > orig_energy * 3:
            return 0.5
        else:
            return 0.0
            
    def safe_enhance(self, audio: np.ndarray, enhancer_fn, threshold: float = 0.5) -> np.ndarray:
        """
        Apply enhancement with artifact detection rollback
        
        Args:
            audio: Input audio
            enhancer_fn: Function that applies enhancement
            threshold: Artifact score threshold (0-1)
            
        Returns:
            Enhanced audio, or original if artifacts detected
        """
        enhanced = enhancer_fn(audio)
        artifact_score = self.detect_artifacts(audio, enhanced)
        
        if artifact_score > threshold:
            print(f"⚠ Artifacts detected (score: {artifact_score:.2f}), rolling back enhancement")
            return audio
        else:
            return enhanced
```

---

## Vocal Enhancement

```python
class VocalEnhancer(StemEnhancer):
    """Enhance vocal stem from phone recording"""
    
    def process(self, audio: np.ndarray) -> np.ndarray:
        """
        Vocal processing chain:
        1. Noise reduction (careful not to remove breath/texture)
        2. De-essing (reduce harsh 's' sounds)
        3. Presence boost (2-5 kHz for clarity)
        4. Air band lift (10-15 kHz for brilliance)
        5. Gentle compression (smooth dynamics)
        """
        
        # Step 1: Noise reduction
        audio = self._reduce_noise(audio)
        
        # Step 2: De-essing (reduce harsh sibilance)
        audio = self._deess(audio)
        
        # Step 3: EQ for presence and air
        audio = self._eq_vocal(audio)
        
        # Step 4: Gentle compression
        audio = self._compress_vocal(audio)
        
        return audio
        
    def _reduce_noise(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply spectral noise reduction
        
        Strategy:
        - Use noisereduce library (spectral gating)
        - Conservative settings to preserve vocal texture
        - Only reduce stationary noise (crowd rumble, hiss)
        """
        
        # Estimate noise from quietest 10% of audio
        # (assumes some sections without vocals)
        
        if self.settings.noise_reduction_strength < 0.1:
            return audio  # Skip if user disabled
            
        try:
            reduced = nr.reduce_noise(
                y=audio.flatten(),
                sr=self.sr,
                stationary=True,
                prop_decrease=self.settings.noise_reduction_strength,
                freq_mask_smooth_hz=500,  # Smooth frequency
                time_mask_smooth_ms=50    # Smooth time
            )
            
            return reduced.reshape(-1, 1)
            
        except Exception as e:
            print(f"Noise reduction failed: {e}, skipping")
            return audio
            
    def _deess(self, audio: np.ndarray) -> np.ndarray:
        """
        Reduce harsh 's' sounds (5-8 kHz)
        
        Method: Multiband compression on sibilance range
        """
        
        # Create pedalboard chain
        board = Pedalboard([
            # Compress only 5-8 kHz range (where sibilance lives)
            PeakFilter(frequency=6500, gain_db=-3, q=2.0)
        ])
        
        return board(audio.flatten(), self.sr).reshape(-1, 1)
        
    def _eq_vocal(self, audio: np.ndarray) -> np.ndarray:
        """
        EQ curve for vocal clarity
        
        Boost:
        - 3 kHz: Presence (intelligibility)
        - 12 kHz: Air (brilliance)
        
        Cut:
        - 200 Hz: Reduce muddiness
        """
        
        intensity = self.settings.eq_intensity
        
        board = Pedalboard([
            # Cut mud
            PeakFilter(frequency=200, gain_db=-2 * intensity, q=1.0),
            
            # Boost presence
            PeakFilter(frequency=3000, gain_db=4 * intensity, q=1.5),
            
            # Boost air
            HighShelfFilter(frequency=10000, gain_db=3 * intensity)
        ])
        
        return board(audio.flatten(), self.sr).reshape(-1, 1)
        
    def _compress_vocal(self, audio: np.ndarray) -> np.ndarray:
        """
        Gentle vocal compression
        
        Purpose: Even out dynamics without squashing
        Ratio: 2:1 (subtle)
        Threshold: -18 dB
        Attack: 5 ms (catch peaks)
        Release: 50 ms (natural)
        """
        
        if self.settings.compression_amount < 0.1:
            return audio  # Skip if user disabled
            
        board = Pedalboard([
            Compressor(
                threshold_db=-18,
                ratio=1 + (self.settings.compression_amount * 3),  # 1:1 to 4:1
                attack_ms=5,
                release_ms=50
            )
        ])
        
        return board(audio.flatten(), self.sr).reshape(-1, 1)
```

---

## Drum Enhancement

```python
class DrumEnhancer(StemEnhancer):
    """Enhance drum stem - preserve transients!"""
    
    def process(self, audio: np.ndarray) -> np.ndarray:
        """
        Drum processing chain:
        1. Transient preservation (critical!)
        2. Punch restoration (80-200 Hz for kick)
        3. Clarity boost (2-5 kHz for snare)
        4. Minimal compression (preserve dynamics)
        """
        
        # Step 1: EQ for punch and clarity
        audio = self._eq_drums(audio)
        
        # Step 2: Light compression (only if needed)
        if self.settings.compression_amount > 0.5:  # Only for heavy compression setting
            audio = self._compress_drums(audio)
            
        return audio
        
    def _eq_drums(self, audio: np.ndarray) -> np.ndarray:
        """
        EQ curve for drum impact
        
        Boost:
        - 100 Hz: Kick fundamental
        - 3 kHz: Snare presence
        - 10 kHz: Cymbals air
        
        Cut:
        - 400 Hz: Reduce boxiness
        """
        
        intensity = self.settings.eq_intensity
        
        board = Pedalboard([
            # Boost kick
            PeakFilter(frequency=100, gain_db=3 * intensity, q=1.5),
            
            # Cut boxiness
            PeakFilter(frequency=400, gain_db=-2 * intensity, q=1.0),
            
            # Boost snare presence
            PeakFilter(frequency=3000, gain_db=4 * intensity, q=2.0),
            
            # Boost cymbals
            HighShelfFilter(frequency=8000, gain_db=2 * intensity)
        ])
        
        return board(audio.flatten(), self.sr).reshape(-1, 1)
        
    def _compress_drums(self, audio: np.ndarray) -> np.ndarray:
        """
        Very light compression - preserve natural dynamics
        
        Drums NEED dynamics to sound alive
        Only apply gentle compression for extreme cases
        """
        
        board = Pedalboard([
            Compressor(
                threshold_db=-12,
                ratio=2.0,  # Gentle 2:1
                attack_ms=1,   # Fast attack (preserve transients)
                release_ms=100  # Medium release
            )
        ])
        
        return board(audio.flatten(), self.sr).reshape(-1, 1)
```

---

## Bass Enhancement

```python
class BassEnhancer(StemEnhancer):
    """Enhance bass stem - focus on low-end clarity"""
    
    def process(self, audio: np.ndarray) -> np.ndarray:
        """
        Bass processing chain:
        1. Sub-bass restoration (40-80 Hz)
        2. Fundamental clarity (80-200 Hz)
        3. Gentle compression (control dynamics)
        4. Harmonic enhancement (add presence)
        """
        
        # Step 1: Clean up ultra-lows (rumble)
        audio = self._highpass_filter(audio)
        
        # Step 2: EQ for clarity
        audio = self._eq_bass(audio)
        
        # Step 3: Compression
        audio = self._compress_bass(audio)
        
        # Step 4: Harmonic enhancement
        if self.settings.harmonic_enhancement > 0.3:
            audio = self._enhance_harmonics(audio)
            
        return audio
        
    def _highpass_filter(self, audio: np.ndarray) -> np.ndarray:
        """Remove rumble below 30 Hz"""
        from scipy import signal
        
        nyq = self.sr / 2
        b, a = signal.butter(4, 30 / nyq, 'high')
        filtered = signal.filtfilt(b, a, audio.flatten())
        
        return filtered.reshape(-1, 1)
        
    def _eq_bass(self, audio: np.ndarray) -> np.ndarray:
        """
        EQ for bass definition
        
        Boost:
        - 60 Hz: Sub-bass weight
        - 100 Hz: Fundamental (where bass lives)
        
        Cut:
        - 300 Hz: Reduce muddiness
        """
        
        intensity = self.settings.eq_intensity
        
        board = Pedalboard([
            # Boost sub-bass
            PeakFilter(frequency=60, gain_db=3 * intensity, q=1.0),
            
            # Boost fundamental
            PeakFilter(frequency=100, gain_db=4 * intensity, q=1.5),
            
            # Cut mud
            PeakFilter(frequency=300, gain_db=-3 * intensity, q=1.0)
        ])
        
        return board(audio.flatten(), self.sr).reshape(-1, 1)
        
    def _compress_bass(self, audio: np.ndarray) -> np.ndarray:
        """
        Bass compression for consistency
        
        Bass benefits from more compression than drums
        Goal: Solid, consistent foundation
        """
        
        amount = self.settings.compression_amount
        
        board = Pedalboard([
            Compressor(
                threshold_db=-20,
                ratio=2 + (amount * 2),  # 2:1 to 4:1
                attack_ms=10,
                release_ms=100
            )
        ])
        
        return board(audio.flatten(), self.sr).reshape(-1, 1)
        
    def _enhance_harmonics(self, audio: np.ndarray) -> np.ndarray:
        """
        Add harmonic content for presence on small speakers
        
        Strategy: Generate 2nd and 3rd harmonics subtly
        Helps bass be heard on phone/laptop speakers (which can't reproduce low bass)
        """
        
        # Simple harmonic generation: soft clipping
        # Creates harmonic distortion (controlled)
        
        strength = self.settings.harmonic_enhancement
        
        # Soft clip: tanh(x * gain) / gain
        # Creates gentle harmonics without harsh distortion
        gain = 1 + (strength * 2)
        enhanced = np.tanh(audio * gain) / gain
        
        # Blend with original (parallel processing)
        blend = 0.3 * strength  # Up to 30% blend
        result = audio * (1 - blend) + enhanced * blend
        
        return result
```

---

## Instrument Enhancement (Other)

```python
class InstrumentEnhancer(StemEnhancer):
    """Enhance 'other' stem (guitars, keys, horns, etc.)"""
    
    def process(self, audio: np.ndarray) -> np.ndarray:
        """
        Instrument processing chain:
        1. Noise reduction (moderate)
        2. Mid-range clarity (500-2000 Hz)
        3. Harmonic enhancement (add shimmer)
        4. Moderate compression
        """
        
        # Step 1: Noise reduction
        audio = self._reduce_noise(audio)
        
        # Step 2: EQ for clarity
        audio = self._eq_instruments(audio)
        
        # Step 3: Harmonic exciter
        if self.settings.harmonic_enhancement > 0.3:
            audio = self._exciter(audio)
            
        # Step 4: Compression
        audio = self._compress_instruments(audio)
        
        return audio
        
    def _reduce_noise(self, audio: np.ndarray) -> np.ndarray:
        """Similar to vocal noise reduction"""
        
        if self.settings.noise_reduction_strength < 0.1:
            return audio
            
        try:
            reduced = nr.reduce_noise(
                y=audio.flatten(),
                sr=self.sr,
                stationary=True,
                prop_decrease=self.settings.noise_reduction_strength * 0.8,  # Slightly less aggressive
                freq_mask_smooth_hz=500,
                time_mask_smooth_ms=50
            )
            
            return reduced.reshape(-1, 1)
            
        except Exception as e:
            print(f"Noise reduction failed: {e}")
            return audio
            
    def _eq_instruments(self, audio: np.ndarray) -> np.ndarray:
        """
        EQ for instrument clarity
        
        Goal: Unmuddy the midrange where guitars/keys live
        """
        
        intensity = self.settings.eq_intensity
        
        board = Pedalboard([
            # Cut mud
            PeakFilter(frequency=250, gain_db=-2 * intensity, q=1.0),
            
            # Boost presence
            PeakFilter(frequency=1500, gain_db=3 * intensity, q=1.5),
            
            # Boost air/shimmer
            HighShelfFilter(frequency=8000, gain_db=3 * intensity)
        ])
        
        return board(audio.flatten(), self.sr).reshape(-1, 1)
        
    def _exciter(self, audio: np.ndarray) -> np.ndarray:
        """
        Harmonic exciter: Add high-frequency harmonics
        
        Makes instruments sound more "alive" and "present"
        """
        
        strength = self.settings.harmonic_enhancement
        
        # Generate harmonics via soft saturation
        gain = 1 + strength
        excited = np.tanh(audio * gain) / gain
        
        # High-pass the excited signal (only add highs)
        from scipy import signal
        nyq = self.sr / 2
        b, a = signal.butter(2, 3000 / nyq, 'high')
        excited_hf = signal.filtfilt(b, a, excited.flatten()).reshape(-1, 1)
        
        # Blend with original
        blend = 0.2 * strength
        result = audio + excited_hf * blend
        
        return result
        
    def _compress_instruments(self, audio: np.ndarray) -> np.ndarray:
        """Moderate compression for instruments"""
        
        amount = self.settings.compression_amount
        
        board = Pedalboard([
            Compressor(
                threshold_db=-15,
                ratio=2 + amount,  # 2:1 to 3:1
                attack_ms=5,
                release_ms=75
            )
        ])
        
        return board(audio.flatten(), self.sr).reshape(-1, 1)
```

---

## Orchestrator

```python
class StemEnhancementPipeline:
    """Coordinate enhancement of all stems"""
    
    def __init__(self, sample_rate: int, settings: EnhancementSettings):
        self.sr = sample_rate
        self.settings = settings
        
        # Create enhancers
        self.vocal_enhancer = VocalEnhancer(sample_rate, settings)
        self.drum_enhancer = DrumEnhancer(sample_rate, settings)
        self.bass_enhancer = BassEnhancer(sample_rate, settings)
        self.instrument_enhancer = InstrumentEnhancer(sample_rate, settings)
        
    def process(self, stems: SeparatedStems) -> SeparatedStems:
        """
        Enhance all stems in parallel (if possible)
        
        Returns new SeparatedStems with enhanced audio
        """
        
        print("Enhancing vocals...")
        enhanced_vocals = self.vocal_enhancer.process(stems.vocals)
        
        print("Enhancing drums...")
        enhanced_drums = self.drum_enhancer.process(stems.drums)
        
        print("Enhancing bass...")
        enhanced_bass = self.bass_enhancer.process(stems.bass)
        
        print("Enhancing instruments...")
        enhanced_other = self.instrument_enhancer.process(stems.other)
        
        return SeparatedStems(
            vocals=enhanced_vocals,
            drums=enhanced_drums,
            bass=enhanced_bass,
            other=enhanced_other,
            sample_rate=stems.sample_rate,
            quality_mode=stems.quality_mode
        )
```

---

## Testing Strategy

```python
def test_vocal_enhancement_improves_snr():
    """Test that vocal enhancement improves SNR"""
    noisy_vocal = load_fixture("noisy_vocal.wav")
    enhancer = VocalEnhancer(44100, EnhancementSettings())
    
    enhanced = enhancer.process(noisy_vocal)
    
    # SNR should improve
    orig_snr = estimate_snr(noisy_vocal)
    enhanced_snr = estimate_snr(enhanced)
    
    assert enhanced_snr > orig_snr
    
def test_artifact_detection_works():
    """Test artifact detection catches over-processing"""
    clean = np.random.randn(44100, 1).astype(np.float32) * 0.1
    
    # Create artifact: add high-freq noise
    artifact = clean + np.random.randn(44100, 1).astype(np.float32) * 0.5
    
    enhancer = VocalEnhancer(44100, EnhancementSettings())
    score = enhancer.detect_artifacts(clean, artifact)
    
    assert score > 0.5  # Should detect artifacts
```

---

## Performance Notes

- **Processing Time**: ~2-5 seconds per stem per 3-minute song
- **Memory**: Minimal (processes in-place where possible)
- **Parallelization**: Can process stems concurrently (4x speedup)

---

## Dependencies
```
pedalboard>=0.7.0      # DSP effects
noisereduce>=2.0.0     # Noise reduction
scipy>=1.10.0          # Filtering
numpy>=1.24.0
```

## File Location
```
resonate/
└── audio_engine/
    └── enhancers/
        ├── __init__.py
        ├── base.py          # StemEnhancer base class
        ├── vocal.py         # VocalEnhancer
        ├── drums.py         # DrumEnhancer
        ├── bass.py          # BassEnhancer
        ├── instruments.py   # InstrumentEnhancer
        └── pipeline.py      # StemEnhancementPipeline
```
