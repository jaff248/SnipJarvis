# Resonate: Live Music Recording Reconstruction - Project Brief

## Core Mission
**Reconstruct a lost musical performance from a phone recording captured in a crowd — transforming it to approach studio recording quality.**

This is audio forensics meets music restoration meets intelligent enhancement.

**Tagline**: "Resurrect the performance that was lost"

## The Problem
A friend's original studio/source recording was lost. All that remains is a smartphone recording captured from the audience during a live performance. This recording suffers from:
- Crowd noise, chatter, movement sounds
- Room acoustics/reverb from the venue
- Phone microphone limitations (poor frequency response, AGC compression)
- Distance from the source (indirect sound capture)
- Possible clipping/distortion from loud peaks

## Primary Use Case
```
INPUT:  Phone recording from crowd at live music performance
        - Muffled musical content buried under ambient noise
        - Limited frequency response (phone mic: ~100Hz-8kHz effective)
        - Reverberant, distant sound character
        
OUTPUT: Reconstructed studio-approaching track
        - Clean, separated stems (vocals, drums, bass, instruments)
        - Restored full frequency response (20Hz-20kHz)
        - Removed crowd/room artifacts  
        - Professional dynamics, loudness, clarity
```

## Hardware Context
- **Development Machine**: Apple M1 Max with 64GB RAM
- **GPU Acceleration**: MPS (Metal Performance Shaders) for PyTorch
- **Processing Philosophy**: Quality over speed — willing to wait hours for best results

## Key Technical Challenges

### 1. Source Separation in Hostile Conditions
Demucs trained on clean studio audio must handle phone recordings with severe degradation.
**Approach**: Pre-conditioning → separation → multi-pass refinement

### 2. Frequency Restoration  
Phone mics roll off bass (<100Hz) and treble (>8kHz). The original had full spectrum.
**Approach**: Spectral modeling → harmonic extension → intelligent high-frequency synthesis

### 3. De-Reverberation
Venue acoustics are baked in. Need to extract "dry" performance from reverberant capture.
**Approach**: Blind dereverberation, room impulse response estimation

### 4. Crowd Noise Isolation
Separate musical content from crowd ambience, clapping, talking, movement.
**Approach**: Spectral gating + source separation + transient analysis

## Success Criteria

| Aspect | Target | Measurement |
|--------|--------|-------------|
| Noise Reduction | ≥20 dB SNR improvement | SNR calculation |
| Vocal Clarity | Clear intelligibility | A/B blind test |
| Instrument Definition | Individually audible parts | Stem isolation quality |
| Frequency Extension | Full 20Hz-20kHz | Spectral analysis |
| Artifacts | No audible "musical noise" | Listening test |
| Overall Quality | "Could be from the soundboard" | Blind comparison |

## What This IS vs. IS NOT

### IS:
- ✅ Forensic music recovery tool
- ✅ Maximum quality extraction from degraded live recordings
- ✅ Multi-stage intelligent processing pipeline
- ✅ Rescue tool for irreplaceable performances

### IS NOT:
- ❌ AI music generation (no hallucination of new content)
- ❌ Real-time processor
- ❌ General-purpose DAW
- ❌ Perfect recreation (impossible, but approaching it)

## Philosophy
**Reveal, don't fabricate.** We enhance what's captured, extend frequencies that were filtered, separate sources that were mixed — but we don't invent musical content that wasn't performed.
