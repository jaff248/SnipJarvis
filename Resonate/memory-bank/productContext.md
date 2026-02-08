# Resonate: Product Context - Live Music Reconstruction

## Why This Exists

### The Specific Problem
Your friend played an incredible live performance. The original recording was lost.
All that survives is a phone recording captured from the crowd — muffled, noisy, reverberant.

**The goal**: Reconstruct this phone capture to approach what a studio recording would have sounded like.

### Why Current Tools Fail

| Tool | Problem for Live Music Reconstruction |
|------|---------------------------------------|
| **iZotope RX** | Designed for post-production cleanup, not reconstruction. $399+, steep learning curve |
| **Adobe Podcast** | Speech-focused, destroys music character |
| **Audacity + noisereduce** | Too basic, creates artifacts, no intelligence |
| **Demucs alone** | Separation only, no enhancement or reconstruction |
| **AI Music Tools** | Generate new content — we want to restore existing |

### The Opportunity
Combine:
- **ML Source Separation** (Demucs) — isolate musical elements from crowd noise
- **Spectral Restoration** — recover lost frequencies
- **Intelligent DSP** — enhance without artifacts
- **Multi-stage Pipeline** — iterative refinement

= **Forensic-grade live music reconstruction**

## User Journey

### Primary Persona: "The Archivist"
- Has irreplaceable phone recording of a live performance
- The original/studio version is lost forever
- Willing to invest time for maximum quality
- Values authenticity — wants the real performance, cleaned up

### Workflow
```
1. ANALYZE  → Upload phone recording, see what we're working with
             (Spectrogram reveals: frequency holes, noise floor, reverb tail)

2. SEPARATE → Demucs isolates: Vocals | Drums | Bass | Other
             (Preview each stem to verify separation quality)

3. ENHANCE  → Per-stem processing with controls:
             - Noise reduction strength
             - Frequency restoration amount  
             - De-reverberation level
             - Clarity/presence boost

4. RECONSTRUCT → Advanced options for power users:
             - Harmonic extension (restore rolled-off highs)
             - Sub-bass synthesis (restore rolled-off lows)
             - Transient shaping (restore attack lost to distance)

5. MIX      → Adjust stem balance, add subtle processing
             (Optional: reference a similar song for tonal matching)

6. MASTER   → Final loudness, dynamics, export
             (Target: streaming-ready -14 LUFS)

7. COMPARE  → A/B toggle: Original phone recording vs. Reconstruction
```

## Success Scenario

### Before
> "This phone recording from the crowd... you can barely tell what song is playing. 
> The vocals are buried, the kick drum is mud, there's someone talking 30 seconds in."

### After  
> "Holy shit. I can hear every word of the lyrics now. The guitar part I never knew 
> existed is actually there. It sounds like it could be from a live album."

## Competitive Landscape

| Capability | iZotope RX | Demucs | SpectraLayers | **Resonate** |
|------------|------------|--------|---------------|--------------|
| Source Separation | Basic | **Excellent** | Good | **Excellent** |
| Noise Reduction | Excellent | None | Good | **Very Good** |
| Frequency Restoration | Manual | None | None | **Automatic** |
| De-reverberation | Good | None | Limited | **Targeted** |
| Crowd Noise Removal | Manual | Partial | Manual | **Automatic** |
| Learning Curve | High | Low | High | **Low** |
| Price | $399+ | Free | $299 | **Free** |
| Music-Optimized | General | Music | General | **Live Music** |

## Technical Philosophy

### "Reveal, Don't Fabricate"
- **DO**: Unmask frequencies that were captured but attenuated
- **DO**: Separate sources that were mixed together
- **DO**: Remove noise that was added by environment
- **DON'T**: Generate musical content that wasn't performed
- **DON'T**: Add synthetic instruments or effects not in original

### Quality Over Speed
- **Preview**: Fast feedback (30 seconds in ~1 minute)
- **Full Render**: Take as long as needed for best quality
- **Acceptable**: 10-30 minutes for a 5-minute song reconstruction
