# Resonate: AI Studio Remastering Engine

Resonate is a professional-grade audio remastering platform that uses advanced AI to transform rough recordings (like phone memos or low-quality demos) into studio-quality productions.

## 🚀 Quick Start

```bash
# Activate environment
source venv/bin/activate

# Run the AI Remastering Tool
python tools/run_studio_ai_remaster_v2.py "input_song.mp3" --genre "future_rave,tech_house"

# Run the UI
streamlit run ui/app.py
```

## 🧠 The Technology

Resonate combines three cutting-edge audio technologies into a single pipeline:

### 1. Source Separation (Demucs)
We use **HTDemucs-ft** (Hybrid Transformer Demucs fine-tuned), a state-of-the-art neural network from Meta Research, to surgically split your audio into four stems:
- Vocals
- Drums
- Bass
- Other Instruments

This allows us to process each element of the music independently.

### 2. Generative AI "Inpainting" (MusicGen/JASCO)
To fix damaged or low-quality audio, we don't just "filter" it—we **regenerate** it.
- **Model**: We use Meta's **MusicGen** (specifically the large/medium variants) wrapped in our custom **JASCO** (Joint Audio and Symbolic Conditioning) pipeline.
- **Conditioned Generation**: Instead of generating random music, we analyze your original stem to extract its "DNA" (melody contour, rhythm pattern, key, tempo).
- **Style Transfer**: We prompt the AI with high-end descriptors (e.g., "studio quality", "future rave", "crisp vocals") while forcing it to follow your song's original structure. This effectively "re-performs" your track with virtual studio instruments.

### 3. DSP Mastering
The final stage uses traditional Digital Signal Processing to glue everything together:
- **Reference Matching**: Matches the frequency balance of professional reference tracks.
- **Multiband Compression**: Ensures tightness and punch.
- **Loudness Normalization**: Targets standard streaming levels (-14 LUFS).

## 🛠️ Features

- **Genre-Aware Remastering**: Specify styles like `future_rave`, `tech_house`, `dance` to guide the AI's sound choices.
- **Conditioned Chunking**: Processes audio in overlapping 30-second windows with crossfading to bypass AI memory limits while maintaining musical continuity.
- **Vocal Boost**: Intelligently balances vocals against the mix using reference track analysis.
- **Pre-Conditioning**: Cleans input audio (noise reduction, de-clipping) *before* it hits the AI to ensure the best possible separation.
- **Taylor's Version Mode**: Full AI rerecording that recreates the song 1:1 with virtual studio instruments (100% AI generation).

## 📦 Installation

```bash
# Clone the repo
git clone https://github.com/your-repo/Resonate.git
cd Resonate

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 🎛️ Advanced Usage

**Remaster a track with specific genre blend:**
```bash
python tools/run_studio_ai_remaster_v2.py "my_demo.mp3" \
  --genre "future_rave,tech_house" \
  --vocal-boost 1.2 \
  --blend 0.4
```

**Use a specific reference track for mixing:**
```bash
python tools/run_studio_ai_remaster_v2.py "my_demo.mp3" \
  --reference "reference_track.mp3"
```

**Run Taylor's Version Mode (Full AI Rerecording):**
```bash
python tools/run_taylors_version_rerecord.py "my_demo.mp3" \
  --genre "tech_house,future_rave" \
  --similarity-target 0.85
```
