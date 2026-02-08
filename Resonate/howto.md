# Basic (uses defaults)
cd Resonate && python tools/run_studio_ai_remaster.py "input.mp3"

# With reference track for EQ matching
python tools/run_studio_ai_remaster.py "input.mp3" -r "reference.mp3"

# Conservative AI (less AI influence)
python tools/run_studio_ai_remaster.py "input.mp3" --blend 0.2 --regen-threshold 0.7

# Force CPU for Demucs (more stable)
python tools/run_studio_ai_remaster.py "input.mp3" --force-cpu

# Full options
python tools/run_studio_ai_remaster.py "input.mp3" \
  --reference "ref.mp3" \
  --precond-strength 0.3 \
  --demucs-model htdemucs_ft \
  --jasco-model medium \
  --blend 0.3 \
  --target-lufs -14 \
  --force-cpu
