import numpy as np
import importlib.util
from pathlib import Path

# Load module directly to avoid importing heavy top-level audio_engine package
spec = importlib.util.spec_from_file_location(
    'quality_detector',
    Path(__file__).parents[1] / 'audio_engine' / 'profiling' / 'quality_detector.py'
)
qd_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qd_mod)  # type: ignore
QualityDetector = getattr(qd_mod, 'QualityDetector')


def test_score_distortion_silent_returns_zero():
    qd = QualityDetector()
    silence = np.zeros(44100, dtype=np.float32)
    score = qd._score_distortion(silence, 44100)
    assert score == 0.0


def test_score_distortion_harmonic_is_reasonable():
    qd = QualityDetector()
    t = np.linspace(0, 1, 44100, endpoint=False)
    harmonic = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    score = qd._score_distortion(harmonic, 44100)
    assert 0.0 <= score <= 1.0
    # Harmonic content alone should not be treated as maximum distortion
    assert score < 0.9
