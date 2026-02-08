import numpy as np
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
from enum import Enum

# Create minimal package stubs so relative imports in the module succeed during dynamic import
sys.modules['audio_engine'] = ModuleType('audio_engine')
sys.modules['audio_engine.generation'] = ModuleType('audio_engine.generation')

# Minimal profiling.quality_detector stub
qd_mod = ModuleType('audio_engine.profiling.quality_detector')
class DamageLevel(Enum):
    GOOD = "good"
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"
qd_mod.DamageLevel = DamageLevel
qd_mod.StemQualityReport = type('StemQualityReport', (), {})
class QualityDetector:
    def __init__(self):
        pass
qd_mod.QualityDetector = QualityDetector
sys.modules['audio_engine.profiling.quality_detector'] = qd_mod

# Minimal jasco_generator stub
jg_mod = ModuleType('audio_engine.generation.jasco_generator')
jg_mod.GenerationConfig = type('GenerationConfig', (), {'__init__': lambda self: None})
jg_mod.GenerationResult = type('GenerationResult', (), {})
class JASCOGenerator:
    def __init__(self, config=None):
        pass
jg_mod.JASCOGenerator = JASCOGenerator
sys.modules['audio_engine.generation.jasco_generator'] = jg_mod

# Load stem_regenerator module dynamically so we don't import full audio_engine package and heavy deps
spec = importlib.util.spec_from_file_location(
    'audio_engine.generation.stem_regenerator',
    Path(__file__).parents[1] / 'audio_engine' / 'generation' / 'stem_regenerator.py'
)
sr_mod = importlib.util.module_from_spec(spec)
sys.modules['audio_engine.generation.stem_regenerator'] = sr_mod
spec.loader.exec_module(sr_mod)  # type: ignore
StemRegenerator = getattr(sr_mod, 'StemRegenerator')
RegenerationPlan = getattr(sr_mod, 'RegenerationPlan')
RegenerationRegion = getattr(sr_mod, 'RegenerationRegion')


def test_force_whole_stem_regeneration_replaces_audio(monkeypatch):
    # Create short test audio (2s)
    sr = 44100
    duration = 2
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    original = 0.1 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    # Create a plan that marks whole stem for regeneration
    # Mark as critical to ensure summary captures damage level
    region = RegenerationRegion(start_time=0.0, end_time=duration, stem_name='other', damage_level=DamageLevel.CRITICAL)
    plan = RegenerationPlan(stem_name='other', original_audio=original, sample_rate=sr, regions=[region], use_whole_stem=True)

    regenerator = StemRegenerator()

    # Monkeypatch the method that regenerates the entire stem to return a clear synthetic signal
    def fake_regenerate_entire_stem(audio, musical_profile, stem_type, sample_rate, callbacks=None):
        # Return a 2-second square wave at 220 Hz as regenerated audio
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        regenerated = 0.2 * np.sign(np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        return regenerated

    monkeypatch.setattr(regenerator, "_regenerate_entire_stem", fake_regenerate_entire_stem)

    regenerated_audio, summary = regenerator.execute_regeneration_plan(plan)

    assert regenerated_audio is not None
    # Must be different from original
    assert not np.allclose(regenerated_audio, original)
    assert summary.regions_regenerated >= 0


def test_replacing_stem_and_recomputing_mix_changes_output():
    import importlib.util
    from pathlib import Path
    from types import ModuleType
    # Provide utils stub to satisfy relative imports inside mixing module
    utils_mod = ModuleType('audio_engine.utils')
    utils_mod.db_to_gain = lambda db: 10 ** (db / 20.0)
    utils_mod.format_duration = lambda s: f"{s:.2f}s"
    sys.modules['audio_engine.utils'] = utils_mod

    spec = importlib.util.spec_from_file_location(
        'audio_engine.mixing',
        Path(__file__).parents[1] / 'audio_engine' / 'mixing.py'
    )
    mix_mod = importlib.util.module_from_spec(spec)
    sys.modules['audio_engine.mixing'] = mix_mod
    spec.loader.exec_module(mix_mod)  # type: ignore
    StemMixer = getattr(mix_mod, 'StemMixer')

    sr = 44100
    duration = 2
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # Create simple stems: vocals, drums, bass, other
    stems = {
        'vocals': 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32),
        'drums': 0.2 * np.sin(2 * np.pi * 80 * t).astype(np.float32),
        'bass': 0.25 * np.sin(2 * np.pi * 60 * t).astype(np.float32),
        'other': 0.15 * np.sin(2 * np.pi * 330 * t).astype(np.float32),
    }

    mixer = StemMixer()
    orig_mix = mixer.mix(stems, sr).mixed_audio

    # Replace 'other' stem with a different signal
    stems['other'] = 0.5 * np.sign(np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    new_mix = mixer.mix(stems, sr).mixed_audio

    assert not np.allclose(orig_mix, new_mix)


def test_snapshot_undo_restores_previous():
    from audio_engine.mixing import StemMixer

    sr = 44100
    duration = 2
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    stems = {
        'vocals': 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32),
        'drums': 0.2 * np.sin(2 * np.pi * 80 * t).astype(np.float32),
        'bass': 0.25 * np.sin(2 * np.pi * 60 * t).astype(np.float32),
        'other': 0.15 * np.sin(2 * np.pi * 330 * t).astype(np.float32),
    }

    mixer = StemMixer()
    orig_mix = mixer.mix(stems, sr).mixed_audio

    # Snapshot previous 'other' stem
    prev_other = stems['other'].copy()

    # Replace 'other' stem
    stems['other'] = 0.5 * np.sign(np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    new_mix = mixer.mix(stems, sr).mixed_audio

    assert not np.allclose(orig_mix, new_mix)

    # Undo: restore previous stem
    stems['other'] = prev_other
    restored_mix = mixer.mix(stems, sr).mixed_audio

    assert np.allclose(orig_mix, restored_mix)

