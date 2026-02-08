import numpy as np
from types import SimpleNamespace

import importlib.util
import sys
from pathlib import Path

# Import the module directly by path to avoid importing package-level heavy dependencies
spec = importlib.util.spec_from_file_location(
    "regeneration_utils",
    Path(__file__).resolve().parents[1] / "audio_engine" / "generation" / "regeneration_utils.py"
)
regen_utils = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = regen_utils
spec.loader.exec_module(regen_utils)


def make_sine(freq, sr, duration, amp=0.3):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return amp * np.sin(2 * np.pi * freq * t)


def test_remaster_project_runs_and_repairs(monkeypatch):
    sample_rate = 22050
    duration = 2.0

    # Create stems
    stems = {
        'vocals': make_sine(440, sample_rate, duration),
        'drums': make_sine(60, sample_rate, duration),
        'bass': make_sine(110, sample_rate, duration),
        'other': make_sine(330, sample_rate, duration),
    }

    # Introduce an obviously "damaged" stem by adding clipping/distortion
    distorted = stems['vocals'].copy()
    distorted[100:200] = 1.0  # hard clip region
    stems['vocals'] = distorted

    # Stub out auto_repair_all_stems to simulate fixing the vocals
    def fake_auto_repair_all_stems(stems_subset, sample_rate_arg, **kwargs):
        # Simulate that vocals were processed and slightly smoothed
        repaired = {}
        for name, audio in stems_subset.items():
            repaired[name] = np.clip(audio * 0.9, -0.9, 0.9)
        return {'stems': repaired, 'snapshots': {k: v.copy() for k, v in stems_subset.items()}, 'results': {'processed': [{'stem': k} for k in stems_subset.keys()]}}

    monkeypatch.setattr(regen_utils, 'auto_repair_all_stems', fake_auto_repair_all_stems)

    # Monkeypatch AudioMaster to avoid file IO and heavy ops
    class FakeMaster:
        def __init__(self, config=None):
            self.config = config

        def master(self, audio, sample_rate, output_path=None):
            # Return a simple namespace mimicking MasteringResult
            return SimpleNamespace(
                audio=audio,
                sample_rate=sample_rate,
                format='wav',
                bit_depth=24,
                file_path=output_path or 'fake_master.wav',
                file_size_bytes=len(audio),
                loudness_lufs=-14.0,
                true_peak_db=-1.0,
                dynamic_range_db=6.0,
            )


    # Ensure dependent modules are loadable under audio_engine.* names so relative imports succeed
    import importlib.util as _spec
    base = Path(__file__).resolve().parents[1]
    # Create a lightweight package placeholder for "audio_engine" so we can
    # import submodules directly without running audio_engine/__init__.py which
    # brings heavy dependencies.
    import types
    pkg = types.ModuleType('audio_engine')
    pkg.__path__ = [str(base / 'audio_engine')]
    sys.modules['audio_engine'] = pkg

    # Load profiling.quality_detector directly into package namespace (or stub it)
    spec_q = _spec.spec_from_file_location('audio_engine.profiling.quality_detector', base / 'audio_engine' / 'profiling' / 'quality_detector.py')
    qmod = _spec.module_from_spec(spec_q)
    sys.modules[spec_q.name] = qmod
    spec_q.loader.exec_module(qmod)

    # Monkeypatch the QualityDetector to avoid librosa dependency and to mark
    # our clipped "vocals" stem as needing regeneration.
    class FakeQualityDetector:
        def analyze(self, audio, sample_rate, *args, **kwargs):
            max_amp = float(np.max(np.abs(audio)))
            needs = max_amp > 0.95
            report = SimpleNamespace(
                needs_regeneration=needs,
                distortion_score=(0.5 if needs else 0.0),
                to_dict=lambda: {
                    'needs_regeneration': needs,
                    'distortion_score': (0.5 if needs else 0.0)
                }
            )
            return report

        def should_regenerate(self, report):
            return report.needs_regeneration

    qmod.QualityDetector = FakeQualityDetector
    sys.modules['audio_engine.profiling.quality_detector'] = qmod

    # Create a lightweight stub for audio_engine.utils to avoid heavy deps (torch/librosa)
    utils_stub = types.ModuleType('audio_engine.utils')
    utils_stub.db_to_gain = lambda db: 2 ** (db / 6)
    utils_stub.format_duration = lambda s: f"{int(s//60)}:{int(s%60):02d}"
    sys.modules['audio_engine.utils'] = utils_stub

    # Load mixing (it will import .utils relative to our fake package)
    spec_m = _spec.spec_from_file_location('audio_engine.mixing', base / 'audio_engine' / 'mixing.py')
    mmod = _spec.module_from_spec(spec_m)
    sys.modules[spec_m.name] = mmod
    spec_m.loader.exec_module(mmod)

    # Inject a lightweight mastering module to avoid soundfile dependency
    mastering_stub = types.ModuleType('audio_engine.mastering')
    mastering_stub.AudioMaster = FakeMaster
    mastering_stub.MasteringConfig = SimpleNamespace
    sys.modules['audio_engine.mastering'] = mastering_stub

    # Set package name so relative imports inside the module succeed
    regen_utils.__package__ = 'audio_engine.generation'

    # Run remaster_project with default settings (use_ai_if_needed=True)
    out = regen_utils.remaster_project(stems, sample_rate, use_ai_if_needed=True, master_class=FakeMaster)

    assert 'reports' in out
    assert 'stems_repaired' in out
    assert 'mix_result' in out
    assert 'mastering' in out

    # Ensure the damaged stem was selected for repair
    assert 'vocals' in out['stems_repaired']

    # Check mix_result contains mixed_audio
    mix = out['mix_result']
    assert hasattr(mix, 'mixed_audio')
    assert mix.sample_rate == sample_rate

    # Check mastering output was returned
    master = out['mastering']
    assert master.file_path.endswith('.wav') or master.file_path == 'fake_master.wav'