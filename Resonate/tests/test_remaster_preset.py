import numpy as np
from types import SimpleNamespace
import importlib.util
import sys
from pathlib import Path

# Load module directly
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


def test_remaster_uses_studio_preset(monkeypatch):
    sample_rate = 22050
    duration = 1.0

    stems = {
        'vocals': make_sine(440, sample_rate, duration),
        'drums': make_sine(60, sample_rate, duration),
    }

    # Stub out dependencies (quality detector, mixer, mastering)
    import importlib.util as _spec
    base = Path(__file__).resolve().parents[1]
    import types

    pkg = types.ModuleType('audio_engine')
    pkg.__path__ = [str(base / 'audio_engine')]
    sys.modules['audio_engine'] = pkg

    spec_q = _spec.spec_from_file_location('audio_engine.profiling.quality_detector', base / 'audio_engine' / 'profiling' / 'quality_detector.py')
    qmod = _spec.module_from_spec(spec_q)
    sys.modules[spec_q.name] = qmod
    spec_q.loader.exec_module(qmod)

    # stub utils
    utils_stub = types.ModuleType('audio_engine.utils')
    utils_stub.db_to_gain = lambda db: 2 ** (db / 6)
    utils_stub.format_duration = lambda s: f"{int(s//60)}:{int(s%60):02d}"
    sys.modules['audio_engine.utils'] = utils_stub

    spec_m = _spec.spec_from_file_location('audio_engine.mixing', base / 'audio_engine' / 'mixing.py')
    mmod = _spec.module_from_spec(spec_m)
    sys.modules[spec_m.name] = mmod
    spec_m.loader.exec_module(mmod)

    # Fake QualityDetector that flags nothing
    class FakeQualityDetector:
        def analyze(self, audio, sample_rate, *args, **kwargs):
            return SimpleNamespace(needs_regeneration=False, distortion_score=0.0, to_dict=lambda: {})
        def should_regenerate(self, report):
            return False

    qmod.QualityDetector = FakeQualityDetector

    # Fake master that asserts config
    class CheckingMaster:
        def __init__(self, config=None):
            self.config = config
            assert getattr(self.config, 'master_eq_high_db', 0) >= 1.0
            assert getattr(self.config, 'stereo_width', 1.0) >= 1.0

        def master(self, audio, sample_rate, output_path=None):
            return SimpleNamespace(audio=audio, sample_rate=sample_rate, file_path='ok.wav')

    # Inject mastering stub
    mastering_stub = types.ModuleType('audio_engine.mastering')
    mastering_stub.AudioMaster = CheckingMaster
    mastering_stub.MasteringConfig = SimpleNamespace
    sys.modules['audio_engine.mastering'] = mastering_stub

    regen_utils.__package__ = 'audio_engine.generation'

    out = regen_utils.remaster_project(stems, sample_rate, use_ai_if_needed=False, master_class=CheckingMaster)

    assert 'mastering' in out
    assert out['mastering'].file_path == 'ok.wav'