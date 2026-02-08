import numpy as np
import importlib.util
from pathlib import Path
import sys
from types import ModuleType

# Create minimal stubs for modules used by regeneration_utils so import works in tests
sys.modules['audio_engine'] = ModuleType('audio_engine')
# profiling and jasco generator are not directly needed by the helper tests since we inject a dummy generator

# Load the module with minimal package stubs to satisfy relative imports
# Create a minimal package structure
pkg = ModuleType('audio_engine')
pkg.generation = ModuleType('audio_engine.generation')
sys.modules['audio_engine'] = pkg
sys.modules['audio_engine.generation'] = pkg.generation

spec = importlib.util.spec_from_file_location(
    'audio_engine.generation.regeneration_utils',
    Path(__file__).parents[1] / 'audio_engine' / 'generation' / 'regeneration_utils.py'
)
reg_mod = importlib.util.module_from_spec(spec)
# Make sure nested modules used by regeneration_utils are available (stem_regenerator, blender)
from types import ModuleType as _ModuleType
sys.modules['audio_engine.generation.stem_regenerator'] = _ModuleType('audio_engine.generation.stem_regenerator')
sys.modules['audio_engine.generation.blender'] = _ModuleType('audio_engine.generation.blender')

spec.loader.exec_module(reg_mod)  # type: ignore
auto_repair_all_stems = getattr(reg_mod, 'auto_repair_all_stems')


class DummyGenerator:
    def __init__(self):
        pass

    def create_regeneration_plan(self, audio, stem_name, sample_rate, musical_profile=None):
        # Fake a plan with regions for 'vocals' and no regions for others
        from types import SimpleNamespace
        if stem_name in ('vocals', 'drums'):
            region = SimpleNamespace(start_time=0.0, end_time=1.0)
            plan = SimpleNamespace(regions=[region], quality_report=SimpleNamespace(damage_level=None, confidence=0.8))
        else:
            plan = SimpleNamespace(regions=[], quality_report=SimpleNamespace(damage_level=None, confidence=0.8))
        return plan

    def execute_regeneration_plan(self, plan):
        # Return a simple altered signal and a fake summary
        audio_len = 44100 * int((plan.regions[0].end_time - plan.regions[0].start_time) if plan.regions else 1)
        t = np.linspace(0, audio_len / 44100, audio_len, endpoint=False)
        regenerated = 0.2 * np.sign(np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        summary = {'regions_regenerated': len(plan.regions)}
        return regenerated, summary


def test_auto_repair_all_stems_skips_and_processes():
    sr = 44100
    duration = 2
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    stems = {
        'vocals': 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32),
        'drums': 0.2 * np.sin(2 * np.pi * 80 * t).astype(np.float32),
        'bass': 0.25 * np.sin(2 * np.pi * 60 * t).astype(np.float32),
        'other': 0.15 * np.sin(2 * np.pi * 330 * t).astype(np.float32),
    }

    results = auto_repair_all_stems(stems.copy(), sr, regenerator=DummyGenerator(), auto_blend=0.4, require_regions=True)

    # 'vocals' and 'drums' should be in processed; 'bass' and 'other' skipped
    processed = [p['stem'] for p in results['results']['processed']]
    skipped = results['results']['skipped']

    assert 'vocals' in processed
    assert 'drums' in processed
    assert 'bass' in skipped
    assert 'other' in skipped

    # Now run with aggressive (require_regions=False) and verify the previously skipped stems are processed
    results_aggressive = auto_repair_all_stems(stems.copy(), sr, regenerator=DummyGenerator(), auto_blend=0.4, require_regions=False)
    processed_aggr = [p['stem'] for p in results_aggressive['results']['processed']]
    assert 'bass' in processed_aggr
    assert 'other' in processed_aggr

    # Snapshots should exist
    assert 'vocals' in results['snapshots']
    assert 'drums' in results['snapshots']

    # Ensure returned stems include processed and unchanged entries
    assert set(results['stems'].keys()) == set(stems.keys())
