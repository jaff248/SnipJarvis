from typing import Dict, Any, Optional
import numpy as np
# Import heavy modules lazily inside function to avoid import-time dependency in tests


def auto_repair_all_stems(
    stems: Dict[str, np.ndarray],
    sample_rate: int,
    regenerator: Optional["StemRegenerator"] = None,
    auto_blend: float = 0.4,
    require_regions: bool = True,
) -> Dict[str, Any]:
    """Apply conservative selective regeneration across all stems.

    Args:
        stems: dict of stem_name -> audio numpy arrays
        sample_rate: sample rate in Hz
        regenerator: optional StemRegenerator (created if None)
        auto_blend: blend ratio used to merge regenerated audio (0-1)
        require_regions: if True, only regenerate stems that have detected regions; otherwise regenerates whole-stem when no regions

    Returns:
        dict containing 'stems' (updated stems dict), 'snapshots' (original stems), and 'results' (processed/skipped/errors lists)
    """
    generator = regenerator or StemRegenerator()
    snapshots: Dict[str, np.ndarray] = {}
    results: Dict[str, list] = {'processed': [], 'skipped': [], 'errors': []}

    for stem_name, audio in list(stems.items()):
        try:
            snapshots[stem_name] = audio.copy()

            if regenerator is None:
                from .stem_regenerator import StemRegenerator
                generator = StemRegenerator()
            else:
                generator = regenerator

            plan = generator.create_regeneration_plan(audio=audio, stem_name=stem_name, sample_rate=sample_rate, musical_profile={})

            if require_regions and not plan.regions:
                results['skipped'].append(stem_name)
                continue

            # If no regions but require_regions False, we regenerate whole stem
            if not plan.regions and not require_regions:
                # Create a synthetic whole-stem region; avoid importing RegenerationRegion to stay lightweight
                plan.use_whole_stem = True
                duration = len(audio) / sample_rate
                try:
                    from .stem_regenerator import RegenerationRegion
                    plan.regions = [RegenerationRegion(0.0, duration, stem_name, getattr(plan, 'quality_report', None).damage_level if getattr(plan, 'quality_report', None) else None, quality_report=getattr(plan, 'quality_report', None), confidence=(getattr(plan, 'quality_report', None).confidence if getattr(plan, 'quality_report', None) else 0.8))]
                except Exception:
                    from types import SimpleNamespace
                    plan.regions = [SimpleNamespace(start_time=0.0, end_time=duration)]

            # Log start of per-stem processing
            import logging
            logger = logging.getLogger('resonate.regen')
            logger.info(f"Auto-Repair All: starting regeneration for stem '{stem_name}' (use_whole_stem={getattr(plan, 'use_whole_stem', False)}, regions={len(getattr(plan, 'regions', []))})")

            regenerated, summary = generator.execute_regeneration_plan(plan)

            # Blend with loudness matching
            # Attempt to use Blender for loudness-matched blending; fall back to simple crossfade when unavailable
            try:
                from .blender import Blender
                blender = Blender()
                regenerated_blended = blender.blend_with_loudness_match(
                    original=audio,
                    regenerated=regenerated,
                    start_time=0.0,
                    end_time=len(audio) / sample_rate,
                    fade_duration=0.1,
                    sample_rate=sample_rate,
                )
            except Exception as e:
                # Fall back to naive approach: use regenerated as-is but log the fallback
                logger.warning(f"Blender unavailable or failed; using naive regenerated audio for blending: {e}")
                regenerated_blended = regenerated

            # If regenerated_blended length matches original, do a global blend
            if len(regenerated_blended) == len(audio):
                final = (1.0 - auto_blend) * audio + auto_blend * regenerated_blended
            else:
                # If regenerated segment is shorter, place it into the corresponding region(s)
                final = audio.copy()
                # Determine sample ranges from plan regions if available
                if getattr(plan, 'regions', None):
                    for r in plan.regions:
                        start_sample = int(r.start_time * sample_rate)
                        end_sample = start_sample + len(regenerated_blended)
                        # Clip to bounds
                        end_sample = min(end_sample, len(audio))
                        seg = regenerated_blended[: end_sample - start_sample]
                        final[start_sample:end_sample] = (1.0 - auto_blend) * audio[start_sample:end_sample] + auto_blend * seg
                else:
                    # Fallback: if we cannot map region, pad or tile regenerated and blend
                    min_len = min(len(audio), len(regenerated_blended))
                    final[:min_len] = (1.0 - auto_blend) * audio[:min_len] + auto_blend * regenerated_blended[:min_len]

            stems[stem_name] = final
            results['processed'].append({'stem': stem_name, 'summary': summary})

            logger.info(f"Auto-Repair All: completed regeneration for stem '{stem_name}', regions_regenerated={summary.get('regions_regenerated', 'N/A')}")

            # Write a brief structured log entry to the regeneration log file
            try:
                from logging_config import get_logger as _get_logger
                _logger = _get_logger()
                _logger.info({
                    'event': 'auto_repair_stem_completed',
                    'stem': stem_name,
                    'regions': len(plan.regions),
                    'regions_regenerated': summary.get('regions_regenerated', 0),
                })
            except Exception:
                # Best-effort logging only
                pass

        except Exception as e:
            results['errors'].append({'stem': stem_name, 'error': str(e)})
            import logging
            logger = logging.getLogger('resonate.regen')
            logger.exception(f"Auto-Repair All: failed regeneration for stem '{stem_name}': {e}")

    return {
        'stems': stems,
        'snapshots': snapshots,
        'results': results,
    }


def remaster_project(
    stems: Dict[str, np.ndarray],
    sample_rate: int,
    regenerator: Optional["StemRegenerator"] = None,
    master_config: Optional["MasteringConfig"] = None,
    use_ai_if_needed: bool = True,
    aggressive: bool = False,
    auto_blend: float = 0.4,
    master_class: Optional[type] = None,
) -> Dict[str, Any]:
    """High-level remaster flow: analyze, optionally repair using AI, then mix and master.

    Args:
        stems: dict of stem_name -> audio arrays
        sample_rate: sample rate in Hz
        regenerator: optional StemRegenerator
        master_config: optional MasteringConfig for AudioMaster
        use_ai_if_needed: if True, run regeneration for stems deemed needing repair
        aggressive: if True, ignore region detection and allow whole-stem regen
        auto_blend: blend ratio used during regeneration blending

    Returns:
        dict with keys: 'reports' (quality reports per stem), 'repair_results' (if run),
        'mix_result' (StemMixer MixResult), 'mastering' (MasteringResult)
    """
    # Lazy imports to avoid heavy deps at module import time
    try:
        from ..profiling.quality_detector import QualityDetector, detect_stem_quality
        from ..mixing import StemMixer
        from ..mastering import AudioMaster, MasteringConfig
    except Exception:
        # If imports fail, raise a clear error to caller
        raise

    detector = QualityDetector()

    reports = {}
    for name, audio in stems.items():
        reports[name] = detector.analyze(audio, sample_rate)

    import logging
    logger = logging.getLogger('resonate.regen')
    logger.info({
        'event': 'remaster_start',
        'num_stems': len(stems),
        'use_ai_if_needed': use_ai_if_needed,
        'aggressive': aggressive,
    })

    # Decide which stems to repair
    stems_to_repair = []
    if use_ai_if_needed:
        for name, report in reports.items():
            if detector.should_regenerate(report) or report.distortion_score > 0.35:
                stems_to_repair.append(name)
    elif aggressive:
        stems_to_repair = list(stems.keys())

    repair_results = None
    snapshots = {}

    if stems_to_repair:
        # Build subset dict
        subset = {name: stems[name] for name in stems_to_repair}
        # Reuse existing auto_repair_all_stems but only for subset
        repair = auto_repair_all_stems(subset, sample_rate, regenerator=regenerator, auto_blend=auto_blend, require_regions=not aggressive)
        # Merge repaired stems back into stems
        for name, audio in repair['stems'].items():
            snapshots[name] = stems[name].copy()
            stems[name] = audio
        repair_results = repair['results']

    # Mix stems into a final mix
    mixer = StemMixer()
    mix_result = mixer.optimize_for_clarity(stems, sample_rate)

    # Apply mastering (allow dependency injection for testing)
    # Use a studio-oriented preset when no explicit master_config is provided
    # Prepare mastering configuration
    if master_config is None:
        # Prefer using MasteringConfig directly to avoid instantiating AudioMaster
        # (which may be monkeypatched in tests).
        from ..mastering import MasteringConfig
        default_conf = MasteringConfig()
        # Gentle mastering boost for "studio" quality
        default_conf.master_eq_high_db = getattr(default_conf, 'master_eq_high_db', 0.0) + 1.0
        default_conf.stereo_width = max(getattr(default_conf, 'stereo_width', 1.0), 1.0)
        default_conf.target_loudness_lufs = -14.0
        master_conf = default_conf
    else:
        master_conf = master_config

    MasterClass = master_class or AudioMaster
    master = MasterClass(master_conf)
    mastering_result = master.master(mix_result.mixed_audio, sample_rate)

    # Log completion
    try:
        logger.info({
            'event': 'remaster_complete',
            'stems_repaired': stems_to_repair,
            'mix_peak_db': getattr(mix_result, 'peak_level_db', None),
            'mix_rms_db': getattr(mix_result, 'rms_level_db', None),
            'master_loudness_lufs': getattr(mastering_result, 'loudness_lufs', None),
            'master_true_peak_db': getattr(mastering_result, 'true_peak_db', None),
        })
    except Exception:
        pass

    # Structured return
    return {
        'reports': {k: v.to_dict() for k, v in reports.items()},
        'stems_repaired': stems_to_repair,
        'repair_results': repair_results,
        'mix_result': mix_result,
        'mastering': mastering_result,
    }