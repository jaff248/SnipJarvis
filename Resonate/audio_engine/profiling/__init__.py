"""
Profiling Module - Musical structure extraction and stem quality analysis

Submodules:
- quality_detector: Automatic stem damage detection
- chord_extractor: Chord progression extraction
- tempo_key_analyzer: BPM and key detection
- melody_extractor: Melody contour extraction
- drum_pattern_extractor: Drum onset/pattern detection

These modules extract musical structure information to condition JASCO
for stem regeneration. They provide structured data about tempo, key,
chords, melody, and drum patterns that guide the regeneration process.
"""

import numpy as np

from .quality_detector import QualityDetector, DamageLevel, StemQualityReport
from .chord_extractor import ChordExtractor, ChordProgressions
from .tempo_key_analyzer import (
    TempoKeyAnalyzer, 
    TempoKeyInfo,
    detect_tempo_key
)
from .melody_extractor import (
    MelodyExtractor, 
    MelodyContour,
    extract_melody
)
from .drum_pattern_extractor import (
    DrumPatternExtractor, 
    DrumOnsets,
    extract_drum_pattern
)

__all__ = [
    # Quality detection
    'QualityDetector',
    'DamageLevel', 
    'StemQualityReport',
    
    # Chord extraction
    'ChordExtractor',
    'ChordProgressions',
    
    # Tempo/Key analysis
    'TempoKeyAnalyzer',
    'TempoKeyInfo',
    'detect_tempo_key',
    
    # Melody extraction
    'MelodyExtractor',
    'MelodyContour',
    'extract_melody',
    
    # Drum pattern extraction
    'DrumPatternExtractor',
    'DrumOnsets',
    'extract_drum_pattern',
]


def extract_all_profiles(audio: np.ndarray, sample_rate: int) -> dict:
    """
    Extract all musical profiles from audio for JASCO conditioning.
    
    Args:
        audio: Audio samples
        sample_rate: Sample rate
        
    Returns:
        Dictionary with all extracted profiles
    """
    import numpy as np
    
    # Extract each profile
    quality = QualityDetector().analyze(audio, sample_rate)
    chords = ChordExtractor().extract(audio, sample_rate)
    tempo_key = TempoKeyAnalyzer().analyze(audio, sample_rate)
    melody = MelodyExtractor().extract(audio, sample_rate)
    drums = DrumPatternExtractor().extract(audio, sample_rate)
    
    return {
        'quality': quality.to_dict(),
        'chords': {
            'progression': chords.to_chord_string(),
            'key': chords.key,
            'tempo': chords.tempo,
            'timeline': chords.get_chord_timeline(),
        },
        'tempo_key': tempo_key.to_dict(),
        'melody': melody.to_dict(),
        'drums': drums.to_dict(),
    }
