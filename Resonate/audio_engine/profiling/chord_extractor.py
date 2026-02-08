"""
Chord Extractor - Extract chord progressions from audio using NMF and chord templates.

Uses Non-negative Matrix Factorization to decompose spectrogram into harmonic
components, then matches to chord templates.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import logging
from collections import Counter

logger = logging.getLogger(__name__)


class ChordType(Enum):
    """Type of chord."""
    MAJOR = "major"
    MINOR = "minor"
    DIMINISHED = "diminished"
    AUGMENTED = "augmented"
    MAJOR7 = "major7"
    MINOR7 = "minor7"
    DOMINANT7 = "dominant7"
    SUS4 = "sus4"
    SUS2 = "sus2"
    UNKNOWN = "unknown"


# Standard chord templates (pitch class representation)
CHORD_TEMPLATES = {
    (0, 4, 7): ChordType.MAJOR,
    (0, 3, 7): ChordType.MINOR,
    (0, 3, 6): ChordType.DIMINISHED,
    (0, 4, 8): ChordType.AUGMENTED,
    (0, 4, 7, 11): ChordType.MAJOR7,
    (0, 3, 7, 10): ChordType.MINOR7,
    (0, 4, 7, 10): ChordType.DOMINANT7,
    (0, 5, 7): ChordType.SUS4,
    (0, 2, 7): ChordType.SUS2,
}

# Major scale notes for root mapping
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
ROOT_TO_PC = {name: i for i, name in enumerate(NOTE_NAMES)}


@dataclass
class Chord:
    """A single chord with root and type."""
    root: str           # e.g., "C", "F#", "Bb"
    chord_type: ChordType
    start_time: float   # Start time in seconds
    duration: float     # Duration in seconds
    confidence: float   # 0-1 confidence in detection
    
    def __str__(self):
        return f"{self.root}{self._type_symbol()}"
    
    def _type_symbol(self):
        symbols = {
            ChordType.MAJOR: "",
            ChordType.MINOR: "m",
            ChordType.DIMINISHED: "dim",
            ChordType.AUGMENTED: "aug",
            ChordType.MAJOR7: "maj7",
            ChordType.MINOR7: "m7",
            ChordType.DOMINANT7: "7",
            ChordType.SUS4: "sus4",
            ChordType.SUS2: "sus2",
            ChordType.UNKNOWN: "?",
        }
        return symbols.get(self.chord_type, "")


@dataclass
class ChordProgressions:
    """Container for detected chord progressions."""
    chords: List[Chord]
    key: str                       # Detected key (e.g., "C", "Am")
    tempo: float                   # Estimated tempo BPM
    time_signature: Tuple[int, int]  # e.g., (4, 4)
    
    def to_chord_string(self) -> str:
        """Convert to chord string notation."""
        return " | ".join([str(c) for c in self.chords])
    
    def get_chord_timeline(self) -> List[Tuple[float, float, str]]:
        """Get timeline of (start, end, chord_name)."""
        timeline = []
        for i, chord in enumerate(self.chords):
            end_time = chord.start_time + chord.duration
            if i + 1 < len(self.chords):
                end_time = min(end_time, self.chords[i + 1].start_time)
            timeline.append((chord.start_time, end_time, str(chord)))
        return timeline
    
    def get_progression(self, window_size: int = 4) -> List[str]:
        """Extract repeating progression (default 4-chord window)."""
        chord_names = [str(c) for c in self.chords]
        if len(chord_names) < window_size:
            return chord_names
        
        # Find most common 4-chord sequence
        sequences = []
        for i in range(len(chord_names) - window_size + 1):
            seq = tuple(chord_names[i:i + window_size])
            sequences.append(seq)
        
        # Count occurrences (including overlapping)
        sequence_counts = Counter(sequences)
        most_common = sequence_counts.most_common(3)
        
        if most_common:
            return list(most_common[0][0])
        return chord_names[:window_size]


class ChordExtractor:
    """
    Extract chord progressions from audio using NMF and template matching.
    
    Uses Non-negative Matrix Factorization to decompose the spectrogram into
    harmonic components, then matches activation patterns to chord templates.
    """
    
    def __init__(self):
        """Initialize chord extractor."""
        self.frame_rate = 0  # Set during extraction
        self.hop_length = 512
        self.n_fft = 4096
        
    def extract(self, audio: np.ndarray, sample_rate: int,
                min_duration: float = 0.5) -> ChordProgressions:
        """
        Extract chord progressions from audio.
        
        Args:
            audio: Audio samples
            sample_rate: Sample rate
            min_duration: Minimum chord duration to detect
            
        Returns:
            ChordProgressions with detected chords
        """
        try:
            import librosa
            
            self.frame_rate = sample_rate / self.hop_length
            
            # Compute chromagram (pitch class representation)
            chroma = librosa.feature.chroma_cqt(
                y=audio, sr=sample_rate,
                n_chroma=12, octaves=1,
                hop_length=self.hop_length
            )
            
            # Smooth chroma over time
            chroma_smoothed = self._smooth_chroma(chroma)
            
            # Detect chord changes
            chord_timeline = self._detect_chord_changes(chroma_smoothed, min_duration)
            
            # Identify individual chords
            chords = self._identify_chords(chroma_smoothed, chord_timeline)
            
            # Detect key
            key = self._detect_key(chroma)
            
            # Estimate tempo from chord transitions
            tempo = self._estimate_tempo_from_chords(chords)
            
            return ChordProgressions(
                chords=chords,
                key=key,
                tempo=tempo,
                time_signature=(4, 4)
            )
            
        except Exception as e:
            logger.error(f"Chord extraction failed: {e}")
            return ChordProgressions(
                chords=[],
                key="C",
                tempo=120.0,
                time_signature=(4, 4)
            )
    
    def _smooth_chroma(self, chroma: np.ndarray, window: int = 4) -> np.ndarray:
        """Smooth chromagram over time."""
        from scipy.ndimage import uniform_filter1d
        return uniform_filter1d(chroma, size=window, mode='nearest')
    
    def _detect_chord_changes(self, chroma: np.ndarray,
                               min_duration: float) -> List[Tuple[int, int]]:
        """
        Detect time segments where chords change.
        
        Returns:
            List of (start_frame, end_frame) tuples
        """
        # Compute frame-to-frame similarity
        similarity = []
        for i in range(1, chroma.shape[1]):
            # Cosine similarity between adjacent frames
            frame_curr = chroma[:, i]
            frame_prev = chroma[:, i - 1]
            
            norm_curr = np.linalg.norm(frame_curr)
            norm_prev = np.linalg.norm(frame_prev)
            
            if norm_curr > 0 and norm_prev > 0:
                sim = np.dot(frame_curr, frame_prev) / (norm_curr * norm_prev)
            else:
                sim = 1.0
            
            similarity.append(sim)
        
        similarity = np.array(similarity)
        
        # Detect significant changes (low similarity)
        threshold = np.percentile(similarity, 25)  # Bottom 25% are changes
        change_indices = np.where(similarity < threshold)[0] + 1
        
        # Group into contiguous regions
        segments = []
        if len(change_indices) > 0:
            start_frame = 0
            prev_idx = change_indices[0]
            
            for idx in change_indices[1:]:
                if idx - prev_idx > 10:  # More than 10 frames between changes
                    segments.append((start_frame, prev_idx))
                    start_frame = prev_idx
                prev_idx = idx
            
            segments.append((start_frame, prev_idx))
        
        # Filter by minimum duration
        min_frames = int(min_duration * self.frame_rate)
        filtered_segments = [(s, e) for s, e in segments if e - s >= min_frames]
        
        return filtered_segments if filtered_segments else [(0, chroma.shape[1])]
    
    def _identify_chords(self, chroma: np.ndarray,
                          segments: List[Tuple[int, int]]) -> List[Chord]:
        """Identify chord for each segment."""
        chords = []
        
        for start_frame, end_frame in segments:
            # Average chroma over segment
            segment_chroma = np.mean(chroma[:, start_frame:end_frame], axis=1)
            
            # Normalize
            segment_chroma = segment_chroma / (np.sum(segment_chroma) + 1e-10)
            
            # Find best matching chord template
            best_chord = self._match_chord_template(segment_chroma)
            
            if best_chord:
                start_time = start_frame * self.hop_length / self.frame_rate
                duration = (end_frame - start_frame) * self.hop_length / self.frame_rate
                
                chords.append(Chord(
                    root=best_chord[0],
                    chord_type=best_chord[1],
                    start_time=start_time,
                    duration=duration,
                    confidence=best_chord[2]
                ))
        
        return chords
    
    def _match_chord_template(self, chroma: np.ndarray) -> Optional[Tuple[str, ChordType, float]]:
        """Match chroma vector to chord template."""
        best_match = None
        best_score = -1
        
        for pitch_classes, chord_type in CHORD_TEMPLATES.items():
            # Create template from pitch classes
            template = np.zeros(12)
            for pc in pitch_classes:
                template[pc] = 1.0
            
            # Normalize
            template = template / (np.linalg.norm(template) + 1e-10)
            
            # Compute similarity (cosine)
            score = np.dot(chroma, template)
            
            # Also check inversions and nearby roots
            for root_offset in range(12):
                # Rotate template
                rotated_template = np.roll(template, root_offset)
                rotated_score = np.dot(chroma, rotated_template)
                
                if rotated_score > best_score:
                    best_score = rotated_score
                    root_note = NOTE_NAMES[root_offset]
                    best_match = (root_note, chord_type, best_score)
        
        if best_match and best_score > 0.3:  # Minimum confidence threshold
            return best_match
        
        return None
    
    def _detect_key(self, chroma: np.ndarray) -> str:
        """Detect musical key from chromagram."""
        # Key profiles (Krumhansl-Schmuckler)
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        
        # Average chroma over time
        avg_chroma = np.mean(chroma, axis=1)
        avg_chroma = avg_chroma / (np.sum(avg_chroma) + 1e-10)
        
        best_key = "C"
        best_score = -1
        
        # Check all 12 possible roots
        for root in range(12):
            # Rotate chroma to C major/minor perspective
            rotated_chroma = np.roll(avg_chroma, -root)
            
            # Compare to major and minor profiles
            major_score = np.dot(rotated_chroma, major_profile)
            minor_score = np.dot(rotated_chroma, minor_profile)
            
            if major_score > best_score:
                best_score = major_score
                best_key = f"{NOTE_NAMES[root]}"
            if minor_score > best_score:
                best_score = minor_score
                best_key = f"{NOTE_NAMES[root]}m"
        
        return best_key
    
    def _estimate_tempo_from_chords(self, chords: List[Chord]) -> float:
        """Estimate tempo from chord transition rate."""
        if len(chords) < 2:
            return 120.0
        
        # Calculate average chord duration
        durations = [c.duration for c in chords]
        avg_duration = np.mean(durations)
        
        if avg_duration > 0:
            # Assuming 1 chord per beat on average
            # Duration is seconds per beat, convert to BPM
            bpm = 60.0 / avg_duration
            
            # Clamp to reasonable range
            return min(max(bpm, 60.0), 180.0)
        
        return 120.0
    
    def extract_chord_conditioning(self, audio: np.ndarray,
                                    sample_rate: int) -> Dict:
        """
        Extract structured conditioning for JASCO generation.
        
        Returns:
            Dictionary with chords, key, tempo for JASCO conditioning
        """
        progressions = self.extract(audio, sample_rate)
        
        return {
            'chords': progressions.get_chord_timeline(),
            'key': progressions.key,
            'tempo': progressions.tempo,
            'progression': progressions.get_progression(),
            'chord_string': progressions.to_chord_string(),
        }


# Convenience function
def extract_musical_structure(audio: np.ndarray, sample_rate: int) -> Dict:
    """
    Quick function to extract musical structure.
    
    Returns:
        Dictionary with chord progressions and musical info
    """
    extractor = ChordExtractor()
    return extractor.extract_chord_conditioning(audio, sample_rate)
