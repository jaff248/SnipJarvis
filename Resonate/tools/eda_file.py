import numpy as np
import librosa
import sys
import json
from pathlib import Path


try:
    import soundfile as sf
    have_sf = True
except Exception:
    have_sf = False

def estimate_snr(audio: np.ndarray, noise_floor_db: float = -60.0) -> float:
    signal_power = np.mean(audio ** 2)
    noise_power = 10 ** (noise_floor_db / 10)
    if noise_power == 0:
        return float('inf')
    snr = 10 * np.log10(signal_power / noise_power)
    return max(snr, 0)


def estimate_spectral_centroid(audio: np.ndarray, sample_rate: int) -> float:
    S = np.abs(librosa.stft(audio))
    centroid = librosa.feature.spectral_centroid(S=S, sr=sample_rate)
    return float(np.mean(centroid))


def estimate_frequency_range(audio: np.ndarray, sample_rate: int) -> tuple:
    S = np.abs(librosa.stft(audio))
    energy = np.mean(S, axis=1)
    freqs = librosa.fft_frequencies(sr=sample_rate)
    cumsum = np.cumsum(energy)
    total = cumsum[-1]
    low_idx = np.searchsorted(cumsum, 0.01 * total)
    high_idx = np.searchsorted(cumsum, 0.99 * total)
    low = float(freqs[low_idx]) if low_idx < len(freqs) else 0.0
    high = float(freqs[high_idx]) if high_idx < len(freqs) else sample_rate / 2
    return low, high


def detect_clipping(audio: np.ndarray, threshold: float = 0.95) -> dict:
    max_val = np.max(np.abs(audio))
    clipped_samples = np.sum(np.abs(audio) > threshold)
    total_samples = len(audio)
    return {
        "max_amplitude": float(max_val),
        "clipped_samples": int(clipped_samples),
        "clipping_percent": float(clipped_samples / total_samples * 100),
        "is_clipped": clipped_samples > 0
    }


try:
    import pyloudnorm as pyln
    have_pyln = True
except Exception:
    have_pyln = False


def approximate_lufs(audio, sr):
    if have_pyln:
        meter = pyln.Meter(sr)
        return meter.integrated_loudness(audio)
    else:
        rms = np.sqrt(np.mean(audio ** 2))
        return 20 * np.log10(rms + 1e-10) - 0.691


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python eda_file.py <audio_file>")
        sys.exit(2)

    path = sys.argv[1]
    if have_sf:
        audio, sr = sf.read(path)
    else:
        audio, sr = librosa.load(path, sr=None, mono=False)
        # librosa returns mono or stereo arrays; make shape consistent
        audio = np.asarray(audio).T if audio.ndim > 1 else np.asarray(audio)

    # If multi-channel, mix to mono for analysis
    if audio.ndim > 1:
        mono = np.mean(audio, axis=1)
    else:
        mono = audio

    duration_s = len(mono) / sr
    snr = estimate_snr(mono)
    centroid = estimate_spectral_centroid(mono, sr)
    freq_low, freq_high = estimate_frequency_range(mono, sr)
    clipping = detect_clipping(mono)
    lufs = approximate_lufs(mono, sr)

    # Dynamic range: approximate via 1s windows
    win = sr
    rms_db = []
    for i in range(0, len(mono), win):
        w = mono[i:i+win]
        if w.size:
            rms = np.sqrt(np.mean(w**2))
            rms_db.append(20*np.log10(rms+1e-10))
    dyn_range = max(rms_db) - min(rms_db) if len(rms_db) >= 2 else 0.0

    out = {
        'file': path,
        'sample_rate': sr,
        'duration_s': duration_s,
        'channels': audio.shape[1] if audio.ndim > 1 else 1,
        'snr_db': float(snr),
        'spectral_centroid_hz': float(centroid),
        'frequency_range_hz': [float(freq_low), float(freq_high)],
        'clipping_percent': float(clipping['clipping_percent']),
        'is_clipped': bool(clipping['is_clipped']),
        'loudness_lufs': float(lufs),
        'dynamic_range_db': float(dyn_range),
    }

    print(json.dumps(out, indent=2))
