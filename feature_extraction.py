
from __future__ import annotations
import numpy as np
import mne
from typing import Dict, Tuple, Optional
from scipy.signal import hilbert

# Optional: PyEMD for HHT
try:
    from PyEMD import EMD
    _HAS_PYEMD = True
except Exception:
    _HAS_PYEMD = False

EEG_BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "beta":  (12.0, 30.0),
    "gamma": (30.0, 45.0),
}

def _bandpower(psd_f: np.ndarray, psd_p: np.ndarray, fmin: float, fmax: float) -> float:
    mask = (psd_f >= fmin) & (psd_f < fmax)
    return np.trapz(psd_p[mask], psd_f[mask]) if np.any(mask) else 0.0

def bandpower_features(segments: mne.io.BaseRaw | mne.Epochs, bands: Dict[str, Tuple[float, float]] = EEG_BANDS) -> np.ndarray:
    """Compute bandpower features per epoch (or over continuous raw if no epochs).

    Returns
    -------
    np.ndarray
        Shape (n_samples, n_channels * n_bands)
    """
    if isinstance(segments, mne.io.BaseRaw):
        # Treat entire recording as single epoch
        data = segments.get_data()[None, ...]  # (1, n_ch, n_times)
        sfreq = segments.info['sfreq']
    else:
        data = segments.get_data()  # (n_epochs, n_ch, n_times)
        sfreq = segments.info['sfreq']

    n_epochs, n_ch, _ = data.shape
    n_bands = len(bands)
    out = np.zeros((n_epochs, n_ch * n_bands), dtype=float)

    for ei in range(n_epochs):
        for ci in range(n_ch):
            freqs, psd = mne.time_frequency.psd_array_welch(
                data[ei, ci],
                sfreq=sfreq,
                fmin=1.0,
                fmax=45.0,
                n_fft=1024,
                n_overlap=256,
                n_per_seg=512,
                verbose=False,
            )
            for bi, (bn, (lo, hi)) in enumerate(bands.items()):
                out[ei, ci * n_bands + bi] = _bandpower(freqs, psd, lo, hi)
    return out

def hht_features(epoch: np.ndarray, sfreq: float, max_imf: int = 6) -> np.ndarray:
    """Extract simple Hilbert-Huang features from a single-channel epoch.

    Produces per-IMF statistics of instantaneous amplitude and frequency.

    Returns
    -------
    np.ndarray of shape (max_imf * 4,)
    """
    if not _HAS_PYEMD:
        # Graceful fallback: zeros if PyEMD is not installed
        return np.zeros(max_imf * 4, dtype=float)

    emd = EMD()
    imfs = emd(epoch)
    if imfs.ndim == 1:
        imfs = imfs[None, :]

    imfs = imfs[:max_imf]
    feats = []
    t = np.arange(epoch.shape[0]) / sfreq
    for imf in imfs:
        analytic = hilbert(imf)
        amp = np.abs(analytic)
        phase = np.unwrap(np.angle(analytic))
        # Instantaneous frequency (derivative of phase / 2pi)
        inst_freq = np.diff(phase) * sfreq / (2 * np.pi)
        # Basic stats
        feats.extend([
            float(np.mean(amp)),
            float(np.std(amp)),
            float(np.mean(inst_freq)) if inst_freq.size else 0.0,
            float(np.std(inst_freq)) if inst_freq.size else 0.0,
        ])

    # Pad to fixed length
    while len(feats) < max_imf * 4:
        feats.append(0.0)
    return np.asarray(feats, dtype=float)
