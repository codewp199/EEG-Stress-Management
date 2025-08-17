
from __future__ import annotations
import numpy as np
import mne
from typing import List, Tuple, Optional

# Common EEG bands in Hz
EEG_BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "beta":  (12.0, 30.0),
    "gamma": (30.0, 45.0),
}

def load_openbci_csv(path: str, sfreq: float = 250.0, ch_names: Optional[List[str]] = None) -> mne.io.RawArray:
    """Load an OpenBCI-exported CSV/TSV file and return an MNE RawArray.

    Parameters
    ----------
    path : str
        CSV/TSV path with EEG samples in columns.
    sfreq : float
        Sampling frequency (Hz). Adjust to your board settings (e.g., 250 for Cyton).
    ch_names : list of str, optional
        Channel names. If None, Channel 0..N-1 will be used.

    Returns
    -------
    mne.io.RawArray
    """
    # Attempt to auto-detect delimiter; OpenBCI sometimes uses tabs.
    try:
        data = np.genfromtxt(path, delimiter='\t', dtype=float)
        if data.ndim == 1:
            data = np.genfromtxt(path, delimiter=',', dtype=float)
    except Exception:
        data = np.genfromtxt(path, delimiter=',', dtype=float)

    if data.ndim != 2:
        raise ValueError(f"Unexpected data shape: {data.shape}. Expected 2D array (samples x channels).")


    # If the file is channels x samples, transpose to samples x channels
    if data.shape[0] < data.shape[1]:
        # Heuristic: most OpenBCI CSVs are rows=samples
        pass

    n_ch = data.shape[1]
    if ch_names is None:
        ch_names = [f"CH{i+1}" for i in range(n_ch)]
    ch_types = ['eeg'] * n_ch

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data.T, info, verbose=False)
    return raw

def apply_filters(raw: mne.io.BaseRaw, l_freq: float = 0.5, h_freq: float = 45.0, notch: Optional[List[float]] = None) -> mne.io.BaseRaw:
    """Apply band-pass and notch filtering (zero-phase) to EEG.

    Parameters
    ----------
    raw : mne.io.BaseRaw
    l_freq, h_freq : float
        Band-pass edges in Hz.
    notch : list of float, optional
        Frequencies to notch (e.g., [50, 60]).

    Returns
    -------
    mne.io.BaseRaw
    """
    raw = raw.copy().filter(l_freq=l_freq, h_freq=h_freq, method='fir', fir_design='firwin', phase='zero', verbose=False)
    if notch:
        raw = raw.notch_filter(freqs=notch, verbose=False)
    return raw

def epoch_fixed(raw: mne.io.BaseRaw, length_s: float = 2.0, overlap: float = 0.5) -> mne.EpochsArray:
    """Create fixed-length, overlapping epochs.

    Parameters
    ----------
    raw : mne.io.BaseRaw
    length_s : float
        Epoch length in seconds.
    overlap : float
        Overlap ratio in [0,1). 0.5 means 50% overlap.

    Returns
    -------
    mne.EpochsArray
    """
    sfreq = raw.info['sfreq']
    n_samples = int(length_s * sfreq)
    step = int(n_samples * (1 - overlap))
    data = raw.get_data()  # shape (n_channels, n_times)

    starts = np.arange(0, data.shape[1] - n_samples + 1, step, dtype=int)
    epochs_data = np.stack([data[:, s:s + n_samples] for s in starts], axis=0)  # (n_epochs, n_channels, n_samples)

    info = raw.info.copy()
    events = np.column_stack([starts, np.zeros_like(starts), np.ones_like(starts)]).astype(int)
    epochs = mne.EpochsArray(epochs_data, info, events=events, verbose=False)
    return epochs
