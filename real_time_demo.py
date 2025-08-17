
"""Real-time EEG demo with BrainFlow + PyQtGraph.

- Streams data from a BrainFlow-supported board (e.g., OpenBCI)
- Plots time series and PSD per channel
- Computes band power and simple stress heuristic
- (Optional) Loads a trained SVM model to predict stress in real-time
"""
from __future__ import annotations
import argparse
import logging
import numpy as np
from pathlib import Path

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, WindowOperations, DetrendOperations
from pyqtgraph.Qt import QtWidgets, QtCore
import pyqtgraph as pg

try:
    from joblib import load as joblib_load
    _HAS_JOBLIB = True
except Exception:
    _HAS_JOBLIB = False

EEG_BANDS = [
    (1.0, 4.0),   # delta
    (4.0, 8.0),   # theta
    (8.0, 12.0),  # alpha
    (12.0, 30.0), # beta
    (30.0, 45.0), # gamma
]

class LiveApp:
    def __init__(self, board_shim: BoardShim, model_path: str | None = None):
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')

        self.board_shim = board_shim
        self.board_id = board_shim.get_board_id()
        self.exg_channels = BoardShim.get_exg_channels(self.board_id)
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)

        self.model = None
        if model_path and _HAS_JOBLIB and Path(model_path).exists():
            self.model = joblib_load(model_path)

        # UI
        self.app = QtWidgets.QApplication([])
        self.win = pg.GraphicsLayoutWidget(title='EEG Stress Demo', size=(1200, 800))

        self.plots = []
        self.curves = []
        for i, ch in enumerate(self.exg_channels):
            p = self.win.addPlot(row=i, col=0)
            p.showAxis('left', False)
            p.showAxis('bottom', False)
            if i == 0:
                p.setTitle('Time Series')
            curve = p.plot()
            self.plots.append(p)
            self.curves.append(curve)

        self.psd_plot = self.win.addPlot(row=0, col=1, rowspan=max(1, len(self.exg_channels)//2))
        self.psd_plot.setTitle('PSD (log power)')
        self.psd_plot.setLogMode(False, True)
        self.psd_curves = [self.psd_plot.plot() for _ in self.exg_channels]

        self.band_plot = self.win.addPlot(row=max(1, len(self.exg_channels)//2), col=1, rowspan=max(1, len(self.exg_channels)//2))
        self.band_plot.setTitle('Band Power (avg across channels)')
        self.band_bar = pg.BarGraphItem(x=np.arange(5)+1, height=[0,0,0,0,0], width=0.8)
        self.band_plot.addItem(self.band_bar)

        self.status = pg.LabelItem(text="Status: ...", justify='left')
        self.win.addItem(self.status, row=len(self.exg_channels), col=0, colspan=2)

        self.window_sec = 4
        self.update_ms = 100
        self.psd_size = DataFilter.get_nearest_power_of_two(self.sampling_rate)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(self.update_ms)
        self.win.show()
        QtWidgets.QApplication.instance().exec_()

    def update(self):
        data = self.board_shim.get_current_board_data(self.window_sec * self.sampling_rate)
        if data.shape[1] < self.window_sec * self.sampling_rate // 2:
            return

        avg_bands = np.zeros(5, dtype=float)
        for idx, ch in enumerate(self.exg_channels):
            ch_data = data[ch].copy()
            DataFilter.detrend(ch_data, DetrendOperations.CONSTANT.value)
            DataFilter.perform_bandpass(ch_data, self.sampling_rate, 1.0, 45.0, 3, FilterTypes.BUTTERWORTH.value, 0)
            DataFilter.perform_bandstop(ch_data, self.sampling_rate, 48.0, 52.0, 3, FilterTypes.BUTTERWORTH.value, 0)
            DataFilter.perform_bandstop(ch_data, self.sampling_rate, 58.0, 62.0, 3, FilterTypes.BUTTERWORTH.value, 0)

            # Plot time series
            self.curves[idx].setData(ch_data.tolist())

            # PSD for plotting + band power
            psd = DataFilter.get_psd_welch(ch_data, self.psd_size, self.psd_size // 2, self.sampling_rate, WindowOperations.BLACKMAN_HARRIS.value)
            f = psd[1]
            p = psd[0]
            lim = min(70, len(f))
            self.psd_curves[idx].setData(f[:lim].tolist(), p[:lim].tolist())

            for bi, (lo, hi) in enumerate(EEG_BANDS):
                avg_bands[bi] += DataFilter.get_band_power(psd, lo, hi)

        avg_bands /= len(self.exg_channels)
        self.band_bar.setOpts(height=avg_bands.tolist())

        # Simple heuristic: high beta & low alpha => stressed
        alpha = avg_bands[2]
        beta = avg_bands[3]
        stress_score = (beta + 1e-8) / (alpha + 1e-8)

        label = "Stressed" if stress_score > 1.2 else "Relaxed"
        if self.model is not None:
            # If a trained model is provided, use it instead of heuristic.
            # Here we feed [delta, theta, alpha, beta, gamma] as features.
            X = avg_bands.reshape(1, -1)
            try:
                proba = self.model.predict_proba(X)[0, 1]
                label = f"Model: Stressed ({proba:.2f})" if proba >= 0.5 else f"Model: Relaxed ({1-proba:.2f})"
            except Exception:
                pass

        self.status.setText(f"Status: {label} | alpha={alpha:.4f}, beta={beta:.4f}, score={stress_score:.3f}")

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--board-id', type=int, default=BoardIds.SYNTHETIC_BOARD, help='See BrainFlow docs for IDs (e.g., OpenBCI Cyton).')
    parser.add_argument('--serial-port', type=str, default='', help='Serial port for your board (e.g., COM3 or /dev/ttyUSB0).')
    parser.add_argument('--model', type=str, default=None, help='Optional path to joblib SVM model.')
    args = parser.parse_args()

    BoardShim.enable_dev_board_logger()
    logging.basicConfig(level=logging.INFO)

    params = BrainFlowInputParams()
    params.serial_port = args.serial_port

    board_shim = BoardShim(args.board_id, params)
    try:
        board_shim.prepare_session()
        board_shim.start_stream(450000)
        LiveApp(board_shim, model_path=args.model)
    finally:
        if board_shim.is_prepared():
            board_shim.release_session()

if __name__ == '__main__':
    run()
