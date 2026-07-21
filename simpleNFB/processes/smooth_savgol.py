"""Savitzky-Golay smoothing (was DAT's svgToggle + process_utils.smooth_data)."""

from scipy.signal import savgol_filter
import numpy as np

NAME = 'Savitzky-Golay'
KIND = '1d'
DESCRIPTION = 'Savitzky-Golay polynomial smoothing.'
PARAMS = [
    {'name': 'window_size', 'label': 'w', 'type': 'int', 'default': 3, 'min': 3, 'max': 101, 'step': 2,
     'tooltip': 'filter window length (odd, >= 3)'},
    {'name': 'order', 'label': 'o', 'type': 'int', 'default': 1, 'min': 1, 'max': 5, 'step': 1,
     'tooltip': 'polynomial order (< window_size)'},
]


def process(y, *, window_size=3, order=1):
    return savgol_filter(np.asarray(y, dtype=float), window_size, order)
