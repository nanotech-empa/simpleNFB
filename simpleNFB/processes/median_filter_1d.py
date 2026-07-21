"""1-D median filter (was DAT's medFiltBtn, scipy.signal.medfilt inlined)."""

from scipy.signal import medfilt
import numpy as np

NAME = 'Median Filter'
KIND = '1d'
DESCRIPTION = '1-D median filter (scipy.signal.medfilt).'
PARAMS = [
    {'name': 'kernel_size', 'label': 'w', 'type': 'int', 'default': 3, 'min': 3, 'max': 21, 'step': 2,
     'tooltip': 'window length (odd)'},
]


def process(y, *, kernel_size=3):
    return medfilt(np.asarray(y, dtype=float), kernel_size)
