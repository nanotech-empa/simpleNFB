"""Fix Zero (spectra): shift y so its mean near x=0 is zero (was DAT's fixZeroBtn).

NOTE (flagged in DYNAMIC_PIPELINE_PLAN.md §4, pending sign-off): the legacy
toggle averaged the UNFILTERED spec_data[i] in the x~0 window; this xy-kind
version averages whatever y is handed to it (i.e. the current pipeline state
at this row), which differs when Fix Zero is not the first step.
"""

import numpy as np

NAME = 'Fix Zero'
KIND = 'xy'
DESCRIPTION = 'Subtract the mean y-value within |x| < window (baseline to zero).'
PARAMS = [
    {'name': 'window', 'label': 'win', 'type': 'float', 'default': 0.1,
     'tooltip': 'half-width of the x-window used to estimate the zero level'},
]


def process(x, y, *, window=0.1):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.abs(x) < window
    return x, y - np.mean(y[mask])
