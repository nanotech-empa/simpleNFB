"""Group Average: batch spectra into groups, median-filter outliers, average.

Moved verbatim from process_utils.group_average, argument order swapped to
the batch KIND contract (xs, ys) -> (xs, ys); see process_utils shim.
"""

import numpy as np
from scipy.signal import medfilt

NAME = 'Group Average'
KIND = 'batch'
DESCRIPTION = 'Average every group_size consecutive traces (median-filtered, max dropped).'
PARAMS = [
    {'name': 'group_size', 'label': 'grp', 'type': 'int', 'default': 3, 'min': 3, 'max': 20, 'step': 1,
     'tooltip': 'number of traces per group'},
]


def process(xs, ys, *, group_size=3):
    grouped_ys = [ys[i:i + group_size] for i in range(0, len(ys), group_size)]
    grouped_xs = [xs[i] for i in range(0, len(xs), group_size)]

    out_ys = []
    for group in grouped_ys:
        median_average = []
        for element_group in zip(*group):
            medians = np.sort(medfilt(element_group, 3))
            medians = medians[:-1]  # drop maximum
            median_average.append(np.average(medians))
        out_ys.append(np.array(median_average))

    return grouped_xs, out_ys
