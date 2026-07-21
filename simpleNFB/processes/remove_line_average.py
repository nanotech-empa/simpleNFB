"""Remove Line Average: per-row linear leveling of 2-D scan data.

Moved verbatim from process_utils.remove_line_average (also used directly by
SXM's session-context overlay code, which still imports it from
process_utils -- see the shim).
"""

import numpy as np

NAME = 'Remove Line Average'
KIND = '2d'
DESCRIPTION = 'Fit and subtract a linear trend from each row (removes tilt/drift).'
PARAMS = []


def process(img):
    data = img.copy()
    x = np.arange(data.shape[1])
    for i in range(data.shape[0]):
        if np.isnan(data[i, :]).any():
            continue
        try:
            coef = np.polyfit(x, data[i, :], 1)
            data[i, :] -= np.polyval(coef, x)
        except Exception:
            pass  # leave row unchanged on polyfit failure
    return data
