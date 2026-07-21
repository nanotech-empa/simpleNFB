'''
Created: 14.06.26. Reduced to a re-export shim: 21.07.26 (DYNAMIC_PIPELINE_PLAN.md
Phase 1). The functions below now live one-per-file under simpleNFB/processes/
(NAME/KIND/PARAMS/process contract, used by the pipeline UI). This module keeps
old notebook/`simpleNFB.__init__` imports working by adapting each new
process(...) signature back to its original call shape.

`relative_position` is NOT part of the pipeline (no KIND fits it) and stays
here as the only real implementation.

Functions (all delegate to simpleNFB.processes.<stem> except relative_position):
    rebin_intensity_nm_to_ev  -- processes.rebin_nm_to_ev
    smooth_data               -- processes.smooth_savgol
    group_average             -- processes.group_average (arg order adapted)
    relative_position         -- unchanged, pure, lives here
    remove_line_average       -- processes.remove_line_average
    despike_z_score           -- processes.despike_z_score
    moving_average            -- processes.moving_average
'''

import numpy as np

from .processes.rebin_nm_to_ev import process as _rebin_process
from .processes.smooth_savgol import process as _smooth_process
from .processes.group_average import process as _group_average_process
from .processes.remove_line_average import process as _remove_line_average_process
from .processes.despike_z_score import process as _despike_process
from .processes.moving_average import process as _moving_average_process


def rebin_intensity_nm_to_ev(wavelengths, intensities):
    '''Convert spectral data from wavelength (nm) to energy (eV) axis. See
    simpleNFB.processes.rebin_nm_to_ev for the implementation.'''
    return _rebin_process(wavelengths, intensities)


def smooth_data(data, window, order):
    '''Savitzky-Golay smoothing. See simpleNFB.processes.smooth_savgol.'''
    return _smooth_process(data, window_size=window, order=order)


def group_average(data, xx, group_size):
    '''Batch spectra into groups and return a median-filtered average per
    group. Old argument order was (ys, xs, n); the process() contract takes
    (xs, ys, **params) -- adapted here, see simpleNFB.processes.group_average.'''
    return _group_average_process(xx, data, group_size=group_size)


def relative_position(img, spec):
    '''Compute tip position in scan-frame coordinates (nm).

    Accounts for scan rotation so that the returned coordinates map correctly
    onto an imageBrowser axes whose extent is [0, width, 0, height].

    Parameters
    ----------
    img  : Spm object  -- reference scan image (provides offset, size, angle)
    spec : Spm object  -- spectrum whose tip position is queried

    Returns
    -------
    [x_rel, y_rel] : list of float  -- position in nm within the scan frame
    '''
    [o_x, o_y] = img.get_param('scan_offset')
    width  = img.get_param('width')[0]
    height = img.get_param('height')[0]
    [o_x, o_y] = [o_x * 1e9, o_y * 1e9]

    angle   = float(img.get_param('scan_angle')) * -1.0 * np.pi / 180.0
    x_spec  = spec.get_param('x')[0]
    y_spec  = spec.get_param('y')[0]

    if angle != 0:
        x_rel = (x_spec - o_x) * np.cos(angle) + (y_spec - o_y) * np.sin(angle) + width  / 2.0
        y_rel = -(x_spec - o_x) * np.sin(angle) + (y_spec - o_y) * np.cos(angle) + height / 2.0
    else:
        x_rel = x_spec - o_x + width  / 2.0
        y_rel = y_spec - o_y + height / 2.0

    return [x_rel, y_rel]


def remove_line_average(image_data):
    '''Line-by-line linear leveling of 2-D scan data. See
    simpleNFB.processes.remove_line_average.'''
    return _remove_line_average_process(image_data)


def despike_z_score(data, window_size=10, threshold=3.0):
    '''Modified Z-score spike removal. See simpleNFB.processes.despike_z_score.'''
    return _despike_process(data, window_size=window_size, threshold=threshold)


def moving_average(data, window_size):
    '''Boxcar moving average. See simpleNFB.processes.moving_average.'''
    return _moving_average_process(data, window_size=window_size)
