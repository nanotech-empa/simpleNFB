'''
Created: 14.06.26
Author: amsp
Description: Standalone data processing functions shared by DAT_browser and SXM_browser.
    All functions are pure (no widget or browser state) and operate on numpy arrays.

Functions:
    rebin_intensity_nm_to_ev  -- rebin spectral intensity from wavelength to energy axis
    smooth_data               -- Savitzky-Golay smoothing
    group_average             -- batched median-filtered group averaging
    relative_position         -- tip position relative to scan frame (with angle correction)
    remove_line_average       -- line-by-line linear leveling of 2D image data
'''

import numpy as np
from scipy.signal import savgol_filter, medfilt
from scipy.interpolate import interp1d


def rebin_intensity_nm_to_ev(wavelengths, intensities):
    '''Convert spectral data from wavelength (nm) to energy (eV) axis.

    Each intensity bin is divided by the corresponding energy bin width dE so
    that the result has units of counts/eV (i.e. the Jacobian is applied).

    Parameters
    ----------
    wavelengths : array-like, shape (N,)
        Wavelength values in nm.
    intensities : array-like, shape (N,)
        Intensity values in counts (or any linear unit).

    Returns
    -------
    center_energies : ndarray, shape (N,)
        Photon energies in eV corresponding to each wavelength bin centre.
    rebinned : ndarray, shape (N,)
        Intensity rescaled by 1/dE (units: counts / eV).
    '''
    wavelengths = np.asarray(wavelengths, dtype=float)
    intensities = np.asarray(intensities, dtype=float)

    center_energies = 1240.0 / wavelengths

    delta_wavelengths = np.abs(np.diff(wavelengths))
    delta_wavelengths = np.insert(delta_wavelengths, 0, delta_wavelengths[0])

    delta_energies = np.array([
        1240.0 / (w - dw / 2.0) - 1240.0 / (w + dw / 2.0)
        for w, dw in zip(wavelengths, delta_wavelengths)
    ])

    return center_energies, intensities / delta_energies


def smooth_data(data, window, order):
    '''Apply a Savitzky-Golay filter to 1-D spectral data.

    Parameters
    ----------
    data   : array-like, shape (N,)
    window : int  -- filter window length (must be odd and >= 3)
    order  : int  -- polynomial order (must be < window)

    Returns
    -------
    ndarray, shape (N,)  -- smoothed data
    '''
    return savgol_filter(np.asarray(data, dtype=float), window, order)


def group_average(data, xx, group_size):
    '''Batch spectra into groups and return a median-filtered average per group.

    Outliers within each element-wise group are suppressed with a size-3 median
    filter, and the maximum value is discarded before averaging.

    Parameters
    ----------
    data       : list of ndarray  -- spectral y-values, one array per spectrum
    xx         : list of ndarray  -- corresponding x-values
    group_size : int              -- number of spectra per group

    Returns
    -------
    grouped_x      : list of ndarray  -- one x-array per group (first of each group)
    group_averaged : list of ndarray  -- averaged y-array per group
    '''
    grouped_data = [data[i:i + group_size] for i in range(0, len(data), group_size)]
    grouped_x = [xx[i] for i in range(0, len(xx), group_size)]

    group_averaged = []
    for group in grouped_data:
        median_average = []
        for element_group in zip(*group):
            medians = np.sort(medfilt(element_group, 3))
            medians = medians[:-1]  # drop maximum
            median_average.append(np.average(medians))
        group_averaged.append(np.array(median_average))

    return grouped_x, group_averaged


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


def moving_average(data, window_size):
    '''Uniform (boxcar) moving average via convolution.

    Each output sample is the arithmetic mean of the *window_size* nearest
    input samples, centred on that sample.  The convolution is run in 'same'
    mode so the output is the same length as the input; edge samples where the
    full window does not fit are normalised by the number of samples that
    actually fall within the array (no zero-padding bias).

    Parameters
    ----------
    data        : array-like, shape (N,)
    window_size : int  -- number of points to average (should be odd for a
                          symmetric window; even values are accepted)

    Returns
    -------
    ndarray, shape (N,)  -- smoothed data
    '''
    data = np.asarray(data, dtype=float)
    window_size = max(1, int(window_size))
    kernel = np.ones(window_size)
    # convolve data and a unit array to get per-sample normalisation counts
    smoothed = np.convolve(data, kernel, mode='same')
    counts   = np.convolve(np.ones_like(data), kernel, mode='same')
    return smoothed / counts


def despike_z_score(data, window_size=10, threshold=3.0):
    '''Detect and replace spikes using a localised modified Z-score (Iglewicz-Hoaglin).

    For each sample, a symmetric window of neighbouring points is used to
    estimate the local median and median absolute deviation (MAD). Samples
    whose modified Z-score exceeds *threshold* are replaced with the local
    median.  The modified Z-score is:

        MZ = 0.6745 * |x_i - median| / MAD

    The constant 0.6745 makes MZ comparable to a standard Z-score for
    normally-distributed data (MAD ≈ 0.6745 σ for a Gaussian).

    Parameters
    ----------
    data        : array-like, shape (N,)   -- 1-D spectral data
    window_size : int, default 10          -- half-width of the local window
                                              (full window = 2 * window_size samples)
    threshold   : float, default 3.0       -- modified Z-score cutoff above
                                              which a point is considered a spike

    Returns
    -------
    ndarray, shape (N,)  -- despiked copy of the input (original unchanged)
    '''
    data = np.asarray(data, dtype=float)
    despiked = np.copy(data)
    window_size = int(window_size)

    for i in range(window_size, len(data) - window_size):
        window = data[i - window_size : i + window_size]
        median = np.median(window)
        mad = np.median(np.abs(window - median))

        modified_z = 0.0 if mad == 0 else 0.6745 * abs(data[i] - median) / mad

        if modified_z > threshold:
            despiked[i] = median  # replace spike with local median

    return despiked


def remove_line_average(image_data):
    '''Line-by-line linear leveling of 2-D scan data.

    Fits a first-order polynomial to each row (ignoring NaN rows) and
    subtracts it, removing tilt and slow drift on a per-line basis.

    Parameters
    ----------
    image_data : ndarray, shape (M, N)  -- raw 2-D scan data (may contain NaNs)

    Returns
    -------
    ndarray, shape (M, N)  -- leveled data (input is not modified in-place)
    '''
    data = image_data.copy()
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
