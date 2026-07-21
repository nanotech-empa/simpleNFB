"""Rebin nm -> eV: convert spectral x-axis from wavelength to energy.

Moved verbatim from process_utils.rebin_intensity_nm_to_ev; the Jacobian
(1/dE) is applied so the output has units of counts/eV. See process_utils
shim for the old name.
"""

import numpy as np

NAME = 'Rebin (nm -> eV)'
KIND = 'xy'
DESCRIPTION = 'Convert wavelength (nm) axis to energy (eV); rescales by 1/dE.'
PARAMS = []


def process(x, y):
    wavelengths = np.asarray(x, dtype=float)
    intensities = np.asarray(y, dtype=float)

    center_energies = 1240.0 / wavelengths

    delta_wavelengths = np.abs(np.diff(wavelengths))
    delta_wavelengths = np.insert(delta_wavelengths, 0, delta_wavelengths[0])

    delta_energies = np.array([
        1240.0 / (w - dw / 2.0) - 1240.0 / (w + dw / 2.0)
        for w, dw in zip(wavelengths, delta_wavelengths)
    ])

    return center_energies, intensities / delta_energies
