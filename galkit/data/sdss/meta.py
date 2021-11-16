"""
Storage of metadata related to SDSS

Variables
---------
sdss_λ / sdss_wavelengths : Dict[str, float]
    Dictionary whose key represents the SDSS filter band and whose value
    contains the effective wavelength of the filter band in meters.
"""

sdss_λ = sdss_wavelengths = {
    'u': 3543e-10,
    'g': 4770e-10,
    'r': 6231e-10,
    'i': 7625e-10,
    'z': 9134e-10,
}