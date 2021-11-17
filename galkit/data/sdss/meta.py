"""
Storage of metadata related to SDSS

Variables
---------
sdss_nMgyPerCount : Dict[str, float]
    Dictionary whose key represents the SDSS filter band and whose value
    is the mean nanomaggies / count.

sdss_λ / sdss_wavelengths : Dict[str, float]
    Dictionary whose key represents the SDSS filter band and whose value
    contains the effective wavelength of the filter band in meters.
"""

sdss_nMgyPerCount = {
    'u': 0.0101,
    'g': 0.0038,
    'r': 0.0051,
    'i': 0.0066,
    'z': 0.0326,
}

sdss_λ = sdss_wavelengths = {
    'u': 3543e-10,
    'g': 4770e-10,
    'r': 6231e-10,
    'i': 7625e-10,
    'z': 9134e-10,
}