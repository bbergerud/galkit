"""
Methods for converting between flux and magnitude.

Functions
---------
pogson_flux2mag(flux, filter_band, m0)
    Converts flux to Pogson magnitudes.

pogson_mag2flux(magnitude, filter_band, m0)
    Converts Pogson magnitudes to flux.

sdss_flux2mag(flux, filter_band)
    Converts nanomaggies to arcsinh magnitudes.

sdss_mag2flux(magnitude, filter_band)
    Converts arcsinh magnitudes to nanomaggies.

Variables
----------
sdss_b : Dict[str, float]
    Dictionary whose key represents the SDSS filter band and whose value
    contains the `b` parameter used when constructing the arcsinh magnitude.
"""
import math
import numpy
import torch
from typing import Optional, Union

sdss_b = {
    'u': 1.4e-10,
    'g': 0.9e-10,
    'r': 1.2e-10,
    'i': 1.8e-10,
    'z': 7.4e-10,
}

def pogson_flux2mag(
    flux        : Union[numpy.ndarray, torch.Tensor], 
    filter_band : Optional[str] = None,
    m0          : float = 22.5
) -> Union[numpy.ndarray, torch.Tensor]:
    """
    Converts flux to Pogson magnitudes.

    Parameters
    ----------
    flux : Array, Tensor
        The flux of the object.

    filter_band : str, optional
        The filter band. Not used, but included for compatibility
        with sdss_flux2mag.

    m0 : float
        The reference magnitude. Default is 22.5.

    Returns
    -------
    magnitude : Array, Tensor
        The magnitude of the object.

    Examples
    --------
    from galkit.functional.magnitude import pogson_flux2mag
    print(pogson_flux2mag(1)) 
    """
    log10 = torch.log10 if isinstance(flux, torch.Tensor) else numpy.log10
    return m0 - 2.5 * log10(flux)

def pogson_mag2flux(
    magnitude   : Union[numpy.ndarray, torch.Tensor], 
    filter_band : Optional[str] = None,
    m0          : float = 22.5,
) -> Union[numpy.ndarray, torch.Tensor]:
    """
    Converts Pogson magnitudes to flux.

    Parameters
    ----------
    magnitude : Array, Tensor
        The magnitude of the object.

    filter_band : str, optional
        The filter band. Not used, but included for compatibility
        with sdss_mag2flux.

    m0 : float
        The reference magnitude. Default is 22.5.

    Returns
    -------
    flux : Array, Tensor
        The flux of the object.

    Examples
    --------
    from galkit.functional.magnitude import pogson_mag2flux
    print(pogson_mag2flux(22.5))
    """
    return 10**(-(magnitude-m0) / 2.5)

def sdss_flux2mag(
    flux        : Union[numpy.ndarray, torch.Tensor], 
    filter_band : str,
) -> Union[numpy.ndarray, torch.Tensor]:
    """
    Converts nanomaggies to arcsinh magnitudes.

    Parameters
    ----------
    flux : Array, Tensor
        The flux of the object in nanomaggies.

    filter_band : str
        The filter band. Can be one of {'u', 'g', 'r', 'i', 'z'}.

    Returns
    -------
    magnitude : Array, Tensor
        The arcsinh magnitude of the object.

    Examples
    --------
    from galkit.functional.magnitude import sdss_flux2mag
    print(sdss_flux2mag(1, 'r'))
    """
    arcsinh = torch.arcsinh if isinstance(flux, torch.Tensor) else numpy.arcsinh

    f  = flux * 1e-9                 # Convert to maggies
    b  = sdss_b[filter_band]
    t1 = arcsinh(f / (2*b))
    t2 = math.log(b)
    return -2.5 / math.log(10) * (t1 + t2)

def sdss_mag2flux(
    magnitude   : Union[numpy.ndarray, torch.Tensor], 
    filter_band : str
) -> Union[numpy.ndarray, torch.Tensor]:
    """
    Converts arcsinh magnitudes to nanomaggies.

    Parameters
    ----------
    magnitude : Tensor
        The arcsinh magnitude of the object.

    filter_band : str
        The filter band. Can be one of {'u', 'g', 'r', 'i', 'z'}.

    Returns
    -------
    flux : Tensor
        The flux of the object in nanomaggies.

    Examples
    --------
    from galkit.functional.magnitude import sdss_mag2flux
    print(sdss_mag2flux(22.5, 'r'))
    """
    sinh = torch.sinh if isinstance(magnitude, torch.Tensor) else numpy.sinh

    b = sdss_b[filter_band]
    t = math.log(b) + magnitude * math.log(10)/2.5
    t = sinh(-t)
    flux = 2*b*t
    return flux * 1e9               # Convert to nanomaggies