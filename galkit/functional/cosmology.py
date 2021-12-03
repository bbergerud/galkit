"""
Methods of cosmological significance.

Functions
---------
get_distance_Mpc(z, cosmology)
    Returns the luminosity distance in Mpc based on the provided cosmology.

get_distance_kpc(z, cosmology)
    Returns the luminosity distance in kpc based on the provided cosmology.

redshift_image(input, z_in, z_out, seeing, plate_scale, cosmology)
    Roughly transforms input image to what it would look like at the
    given redshift. No color correction factors are applied.
        If z_in < z_out, then a simple gaussian convolution is performed to 
    correct for the change in seeing.
"""

import numpy
import torch
import torch.nn.functional as F
from astropy.cosmology import FlatLambdaCDM
from astropy.constants import c
from kornia.filters import gaussian_blur2d
from typing import Union
from .convolution import Gaussian
from ..utils import round_up_to_odd_integer

def get_distance_Mpc(z:float, cosmology=FlatLambdaCDM(70, 0.3)):
    """
    Returns the luminosity distance in Mpc based on the provided cosmology.
    """
    return cosmology.luminosity_distance(z).value

def get_distance_kpc(z:float, cosmology=FlatLambdaCDM(70, 0.3)):
    """
    Returns the luminosity distance in kpc based on the provided cosmology.
    """
    return get_distance_Mpc(z,cosmology) * 1e3

def redshift_image(
    input  : Union[numpy.ndarray, torch.Tensor],
    z_in   : Union[callable, float],
    z_out  : Union[callable, float],
    seeing : Union[callable, float] = 1.5,
    plate_scale : float = 0.396,
    cosmology = FlatLambdaCDM(70.0, 0.3)
) -> Union[numpy.ndarray, torch.Tensor]:
    """
    Roughly transforms input image to what it would look like at the
    given redshift. No color correction factors are applied.

    If z_in < z_out, then a simple gaussian convolution is performed to 
    correct for the change in seeing.

    Parameters
    ----------
    input : array, tensor
        The input image array. Should have dimensions of either 
        (B × C × H × W) or (C × H × W).
    
    z_in : callable, float
        The reference redshift of the input image. If callable,
        it should return a float value.
    
    z_out : callable, float
        The output redshift of the transformed image. If callable,
        it should return a float value.

    seeing : callable, float
        The FWHM of the PSF in arcseconds. If callable, it should
        return a float value.

    plate_scale : float
        The plate scale of the image in arcseconds / pixel.
    
    cosmology
        The astropy cosmology model. Should have a luminosity_distance
        method.

    Returns
    -------
    output : array, tensor
        The redshifted image.
    """
    if callable(z_in): z_in = z_in()
    if callable(z_out): z_out = z_out()
    if callable(seeing): seeing = seeing()

    d_i = get_distance_Mpc(z_in, cosmology)
    d_o = get_distance_Mpc(z_out, cosmology)
    
    flux_scale   = ((1 + z_in) / (1 + z_out))**4
    image_scale  = (d_i / (1 + z_in)**2) / (d_o / (1 + z_out)**2)
    kernel_fwhm  = (seeing / plate_scale) * (1 - image_scale**2)**0.5
    kernel_sigma = Gaussian.fwhm2std(kernel_fwhm)

    is_tensor = isinstance(input, torch.Tensor)
    if not is_tensor:
        input = torch.tensor(input)

    ndim = input.ndim
    if ndim not in [3,4]:
        raise ValueError("input must have a BCHW or a CHW format.")
    
    output = F.interpolate(
        input = input if input.ndim == 4 else input.unsqueeze(0),
        scale_factor = image_scale,
        align_corners = True,
        mode = 'bilinear',
    ) * (flux_scale / image_scale**2)

    if image_scale < 1:
        kernel_size = int(round(kernel_sigma * 3))
        kernel_size = round_up_to_odd_integer(min(3, kernel_size))
        kernel_size = (kernel_size, kernel_size)
        sigma = (kernel_sigma, kernel_sigma)
        output = gaussian_blur2d(output, kernel_size=kernel_size, sigma=sigma)

    if ndim == 3:
        input = input.squeeze(0)
    
    return output if is_tensor else output.numpy()