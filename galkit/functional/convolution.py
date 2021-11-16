"""
Profiles for modeling the point-spread function (PSF).

Classes
-------
Gaussian
    Class interface to the gaussian function. Also stores the
    FWHM of the profile.

Moffat
    Class interface to the moffat function. Also stores the
    FWHM of the profile.

DoubleGaussianPowerlaw
    Class interface to the double_guassian_powerlaw function.
    Also stores the FWHM of the profile.

Functions
---------
get_kernel2d(kernel_size, profile, device, oversample)
    Returns a 2D kernel representation of the provided profile.

get_kernel2d_fwhm(profile, default_size, device, oversample)
    Returns a 2D kernel representation of the provided profile
    out to the indicated number of FWHM.

double_gaussian_powerlaw(r, b, beta, p0, sigma1, sigma2, sigmaP, normalize)
    Double gaussian + powerlaw profile used by SDSS for modeling the PSF,
        f(r) = [exp(-r²/(2σ₁²)) + b⋅exp(-r²/(2σ₂²)) + p₀[1 + r²/(β⋅σₚ²)]^(-β/2)] / (1 + b + p0)

gaussian(r, sigma, normalize)
    Gaussian profile,
        1/sqrt(2πσ²) ⋅ exp(-r² / (2σ²))

moffat(r, core_width, power_index, normalize)
    Moffat Profile,
        f(r;α,β) = (β-1)/(πα²)⋅[1 + (r²/α²)]⁻ᵝ
    where α is the core width of the profile and β the power index.
"""

import math
import numpy
import torch
from typing import Optional, Tuple, Union
from dataclasses import dataclass
from ..spatial import resample
from ..utils import parse_parameter, round_up_to_odd_integer

@dataclass
class DoubleGaussianPowerlaw:
    """
    Class interface to the double_guassian_powerlaw function

    Methods
    -------
    __call__(r)

    Properties
    ----------
    fwhm
        Returns the fwhm of the profile.
    """
    b         : float
    beta      : float
    p0        : float
    sigma1    : float
    sigma2    : float
    sigmaP    : float
    normalize : bool = False

    def __call__(self, r, *args, **kwargs):
        return double_gaussian_powerlaw(r, **self.__dict__)

    @property
    def fwhm(self):
        return 0.5 * math.sqrt(math.pi) * (
            math.sqrt(2) * (self.sigma1 + self.b*self.sigma2) +
            math.sqrt(self.beta) * self.p0 * self.sigmaP * math.gamma(0.5*(self.beta-1) / math.gamma(0.5*self.beta))
        )

@dataclass
class Gaussian:
    """
    Class interface to the gaussian function.
    """
    sigma     : float
    normalize : bool = False

    def __call__(self, r, *args, **kwargs):
        return gaussian(r=r, **self.__dict__)

    @property
    def fwhm(self):
        return Gaussian.std2fwhm(self.sigma)

    @staticmethod
    def fwhm2std(fwhm):
        """
        Converts the full width at half maxmium of a gaussian function
        to the standard deviation.

        Parameters
        ----------
        fwhm : float
            The full width at half maximum of the gaussian.

        Returns
        -------
        std : float
            The standard deviation of the Gaussian.

        Examples
        --------
        import matplotlib.pyplot as plt
        import numpy
        from galkit.functional import Gaussian
        from scipy.stats import norm

        fwhm = 2
        N = norm(0, Gaussian.fwhm2std(fwhm))
        x = numpy.linspace(-3, 3)
        y = N.pdf(x) / N.pdf(0)

        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.hlines(0.5, -fwhm/2, fwhm/2, colors='orange')
        fig.show()
        """
        return fwhm / math.sqrt(8 * math.log(2))

    @staticmethod
    def std2fwhm(std):
        """
        Converts the standard deviation of a gaussian function to
        the full width at half maximum.

        Parameters
        ----------
        std : float
            The standard deviation of the Gaussian.
        
        Returns
        -------
        fwhm : float
            The full width at half maximum of the gaussian.
        
        Examples
        --------
        import matplotlib.pyplot as plt
        import numpy
        from galkit.functional import Gaussian
        from scipy.stats import norm

        sigma = 1
        fwhm = Gaussian.std2fwhm(sigma)
        N = norm(0, sigma)
        x = numpy.linspace(-3,3)
        y = N.pdf(x) / N.pdf(0)
        
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.hlines(0.5, -fwhm/2, fwhm/2, colors='orange')
        fig.show()
        """
        return std * math.sqrt(8 * math.log(2))

@dataclass
class Moffat:
    """
    Class interface to the moffat function. Also stores the
    FWHM of the profile.
    """
    core_width  : float
    power_index : float
    normalize   : bool = False

    def __call__(self, r, *args, **kwargs):
        return moffat(r, **self.__dict__)

    @property
    def fwhm(self):
        return 2 * self.core_width * (2**(1/self.power_index) - 1)**0.5

def double_gaussian_powerlaw(
    r         : Union[numpy.ndarray, torch.Tensor],
    b         : float,
    beta      : float,
    p0        : float,
    sigma1    : float,
    sigma2    : float,
    sigmaP    : float,
    normalize : bool = False,
) -> Union[numpy.ndarray, torch.Tensor]:
    """
    Double gaussian + powerlaw profile used by SDSS for modeling the PSF,

        f(r) = [exp(-r²/(2σ₁²)) + b⋅exp(-r²/(2σ₂²)) + p₀[1 + r²/(β⋅σₚ²)]^(-β/2)] / (1 + b + p0)

    Parameters
    ----------
    r : array, tensor
        The radial distances.
    
    b : float
        Ratio of the inner to outer PSF at the origin.

    beta : float
        Slope of the powerlaw.

    p0 : float
        Value of the powerlaw at the origin.

    sigma1 : float
        Inner gaussian sigma for the composite fit.
    
    sigma2 : float
        Outer gaussian sigma for the composite fit.
    
    sigmaP : float
        Width parameter for the powerlaw.

    normalize : bool
        Boolean indicating whether to normalize the values
        so the expression sums to 1. Assumes a 2D distribution.
        Note that this applies just to the analytical form
        of the expression.

    Returns
    -------
    psf : array, tensor
        The value of the psf at the provided radial distances.

    Examples
    --------
    import matplotlib.pyplot as plt
    from galkit.functional import double_gaussian_powerlaw
    from galkit.spatial import grid, coordinate

    shape = (128,128)
    _, r = coordinate.polar(
        grid = grid.pixel_grid(shape),
        h0 = (shape[0] - 1) / 2,
        w0 = (shape[1] - 1) / 2,
    )

    p = double_gaussian_powerlaw(r, 
        b=0.3, 
        beta=3, 
        p0=0.5, 
        sigma1=1, 
        sigma2=2, 
        sigmaP=1, 
        normalize=True
    )
    print(p.sum())

    fig, ax = plt.subplots()
    ax.imshow(p.squeeze())
    fig.show()
    """
    exp = torch.exp if isinstance(r, torch.Tensor) else numpy.exp

    r_sq  = r**2
    sigma1_sq = sigma1**2
    sigma2_sq = sigma2**2
    sigmaP_sq = sigmaP**2

    num = exp(-r_sq / (2*sigma1_sq)) \
        + b * exp(-r_sq / (2*sigma2_sq)) \
        + p0 * (1 + r_sq / (beta * sigmaP_sq))**(-0.5*beta)
    den = 1 + b + p0

    out = num / den

    if normalize:
        intval = (2*math.pi/den) * (sigma1_sq + b*sigma2_sq + p0*beta*sigmaP_sq/(beta - 2))
        out = out / intval
    
    return out

def gaussian(
    r         : Union[numpy.ndarray, torch.Tensor],
    sigma     : float,
    normalize : bool = False,
) -> Union[numpy.ndarray, torch.Tensor]:
    """
    Gaussian profile

        1/sqrt(2πσ²) ⋅ exp(-r² / (2σ²))

    Parameters
    ----------
    r : array, tensor
        The radial distances.
    
    sigma : float
        The standard deviation of the profile.
    
    normalize : bool
        Boolean indicating whether to normalize the values
        so the expression sums to 1. Assumes a 2D distribution.
        Note that this applies just to the analytical form
        of the expression.

    Returns
    -------
    psf : array, tensor
        The value of the psf at the provided radial distances.

    Examples
    --------
    import matplotlib.pyplot as plt
    from galkit.functional import gaussian
    from galkit.spatial import grid, coordinate

    shape = (128,128)
    _, r = coordinate.polar(
        grid = grid.pixel_grid(shape),
        h0 = (shape[0] - 1) / 2,
        w0 = (shape[1] - 1) / 2,
    )

    p = gaussian(r, 
        sigma = 3,
        normalize=True
    )
    print(p.sum())

    fig, ax = plt.subplots()
    ax.imshow(p.squeeze())
    fig.show()
    """
    exp = torch.exp if isinstance(r, torch.Tensor) else numpy.exp
    out =  exp(-0.5 * (r / sigma)**2)
    if normalize:
        out = out / (2 * math.pi * sigma**2)
    return out

def moffat(
    r           : Union[numpy.ndarray, torch.Tensor],
    core_width  : float,
    power_index : float,
    normalize   : bool = False,
) -> Union[numpy.ndarray, torch.Tensor]:
    """
    Moffat Profile,

        f(r;α,β) = (β-1)/(πα²)⋅[1 + (r²/α²)]ᵝ

    where α is the core width of the profile and β the power index.

    Parameters
    ----------
    r : array, tensor
        The radial distances.

    core_width : float
        The core width of the profile.
    
    power_index : float
        The power index of the profile.

    normalize : bool
        Boolean indicating whether to normalize the values
        so the expression sums to 1. Assumes a 2D distribution.
        Note that this applies just to the analytical form
        of the expression.

    Returns
    -------
    psf : array, tensor
        The value of the psf at the provided radial distances.

    Examples
    --------
    import matplotlib.pyplot as plt
    from galkit.functional import moffat
    from galkit.spatial import grid, coordinate

    shape = (128,128)
    _, r = coordinate.polar(
        grid = grid.pixel_grid(shape),
        h0 = (shape[0] - 1) / 2,
        w0 = (shape[1] - 1) / 2,
    )

    p = moffat(r, 
        core_width = 3,
        power_index = 3,
        normalize=True
    )
    print(p.sum())

    fig, ax = plt.subplots()
    ax.imshow(p.squeeze())
    fig.show()
    """
    out = (1 + (r / core_width)**2)**(-power_index)
    if normalize:
        out = out * (power_index - 1) / (math.pi * core_width**2)
    return out

def get_kernel2d(
    kernel_size : Union[int, Tuple[int,int]],
    profile     : callable,
    device      : Optional[torch.device] = None,
    oversample  : int = 3,
):
    """
    Returns a 2D kernel representation of the provided profile.

    Parameters
    ----------
    kernel_size : int, Tuple[int,int]
        The size of the kernel. If an integer, then this is used
        for both dimensions of the kernel.

    profile : callable
        The profile to use. It should take as input the radial
        positions (r) as well as the vertical (x) and horizontal
        (y) positions.

    device : torch.device, optional
        The device to generate the data on.
    
    oversample : int
        The oversampling factor for constructing the kernel.

    Returns
    -------
    kernel : tensor (H × W)
        A 2D tensor representing the profile.
    
    Examples
    --------
    import matplotlib.pyplot as plt
    from galkit.functional import Gaussian, get_kernel2d

    kernel = get_kernel2d(
        kernel_size = (11,11),
        profile = Gaussian(1),
    )
    
    fig, ax = plt.subplots()
    ax.imshow(kernel)
    fig.show()
    """
    ksize_x, ksize_y = parse_parameter(kernel_size, 2)

    x = torch.linspace(0, ksize_x-1, ksize_x * oversample, device=device) - (ksize_x-1)/2
    y = torch.linspace(0, ksize_y-1, ksize_y * oversample, device=device) - (ksize_y-1)/2
    x, y = torch.meshgrid(x,y)    

    r = (x**2 + y**2).sqrt()

    kernel = profile(r=r, x=x, y=y)

    if oversample > 1:
        kernel = resample.downscale_local_mean(kernel.unsqueeze(0), oversample).squeeze()

    return kernel / kernel.sum()

def get_kernel2d_fwhm(
    profile      : callable,
    default_size : int = 4,
    device       : Optional[torch.device] = None,
    oversample   : int = 3,
):
    """
    Returns a 2D kernel representation of the provided profile
    out to the indicated number of FWHM.

    Parameters
    ----------
    profile : callable
        The profile to use. It should take as input the radial
        positions (r) as well as the vertical (x) and horizontal
        (y) positions. The FWHM parameter should be stored as
        profile.fwhm.

    default_size : int
        The number of FWHM for constructing the kernel.

    device : torch.device, optional
        The device to generate the data on.
    
    oversample : int
        The oversampling factor for constructing the kernel.

    Returns
    -------
    kernel : tensor (H × W)
        A 2D tensor representing the profile.
    
    Examples
    --------
    import matplotlib.pyplot as plt
    from galkit.functional import Gaussian, get_kernel2d_fwhm

    kernel = get_kernel2d_fwhm(
        profile = Gaussian(1),
    )
    
    fig, ax = plt.subplots()
    ax.imshow(kernel)
    fig.show()
    """
    ksize = round_up_to_odd_integer(profile.fwhm * default_size)
    return get_kernel2d(kernel_size=(ksize, ksize), profile=profile, device=device, oversample=oversample)