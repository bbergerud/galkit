"""
Photometric profiles

Functions
---------
exponential(r, amplitude, scale)
    Exponential disk profile,
        I(r) = amplitude * exp(-r / scale)

exponential_break(r, amplitude, scale_inner, scale_outer, breakpoint)
    Exponential profile with a breakpoint,
        I(r) = amplitude1 ⋅ exp(-r / scale_inner) ⋅ Θ(r <= breakpoint) 
             + amplitude2 ⋅ exp(-r / scale_outer) ⋅ Θ(r > breakpoint)
    where Θ is the heaviside step function. The amplitude of the second
    component is calculated to ensure the function is continuous.

ferrer(r, amplitude, scale, index)
    Ferrer ellipsoid used for modelling bar profiles. The functional
    form of the expression is     
        I(r) = amplitude ⋅ [1 - (r / scale)²]^(index + 0.5)     [r < scale]
    where I(r > scale) = 0.

ferrer_modified(r, amplitude, scale, alpha, beta)
    Modified version of the Ferrer function,
        I(r) = amplitude ⋅ [1 - (r / scale)²⁻ᵝ]ᵅ        [r < scale]
    where I(r > scale) = 0.

sersic(r, amplitude, scale, index)
    Sersic profile,
        I(r) = amplitude * exp(-bₙ⋅[(r/scale)^(1/index) - 1])
    where bₙ is the solution to Γ(2n)/2 = γ(2n,bₙ). The parameters
    are specified in terms of the half-light radius. Note that this
    reduces to the exponential profile when index=1.
"""
import math
import numpy
import torch
from scipy.special import gammaincinv
from typing import Union

def exponential(
    r         : Union[numpy.ndarray, torch.Tensor], 
    amplitude : float, 
    scale     : float,
) -> Union[numpy.ndarray, torch.Tensor]:
    """
    Exponential disk profile,

        I(r) = amplitude * exp(-r / scale)

    Parameters
    ----------
    r : array, tensor
        The radial distances

    amplitude : float
        The amplitude at r=0

    scale : float
        The scale length

    Returns
    -------
    flux : array, tensor
        The profile flux.
    
    Examples
    --------
    import matplotlib.pyplot as plt
    from galkit.functional.photometric import exponential
    from galkit.spatial import coordinate, grid

    θ, r = coordinate.polar(
        grid = grid.pytorch_grid(100,100),
    )

    flux = exponential(r=r, amplitude=1, scale=0.2)

    fig, ax = plt.subplots()
    ax.imshow(flux.squeeze())
    fig.show()
    """
    exp = torch.exp if isinstance(r, torch.Tensor) else numpy.exp
    return amplitude * exp(-r / scale)

def exponential_break(
    r           : Union[numpy.ndarray, torch.Tensor], 
    amplitude   : float, 
    scale_inner : float,
    scale_outer : float,
    breakpoint  : float,
) -> Union[numpy.ndarray, torch.Tensor]:
    """
    Exponential profile with a breakpoint,

        I(r) = amplitude1 ⋅ exp(-r / scale_inner) ⋅ Θ(r <= breakpoint) 
             + amplitude2 ⋅ exp(-r / scale_outer) ⋅ Θ(r > breakpoint)

    where Θ is the heaviside step function. The amplitude of the second
    component is calculated to ensure the function is continuous.

    Parameters
    ----------
    r : array, tensor
        The radial distances

    amplitude : float
        The amplitude at r=0 for the first component.

    scale_inner : float
        The scale length of the first component.
    
    scale_outer : float
        The scale length of the outer component.

    breakpoint : float
        The breakpoint at which the profile changes from
        the inner to the outer component.

    Returns
    -------
    flux : array, tensor
        The profile flux.
    
    Examples
    --------
    import matplotlib.pyplot as plt
    from galkit.functional.photometric import exponential_break
    from galkit.spatial import coordinate, grid

    θ, r = coordinate.polar(
        grid = grid.pytorch_grid(100,100),
    )

    flux = exponential_break(r=r, amplitude=1, scale_inner=0.2, scale_outer=0.1, breakpoint=0.2)

    fig, ax = plt.subplots()
    ax.imshow(flux.squeeze())
    fig.show()
    """
    is_tensor = isinstance(r, torch.Tensor)
    exp   = torch.exp if is_tensor else numpy.exp
    empty = torch.empty_like if is_tensor else numpy.empty_like

    amplitude2 = amplitude*math.exp(breakpoint * (1/scale_outer - 1/scale_inner))

    mask_inner = r < breakpoint
    mask_outer = ~mask_inner

    flux = empty(r)
    flux[mask_inner] = amplitude  * exp(-r[mask_inner] / scale_inner)
    flux[mask_outer] = amplitude2 * exp(-r[mask_outer] / scale_outer)

    return flux

def ferrer(
    r         : Union[numpy.ndarray, torch.Tensor], 
    amplitude : float, 
    scale     : float,
    index     : float = 2,
) -> Union[numpy.ndarray, torch.Tensor]:
    """
    Ferrer ellipsoid used for modelling bar profiles. The functional
    form of the expression is     

        I(r) = amplitude ⋅ [1 - (r / scale)²]^(index + 0.5)     [r < scale]

    where I(r > scale) = 0.

    Parameters
    ----------
    r : array, tensor
        The radial distances

    amplitude : float
        The amplitude at r=0

    scale : float
        The length of the ellipsoid.

    index : float
        The power index. Default is 2.

    Returns
    -------
    flux : array, tensor
        The profile flux.
    
    Examples
    --------
    import matplotlib.pyplot as plt
    from galkit.functional.photometric import ferrer
    from galkit.spatial import coordinate, grid

    θ, r = coordinate.polar(
        grid = grid.pytorch_grid(100,100),
        q = 0.5
    )

    flux = ferrer(r=r, amplitude=1, scale=0.5, index=2)

    fig, ax = plt.subplots()
    ax.imshow(flux.squeeze())
    fig.show()
    """
    is_tensor = isinstance(r, torch.Tensor)
    zeros_like = torch.zeros_like if is_tensor else numpy.zeros_like

    mask = r < scale

    flux = zeros_like(r)
    flux[mask] = amplitude * (1 - (r[mask] / scale)**2)**(index + 0.5)

    return flux

def ferrer_modified(
    r         : Union[numpy.ndarray, torch.Tensor], 
    amplitude : float, 
    scale     : float, 
    alpha     : float = 2.5, 
    beta      : float = 2
) -> Union[numpy.ndarray, torch.Tensor]:
    """
    Modified version of the Ferrer function,

        I(r) = amplitude ⋅ [1 - (r / scale)²⁻ᵝ]ᵅ        [r < scale]
 
    where I(r > scale) = 0.

    Parameters
    ----------
    r : array, tensor
        The radial distances

    amplitude : float
        The amplitude at r=0

    scale : float
        The length of the ellipsoid.

    alpha : float
        The outer power index. Default is 0.

    beta : float
        The inner power index. Default is 2.5

    Returns
    -------
    flux : array, tensor
        The profile flux.
    
    Examples
    --------
    import matplotlib.pyplot as plt
    from galkit.functional.photometric import ferrer_modified
    from galkit.spatial import coordinate, grid

    θ, r = coordinate.polar(
        grid = grid.pytorch_grid(100,100),
        q = 0.5
    )

    flux = ferrer_modified(r=r, amplitude=1, scale=0.5, alpha=2.5, beta=-1)

    fig, ax = plt.subplots()
    ax.imshow(flux.squeeze())
    fig.show()  
    """
    is_tensor = isinstance(r, torch.Tensor)
    zeros_like = torch.zeros_like if is_tensor else numpy.zeros_like

    mask = r < scale

    flux = zeros_like(r)
    flux[mask] = amplitude * (1 - (r[mask]/scale)**(2-beta))**alpha
    return flux

def sersic(
    r         : Union[numpy.ndarray, torch.Tensor], 
    amplitude : float, 
    scale     : float,
    index     : float,
) -> Union[numpy.ndarray, torch.Tensor]:
    """
    Sersic profile,

        I(r) = amplitude * exp(-bₙ⋅[(r/scale)^(1/index) - 1])
    
    where bₙ is the solution to Γ(2n)/2 = γ(2n,bₙ). The parameters
    are specified in terms of the half-light radius. Note that this
    reduces to the exponential profile when index=1.

    Parameters
    ----------
    r : array, tensor
        The radial distances

    amplitude : float
        The amplitude at the half-light radius.

    scale : float
        The half-light radius

    index : float
        The power index.

    Returns
    -------
    flux : array, tensor
        The profile flux.
    
    Examples
    --------
    import matplotlib.pyplot as plt
    from galkit.functional.photometric import sersic
    from galkit.spatial import coordinate, grid

    θ, r = coordinate.polar(
        grid = grid.pytorch_grid(100,100),
        q = 0.5
    )

    flux = sersic(r=r, amplitude=1, scale=0.5, index=2)

    fig, ax = plt.subplots()
    ax.imshow(flux.squeeze())
    fig.show()
    """
    is_tensor = isinstance(r, torch.Tensor)
    exp = torch.exp if is_tensor else numpy.exp

    if isinstance(index, torch.Tensor):
        index = index.cpu().item()
    
    bn   = gammaincinv(2*index, 0.5)    # Solves Γ(2n)/2 = γ(2n,bn)
    t    = (r / scale)**(1/index) - 1
    flux = amplitude * exp(-bn * t)

    return flux
