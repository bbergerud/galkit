"""
Methods for working with angles.

Functions
---------
angular_distance(x1, x2, absolute)
    Computes the angular distance between the azimuthal angles x1 and x2

mod_angle(θ, Δθ, deg)
    Adds the angles and returns the modulus such that -π ≤ θ < π
    for radians and -180° ≤ θ < 180° for degrees.
"""
import math
import numpy
import torch
from typing import Union

def angular_distance(
    x1       : Union[float, numpy.ndarray, torch.Tensor], 
    x2       : Union[float, numpy.ndarray, torch.Tensor], 
    absolute : bool = True
) -> Union[float, numpy.ndarray, torch.Tensor]:
    """
    Computes the angular distance between the azimuthal angles x1 and x2

    Parameters
    ----------
    x1 : float, array, tensor
        The first set of azimuthal angles in radians.

    x2 : float, array, tensor
        The second set of azimuthal angles in radians.

    absolute : bool
        Boolean indicating whether to return the absolute value of
        the angular distance (True) or the signed difference (False).
        Default is True.

    Returns
    -------
    Δθ : float, array, tensor
        The angular separation between x1 and x2

    Examples
    --------
    import torch
    from math import pi
    from galkit.functional.angle import angular_distance

    n  = 5
    x1 = torch.ones(n) * pi
    x2 = torch.linspace(-pi, pi, n)
    Δθ = angular_distance(x1, x2)

    for i,j,k in zip(x1, x2, Δθ):
        print(f'Δθ({i},{j}) = {k}')
    """
    is_tensor = isinstance(x1, torch.Tensor) or isinstance(x2, torch.Tensor)
    sin   = torch.sin if is_tensor else numpy.sin
    cos   = torch.cos if is_tensor else numpy.cos
    atan2 = torch.atan2 if is_tensor else numpy.arctan2

    z  = x1 - x2
    dz = atan2(sin(z), cos(z))

    return abs(dz) if absolute else dz

def mod_angle(
    θ   : Union[float, numpy.ndarray, torch.Tensor], 
    Δθ  : Union[float, numpy.ndarray, torch.Tensor], 
    deg : bool = False
) -> Union[float, numpy.ndarray, torch.Tensor]:
    """
    Adds the angles and returns the modulus such that -π ≤ θ < π
    for radians and -180° ≤ θ < 180° for degrees.

    Parameters
    ----------
    θ : float, array, tensor
        The first set of angles to add

    Δθ : float, array, tensor
        The second set of angles to add
    
    deg : bool
        Boolean indicating whether the angles are in degrees (True)
        or radians (False). Default is False.

    Returns
    -------
    θ' : float, array, tensor
        The modulus of θ + Δθ
    
    Examples
    --------
    import torch
    from math import pi
    from galkit.functional.angle import mod_angle

    x = torch.tensor([-pi, 0, pi])
    y = mod_angle(x, pi)
    print(y)
    """
    φ = 180.0 if deg else math.pi
    return ((θ + Δθ + φ) % (2*φ)) - φ