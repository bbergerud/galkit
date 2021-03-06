"""
Methods for working with angles.

Functions
---------
angular_distance(x1, x2, absolute)
    Computes the angular distance between the azimuthal angles x1 and x2

circular_mean(input, weights)
    Computes the (weighted) circular mean of a collection of angles.

mod_angle(θ, Δθ, deg)
    Adds the angles and returns the modulus such that -π ≤ θ < π
    for radians and -180° ≤ θ < 180° for degrees.
"""
import math
import numpy
import torch
from typing import Optional, Union

def angular_distance(
    x1       : Union[float, numpy.ndarray, torch.Tensor], 
    x2       : Union[float, numpy.ndarray, torch.Tensor], 
    absolute : bool = True,
    power    : Optional[float] = None,
) -> Union[float, numpy.ndarray, torch.Tensor]:
    """
    Computes the angular distance between the azimuthal angles x1 and x2
    where the second angle serves as the reference.

    Parameters
    ----------
    x1 : float, array, tensor
        The first set of azimuthal angles in radians.

    x2 : float, array, tensor
        The reference set of azimuthal angles in radians.

    absolute : bool
        Boolean indicating whether to return the absolute value of
        the angular distance (True) or the signed difference (False).
        Default is True.

    power : float, optional
        A power factor to weight the inputs by. If there is 180° symmetry,
        then setting power=2 will cause the mean of π/2 and -π/2 to point
        along the same (y) axis.

    Returns
    -------
    Δθ : float, array, tensor
        The angular separation between x1 and x2

    Examples
    --------
    import torch
    from math import pi
    from galkit.functional.angle import angular_distance

    n  = 9
    x1 = torch.linspace(-pi, pi, n)
    x2 = torch.zeros(n)
    Δθ = angular_distance(x1, x2, absolute=False)

    for i,j,k in zip(x1, x2, Δθ):
        print(f'Δθ({i},{j}) = {k}')
    """
    is_tensor = isinstance(x1, torch.Tensor) or isinstance(x2, torch.Tensor)
    exp = torch.exp if is_tensor else numpy.exp
    to_angle = torch.angle if is_tensor else numpy.angle

    z = exp(1j*(x1-x2))
    z = to_angle(z if power is None else z**power)
    z = abs(z) if absolute else z
    return z if power is None else (z/power)


def circular_mean(
        input   : Union[numpy.ndarray, torch.Tensor],
        weights : Optional[Union[numpy.ndarray, torch.Tensor]] = None,
        power   : Optional[float] = None
    ) -> float:
    """
    Computes the (weighted) circular mean of a collection of angles.

    Parameters
    ----------
    input : array, tensor
        Collection of angles (in radians) to average over.

    weights : array, tensor, optional
        Optional array of weights. If set to None, no weighting is applied.
        Default is None.

    power : float, optional
        A power factor to weight the inputs by. If there is 180° symmetry,
        then setting power=2 will cause the mean of π/2 and -π/2 to point
        along the same (y) axis.

    Returns
    -------
    circular_mean : float
        The (weighted) circular mean of the input angles, which is calculed as

            z = cos(input) + 1j⋅sin(input)
            z = sum(weight*(z**power))**(1/power)
            output =arctan2(z.imag, z.real)

    Examples
    --------
    import matplotlib.pyplot as plt
    import numpy
    from galkit.functional.angle import circular_mean

    n = 10
    x = numpy.ones(n)
    y = numpy.random.randn(n)
    θ = numpy.arctan2(y, x)
    r = numpy.ones(n)
    w = numpy.random.rand(n)

    c = circular_mean(θ, w)

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.scatter(θ, r, c=w, cmap=plt.cm.Blues, label='angles')
    ax.scatter(c, 1, c='orange', label='circular mean')
    ax.legend(loc='upper left')
    fig.show()
    """
    is_tensor = isinstance(input, torch.Tensor)
    exp = torch.exp if is_tensor else numpy.exp
    to_angle = torch.angle if is_tensor else numpy.angle
    sum = torch.sum if is_tensor else numpy.sum

    v = exp(1j*input)

    if power is not None:
        v = v**power

    if weights is not None:
        v = v * weights

    angle = sum(v)

    if power is not None:
        angle = angle**(1/power)

    return to_angle(angle)

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