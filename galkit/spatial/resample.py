'''
Methods for resampling tensors.

Methods
-------
downscale_local_mean(input, factor):
    Downscales a (C x H x W) tensor along the spatial axis by `factor`.

reproject(input, h0_old, w0_old, pa_old, q_old, q0_old, p_old,
        h0_new, w0_new, pa_new, q_new, q0_new, p_new,
        scale_old, scale_new, flip_lr, flip_ud, mode)
    Reprojects a tensor from one coordinate system to another. Useful for
    deprojecting galaxy images.

to_cartesian(input, θ, r, h0, w0, pa, q, q0, p, scale, flip_lr, flip_ud,
        transform, untransform, x_min, x_max, return_grid, dense_grid,
        alpha_shift, theta_shift, mode, repeat)
    Reprojects the input tensor from a polar coordinate system to a cartesian
    coordinate system.

to_polar(input, θ, r, h0, w0, pa, q, q0, p, scale, flip_lr, flip_ud,
        transform, untransform, x_min, x_max, return_grid, dense_grid,
        alpha_shift, theta_shift, mode, repeat)
    Reprojects the input tensor from a cartesian coordinate system to a polar
    coordinate system. The first spatial dimension (height) corresponds to the
    azimuthal coordinate and the second spatial dimension (width) the radial
    coordinate.

to_new(dict)
    Appends the string `_new` to the end of the all the keys in the dictionary.

to_old(dict)
    Appends the string `_old` to the end of the all the keys in the dictionary.

transform_arcsinh(beta)
    Provides arcsinh transformation methods to the polar projection functions,
    x = arcsinh(r / beta).

transform_log(eps)
    Provides logarithmic transformation methods to the polar projection functions,
    x = log(r + eps)
'''

import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union
from math import pi
from . import coordinate, grid
from ..functional import mod_angle
from ..utils import to_tensor

def downscale_local_mean(
    input  : torch.Tensor,
    factor : int
) -> torch.Tensor:
    """
    Downscales a (C x H x W) tensor along the spatial axis by `factor`.

    Parameters
    ----------
    input : Tensor
        The input tensor to downscale. Should be of dimensions (C x H x W),
        where H and W are divisible by the downscaling factor.

    factor : int
        The downscaling factor.

    Returns
    -------
    output : Tensor
        The downsampled input tensor with shape (C x (H/factor) x (W/factor))
    
    Examples
    --------
    import matplotlib.pyplot as plt
    import torch
    from galkit.spatial import coordinate, grid, resample

    _, r = coordinate.polar(
        grid = grid.pytorch_grid(300, 300),
        q = 0.5
    )

    input = (-r).div(0.2).exp()
    output = resample.downscale_local_mean(input, 3)

    vmin = 0; vmax = input.max()

    fig, ax = plt.subplots(ncols=2)
    ax[0].imshow(input.squeeze(), vmin=vmin, vmax=vmax)
    ax[1].imshow(output.squeeze(), vmin=vmin, vmax=vmax)
    fig.show()
    """
    shape = (input.size(0), input.size(1) // factor, factor, input.size(2) // factor, factor)
    return input.view(*shape).mean(dim=2).mean(dim=-1)

def to_new(dict:Dict) -> Dict:
    """
    Appends the string `_new` to the end of the all the keys in the dictionary.
    """
    return {k+'_new':v for k,v in dict.items()}

def to_old(dict:Dict) -> Dict:
    """
    Appends the string `_old` to the end of the all the keys in the dictionary.
    """
    return {k+'_old':v for k,v in dict.items()}

def transform_arcsinh(beta:float=1) -> Dict:
    """
    Provides arcsinh transformation methods to the polar projection functions,
    x = arcsinh(r / beta).

    Parameters
    ----------
    beta : float
        Division parameter

    Returns
    -------
    output : Dict   
        Dictionary containing the `transform` and `untransform` parameters
        to pass into the function.

    Examples
    --------
    import torch
    from galkit.spatial.resample import transform_arcsinh

    t = transform_arcsinh(0.01)
    x = torch.linspace(0, 10, 10)
    y = t['transform'](x)
    z = t['untransform'](y)
    print(x)
    print(y)
    print(z)
    """
    return {
        'transform'  : lambda x: x.div(beta).arcsinh(),
        'untransform': lambda x: x.sinh().mul(beta),        
    }

def transform_log(eps:float=0.01):
    """
    Provides logarithmic transformation methods to the polar projection functions,
    x = log(r + eps)

    Parameters
    ----------
    eps : float
        Padding factor to avoid division by zero or to minimize the influence
        of very small values when transforming.

    Returns
    -------
    output : Dict   
        Dictionary containing the `transform` and `untransform` parameters
        to pass into the function.

    Examples
    --------
    import torch
    from galkit.spatial.resample import transform_log

    t = transform_log(0.1)
    x = torch.linspace(0, 10, 10)
    y = t['transform'](x)
    z = t['untransform'](y)
    print(x)
    print(y)
    print(z)
    """
    return {
        'transform'  : lambda x: x.add(eps).log(),
        'untransform': lambda x: x.exp().sub(eps)
    }

def to_polar(
    input       : torch.Tensor,
    θ           : Optional[torch.Tensor] = None,
    r           : Optional[torch.Tensor] = None,
    h0          : Optional[torch.Tensor] = None,
    w0          : Optional[torch.Tensor] = None,
    pa          : torch.Tensor = 0,
    q           : torch.Tensor = 1,
    q0          : Optional[torch.Tensor] = None,
    p           : torch.Tensor = 2,
    scale       : Optional[torch.Tensor] = None,
    flip_lr     : bool = True,
    flip_ud     : bool = False,
    transform   : Optional[callable] = None,
    untransform : Optional[callable] = None,
    x_min       : Optional[torch.Tensor] = None,
    x_max       : Optional[torch.Tensor] = None,
    return_grid : bool = False,
    dense_grid  : bool = True,
    alpha_shift : Optional[torch.Tensor] = None,
    theta_shift : Optional[torch.Tensor] = None,
    eps         : float = 1e-4,
    mode        : str = 'bilinear',
    repeat      : bool = False
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Reprojects the input tensor from a cartesian coordinate system to a polar
    coordinate system. The first spatial dimension (height) corresponds to the
    azimuthal coordinate and the second spatial dimension (width) the radial
    coordinate.

    Parameters
    ----------
    input : Tensor
        The input tensor to resample.

    θ : Tensor, optional
        The azimuthal coordinates. If these are not passed, then they are generated.

    r : Tensor, optional
        The radial coordinates. If these are not passed, then they are generated.

    h0: Tensor
        The height (vertical) center coordinate in the pytorch grid system. If not a
        tensor, it will be converted to one.

    w0: Tensor
        The width (horizontal) center coordinate in the pytorch grid system. If not a
        tensor, it will be converted to one.

    pa : Tensor
        Position angle of the semi-major axis in radians. Note that the rotation
        angle is θ_rot = -pa

    q : Tensor
        Flattening parameter representing the ratio of the semi-minor to semi-
        major axis (b/a)

    q0 : Tensor, optional
        Axis ratio correction for an obolate spheroid disk. Some typical values
        are 0.13 or 0.20. Default is None.

    p : Tensor
        Generalized ellipse parameter, rᵖ = |x|ᵖ + |y/q'|ᵖ. Default is 2.

    scale: Tensor, optional
        Scaling parameter to multiply the coordinates by. If not a tensor,
        it will be converted to one.

    flip_lr : boolean
        Boolean indicating to return the negative of the horizontal coordinates.
        Useful for adjusting the horizontal direction of increasing values. Default 
        is True so that values increase towards the left, in keeping with the 
        convention of right ascension increasing towards the East.

    flip_ud : boolean
        Boolean indicating to return the negative of the vertical coordinates.
        Useful for adjusting the vertical direction of increasing values. Default
        is False.

    transform : callable, optional
        Function for transforming the radial coordinates when generating a linearly
        spaced grid, x = transform(r).

    untransform : callable, optional
        Function for undoing the radial coordinate transformation after generating
        a linearly space grid, r = untransform(transform(r)).

    x_min : Tensor, optional
        The minimum value for the radial coordinate grid, corresponding to the transformed
        radial coordinate, x = transform(r). If `None`, then the minimum value is used.

    x_max : Tensor, optional
        The maximum value for the radial coordinate grid, corresponding to the transformed
        radial coordinate, x = transform(r). If `None`, then the maximum value is used.

    return_grid : bool
        Boolean indicating whether to return the grid coordinates (True) or not (False).
        Default is False.

    dense_grid  : bool
        Boolean indicating whether to return a dense representation of the grid. Only
        applicable when `return_grid=True`.

    alpha_shift : Tensor, optional
        If passed, then a logarithmic shift is applied to the azimuthal coordinates
        based on the provided pitch angle. Useful for undoing a spiral pattern.

    theta_shift : Tensor, optional
        If passed, then a phase shift is applied to the azimuthal coordinates. Mostly
        implemented for training purposes.

    eps : float
        Padding factor to avoid log(0) when taking the logarithm of the radial coordinate
        during the pitch angle rotation shift. Default is 1e-4.

    mode : str
        Interpolation mode. Can be: 'bilinear', 'nearest', or 'bicubic'. Default
        is 'bilinear'.

    repeat : bool
        Boolean indicating whether to include ±π at the boundaries or just one.

    Returns
    -------
    output : Tensor
        The resampled tensor

    θ_grid: Tensor, optional
        Azimuthal grid coordinates. Only returned if `return_grid=True`.

    x_grid : Tensor, optional
        Transformed radial grid coordinates. Only returned if `return_grid=True`.

    Examples
    --------
    import matplotlib.pyplot as plt
    from galkit.spatial import coordinate, grid, resample
    from math import pi

    geom = {'h0': 0, 'w0': 0, 'pa': 0, 'q': 0.5}

    θ, r = coordinate.polar(
        grid = grid.pytorch_grid(100,100),
        **geom
    )

    c_imag = (-r).exp().unsqueeze(0)
    p_imag1 = resample.to_polar(c_imag, theta_shift=0, **geom)
    p_imag2 = resample.to_polar(c_imag, **geom, **resample.transform_arcsinh(1.0))
    p_imag3 = resample.to_polar(c_imag, **geom, **resample.transform_log(0.01))

    fig, ax = plt.subplots(ncols=2, nrows=2)
    ax[0,0].imshow(c_imag.squeeze())
    ax[0,1].imshow(p_imag1.squeeze())
    ax[1,0].imshow(p_imag2.squeeze())
    ax[1,1].imshow(p_imag3.squeeze())

    ax[0,0].set_title('Cartesian Image')
    ax[0,1].set_title('Linear Projection')
    ax[1,0].set_title('Arcsinh Projection')
    ax[1,1].set_title('Log Projection')

    fig.tight_layout()
    fig.show()
    """
    # ==========================================================================
    # Store the polar and cartesian parameters in dictionaries for easy passing
    # into the appropriate functions
    # ==========================================================================
    ckeys = {'h0': h0, 'w0': w0, 'scale': scale, 'flip_lr': flip_lr, 'flip_ud': flip_ud}
    pkeys = {'pa': pa, 'q' : q, 'q0': q0, 'p' : p}

    # ==========================================================================
    # Generate the azimuthal and radial coordinates
    # ==========================================================================
    if (θ is None) or (r is None):
        θ, r = coordinate.polar(
            grid=grid.pytorch_grid(input.shape[-2:], device=input.device),
            **ckeys, **pkeys
        )

    # ==========================================================================
    # Generate the azimuthal grid for sampling. Since -π == π, the latter is
    # removed to prevent a duplicate. This grid will correspond to the height
    # coordinate.
    # ==========================================================================
    if repeat:
        θ_grid = torch.linspace(-pi, pi, input.size(-2), device=input.device).view(1,-1,1)
    else:
        θ_grid = torch.linspace(-pi, pi, input.size(-2)+1, device=input.device)[:-1].view(1,-1,1)

    # ==========================================================================
    # Generate the radial grid for resampling. This grid will correspond to the
    # width coordinate.
    # ==========================================================================
    x = r if transform is None else transform(r)
    x_min = to_tensor(x_min, device=input.device) if x_min is not None else x.flatten(1).min(dim=-1)[0].view(-1,1,1)
    x_max = to_tensor(x_max, device=input.device) if x_max is not None else x.flatten(1).max(dim=-1)[0].view(-1,1,1)

    x_grid = torch.linspace(0, 1, input.size(-1), device=input.device).view(1,1,-1)
    x_grid = x_grid * (x_max - x_min) + x_min

    # ==========================================================================
    # Convert the grid coordinates to the (h,w) coordinates in the image
    # ==========================================================================
    r_grid = x_grid if transform is None else untransform(x_grid)

    if alpha_shift is not None:
        alpha_shift = to_tensor(alpha_shift, device=input.device)
        θ_grid = θ_grid + r_grid.add(eps).log() / alpha_shift.tan()
    if theta_shift is not None:
        theta_shift = to_tensor(theta_shift, device=input.device)
        θ_grid = θ_grid + theta_shift

    h_grid, w_grid = coordinate.polar_to_grid(θ=θ_grid, r=r_grid, **ckeys, **pkeys)

    # ==========================================================================
    # Sample the grid values from the cartesian image.
    #   NOTE: For some reason, the grid ordering needs to be reversed
    # ==========================================================================
    output = F.grid_sample(input,
        grid = torch.stack([w_grid, h_grid], axis=-1),
        align_corners=True,
        mode = mode,
    )    

    # ==========================================================================
    # Return output
    # ==========================================================================
    if return_grid:
        if dense_grid:
            x_grid = x_grid.expand(input.size(0),x_grid.size(-1),-1)
            θ_grid = θ_grid.expand(input.size(0),-1,x_grid.size(1))
        return output, θ_grid, x_grid
    else:
        return output

def to_cartesian(
    input       : torch.Tensor,
    θ           : Optional[torch.Tensor] = None,
    r           : Optional[torch.Tensor] = None,
    h0          : Optional[Union[float,torch.Tensor]] = None,
    w0          : Optional[Union[float,torch.Tensor]] = None,
    pa          : Union[float,torch.Tensor] = 0,
    q           : Union[float,torch.Tensor] = 1,
    q0          : Optional[Union[float,torch.Tensor]] = None,
    p           : Union[float,torch.Tensor] = 2,
    scale       : Optional[Union[float,torch.Tensor]] = None,
    flip_lr     : bool = True,
    flip_ud     : bool = False,
    transform   : Optional[callable] = None,
    untransform : Optional[callable] = None,
    x_min       : Optional[Union[float,torch.Tensor]] = None,
    x_max       : Optional[Union[float,torch.Tensor]] = None,
    alpha_shift : Optional[Union[float,torch.Tensor]] = None,
    theta_shift : Optional[Union[float,torch.Tensor]] = None,
    eps         : Optional[float] = 1e-4,
    mode        : str = 'bilinear',
    repeat      : bool = False,
) -> torch.Tensor:
    """
    Reprojects the input tensor from a polar coordinate system to a cartesian
    coordinate system.

    Parameters
    ----------
    input : Tensor
        The input tensor to resample.

    θ : Tensor, optional
        The azimuthal coordinates. If these are not passed, then they are generated.

    r : Tensor, optional
        The radial coordinates. If these are not passed, then they are generated.

    h0: Tensor
        The height (vertical) center coordinate in the pytorch grid system. If not a
        tensor, it will be converted to one.

    w0: Tensor
        The width (horizontal) center coordinate in the pytorch grid system. If not a
        tensor, it will be converted to one.

    pa : Tensor
        Position angle of the semi-major axis in radians. Note that the rotation
        angle is θ_rot = -pa

    q : Tensor
        Flattening parameter representing the ratio of the semi-minor to semi-
        major axis (b/a)

    q0 : Tensor, optional
        Axis ratio correction for an obolate spheroid disk. Some typical values
        are 0.13 or 0.20. Default is None.

    p : Tensor
        Generalized ellipse parameter, rᵖ = |x|ᵖ + |y/q'|ᵖ. Default is 2.

    scale: Tensor, optional
        Scaling parameter to multiply the coordinates by. If not a tensor,
        it will be converted to one.

    flip_lr : boolean
        Boolean indicating to return the negative of the horizontal coordinates.
        Useful for adjusting the horizontal direction of increasing values. Default 
        is True so that values increase towards the left, in keeping with the 
        convention of right ascension increasing towards the East.

    flip_ud : boolean
        Boolean indicating to return the negative of the vertical coordinates.
        Useful for adjusting the vertical direction of increasing values. Default
        is False.

    transform : callable, optional
        Function for transforming the radial coordinates when generating a linearly
        spaced grid, x = transform(r).

    untransform : callable, optional
        Function for undoing the radial coordinate transformation after generating
        a linearly space grid, r = untransform(transform(r)).
        
        Not used by the function, but included for retaining the same arguments between
        `to_polar` and `to_cartesian`.

    x_min : Tensor, optional
        The minimum value for the radial coordinate grid, corresponding to the transformed
        radial coordinate, x = transform(r). If `None`, then the minimum value is used.

    x_max : Tensor, optional
        The maximum value for the radial coordinate grid, corresponding to the transformed
        radial coordinate, x = transform(r). If `None`, then the maximum value is used.

    return_grid : bool
        Boolean indicating whether to return the grid coordinates (True) or not (False).
        Default is False.

    dense_grid  : bool
        Boolean indicating whether to return a dense representation of the grid. Only
        applicable when `return_grid=True`.

    alpha_shift : Tensor, optional
        If passed, then a logarithmic shift is applied to the azimuthal coordinates. Useful
        for undoing a spiral pattern.

    theta_shift : Tensor, optional
        If passed, then a phase shift is applied to the azimuthal coordinates. Mostly
        implemented for training purposes.

    eps : float
        Padding factor to avoid log(0) when taking the logarithm of the radial coordinate
        during the pitch angle rotation shift. Default is 1e-4.

    mode : str
        Interpolation mode. Can be: 'bilinear', 'nearest', or 'bicubic'. Default
        is 'bilinear'.

    repeat : bool
        Boolean indicating whether to include ±π at the boundaries or just one.

    Returns
    -------
    output : Tensor
        The resampled tensor

    Examples
    --------
    import matplotlib.pyplot as plt
    from galkit.spatial import coordinate, grid, resample
    from math import pi

    geom = {'h0': 0, 'w0': 0, 'pa': 0, 'q': 0.5}

    θ, r = coordinate.polar(
        grid = grid.pytorch_grid(100,100),
        **geom
    )

    geom.update({
        'theta_shift': pi/2,
        **resample.transform_arcsinh(1.0)
    })

    imag   = (-r).exp().unsqueeze(0)
    p_imag = resample.to_polar(imag, **geom)
    c_imag = resample.to_cartesian(p_imag, **geom)

    fig, ax = plt.subplots(ncols=3)
    ax[0].imshow(imag.squeeze())
    ax[1].imshow(p_imag.squeeze())
    ax[2].imshow(c_imag.squeeze())
    fig.tight_layout()
    fig.show()
    """
    # ==========================================================================
    # Store the polar and cartesian parameters in dictionaries for easy passing
    # into the appropriate functions
    # ==========================================================================
    ckeys = {'h0': h0, 'w0': w0, 'scale': scale, 'flip_lr': flip_lr, 'flip_ud': flip_ud}
    pkeys = {'pa': pa, 'q' : q, 'q0': q0, 'p' : p}

    # ==========================================================================
    # Generate the azimuthal and radial coordinates
    # ==========================================================================
    if (θ is None) or (r is None):
        θ, r = coordinate.polar(
            grid=grid.pytorch_grid(input.shape[-2:], device=input.device),
            **ckeys, **pkeys
        )

    if alpha_shift is not None:
        alpha_shift = to_tensor(alpha_shift, device=input.device)
        Δθ = r.add(eps).log() / alpha_shift.tan()
        θ  = mod_angle(θ, -Δθ)

    if theta_shift is not None:
        theta_shift = to_tensor(theta_shift, device=input.device)
        θ = mod_angle(θ, -theta_shift)

    # ==========================================================================
    # For grid sampling purposes, pad the azimuth so that the ends correspond to
    # -π and +π radians if the input doesn't repeat at the end
    # ==========================================================================
    if not repeat:
        input = F.pad(input, (0,0,0,1), mode='circular')

    # ==========================================================================
    # Calculate the location of the (θ,r) coordinates in the polar image. Note
    # that grid_sample has a range of -1 to +1
    # ==========================================================================
    x = r if transform is None else transform(r)
    x_min = to_tensor(x_min, device=input.device) if x_min is not None else x.flatten(1).min(dim=-1)[0].view(-1,1,1)
    x_max = to_tensor(x_max, device=input.device) if x_max is not None else x.flatten(1).max(dim=-1)[0].view(-1,1,1)

    x = 2 * (x - x_min) / (x_max - x_min) - 1

    return F.grid_sample(input,
        grid=torch.stack([x, θ / pi], axis=-1),
        align_corners=True,
        mode=mode,
    )

def reproject(
    input           : torch.Tensor,
    h0_old          : Union[float, torch.Tensor],
    w0_old          : Union[float, torch.Tensor],
    pa_old          : Union[float, torch.Tensor],
    q_old           : Union[float, torch.Tensor],
    q0_old          : Optional[Union[float, torch.Tensor]] = None,
    p_old           : Union[float, torch.Tensor] = 2,
    h0_new          : Union[float, torch.Tensor] = 0,
    w0_new          : Union[float, torch.Tensor] = 0,
    pa_new          : Optional[Union[float, torch.Tensor]] = None,
    q_new           : Union[float, torch.Tensor] = 1,
    q0_new          : Optional[Union[float, torch.Tensor]] = None,
    p_new           : Union[float, torch.Tensor] = 2,
    scale_old       : Optional[Union[float, torch.Tensor]] = None,
    scale_new       : Optional[Union[float, torch.Tensor]] = None,
    flip_lr         : bool = True,
    flip_ud         : bool = False,
    alpha_shift_old : Optional[torch.Tensor] = None,
    alpha_shift_new : Optional[torch.Tensor] = None,
    theta_shift_old : Optional[torch.Tensor] = None,
    theta_shift_new : Optional[torch.Tensor] = None,   
    eps             : float = 1e-4,
    mode            : str = 'bilinear',
    output_shape    : Optional[Tuple[int,int]] = None,
) -> torch.Tensor:
    """
    Reprojects a tensor from one coordinate system to another. Useful for
    deprojecting galaxy images.

    Parameters
    ----------
    input : Tensor
        The input tensor to resample.

    h0(_old, _new): Tensor
        The height (vertical) center coordinate in the pytorch grid system. If not a
        tensor, it will be converted to one.

    w0(_old, _new): Tensor
        The width (horizontal) center coordinate in the pytorch grid system. If not a
        tensor, it will be converted to one.

    pa(_old, _new) : Tensor
        Position angle of the semi-major axis in radians. Note that the rotation
        angle is θ_rot = -pa. If `pa_new=None`, then the old value is retained,
        and likewise for `pa_old=None`.

    q(_old, _new) : Tensor
        Flattening parameter representing the ratio of the semi-minor to semi-
        major axis (b/a)

    q0(_old, _new) : Tensor, optional
        Axis ratio correction for an obolate spheroid disk. Some typical values
        are 0.13 or 0.20. Default is None.

    p(_old, _new) : Tensor
        Generalized ellipse parameter, rᵖ = |x|ᵖ + |y/q'|ᵖ. Default is 2.

    scale(_old, _new): Tensor, optional
        Scaling parameter to multiply the coordinates by. If not a tensor,
        it will be converted to one.

    flip_lr : boolean
        Boolean indicating to return the negative of the horizontal coordinates.
        Useful for adjusting the horizontal direction of increasing values. Default 
        is True so that values increase towards the left, in keeping with the 
        convention of right ascension increasing towards the East.

    flip_ud : boolean
        Boolean indicating to return the negative of the vertical coordinates.
        Useful for adjusting the vertical direction of increasing values. Default
        is False.

    alpha_shift(_old, _new) : Tensor, optional
        If passed, then a logarithmic shift is applied to the azimuthal coordinates. Useful
        for undoing a spiral pattern.

    theta_shift(_old, _new) : Tensor, optional
        If passed, then a phase shift is applied to the azimuthal coordinates. Mostly
        implemented for training purposes.

    eps : float
        Padding factor to avoid log(0) when taking the logarithm of the radial coordinate
        during the pitch angle rotation shift. Default is 1e-4.

    mode : str
        Interpolation mode. Can be: 'bilinear', 'nearest', or 'bicubic'. Default
        is 'bilinear'.

    output_shape : Tuple[int,int], optional
        The desired output shape. If `None`, then the output has the same shape
        as the input.

    Returns
    -------
    output : Tensor
        The reprojected input tensor.

    Examples
    --------
    import matplotlib.pyplot as plt
    from galkit.spatial import grid, coordinate, resample
    from galkit.spatial.resample import to_new, to_old
    from math import pi

    geom = {'h0': 0., 'w0': 0., 'pa': pi/4, 'q': 0.25, 'q0': 0.2}
    face_on = {'h0': 0, 'w0': 0, 'pa': 0, 'q': 1}

    θ, r = coordinate.polar(
        grid = grid.pytorch_grid(100,100),
        **geom,
    )
    imag = (-r).exp().unsqueeze(0)

    p_imag = resample.reproject(imag, **to_old(geom), **to_new(face_on), output_shape=(50,50))
    c_imag = resample.reproject(p_imag, **to_new(geom), **to_old(face_on), output_shape=(50,50))

    fig, ax = plt.subplots(ncols=3)
    ax[0].imshow(imag.squeeze())
    ax[1].imshow(p_imag.squeeze())
    ax[2].imshow(c_imag.squeeze())
    fig.show()
    """
    # ==========================================================================
    # Store the polar and cartesian parameters in dictionaries for easy passing
    # into the appropriate functions
    # ==========================================================================
    ckeys_old = {'h0': h0_old, 'w0': w0_old, 'scale': scale_old, 'flip_lr': flip_lr, 'flip_ud': flip_ud}
    ckeys_new = {'h0': h0_new, 'w0': w0_new, 'scale': scale_new, 'flip_lr': flip_lr, 'flip_ud': flip_ud}
    pkeys_old = {'pa': pa_old if pa_old is not None else pa_new, 'q': q_old, 'q0': q0_old, 'p': p_old}
    pkeys_new = {'pa': pa_new if pa_new is not None else pa_old, 'q': q_new, 'q0': q0_new, 'p': p_new}

    # ==========================================================================
    # Generate the new coordinate system. Only want to use the spatial 
    # dimensions when constructing the grid.
    # ==========================================================================
    θ, r = coordinate.polar(
        grid=grid.pytorch_grid(input.shape[-2:] if output_shape is None else output_shape, device=input.device),
        **ckeys_new, **pkeys_new
    )

    # ==========================================================================
    # Apply shifts (if applicable)
    # ==========================================================================
    if alpha_shift_old is not None:
        alpha_shift_old = to_tensor(alpha_shift_old, device=input.device)
        u = r.add(eps).log()
        Δθ = u / alpha_shift_old.tan()
        θ = mod_angle(θ, Δθ)

    if alpha_shift_new is not None:
        alpha_shift_new = to_tensor(alpha_shift_new, device=input.device) 
        u = r.add(eps).log()
        Δθ = u / alpha_shift_new.tan()
        θ = mod_angle(θ, -Δθ)

    if theta_shift_old is not None:
        theta_shift_old = to_tensor(theta_shift_old, device=input.device)
        θ = mod_angle(θ, -theta_shift_old)        

    if theta_shift_new is not None:
        theta_shift_new = to_tensor(theta_shift_new, device=input.device)
        θ = mod_angle(θ, theta_shift_new)     

    # ==========================================================================
    # Convert the polar coordinates to the grid coordinates in the old system
    # ==========================================================================
    h_grid, w_grid = coordinate.polar_to_grid(θ=θ, r=r, **ckeys_old, **pkeys_old)

    # ==========================================================================
    # Sample the grid values from the cartesian image.
    #   NOTE: For some reason, the grid ordering needs to be reversed
    # ==========================================================================
    return F.grid_sample(input,
        grid = torch.stack([w_grid, h_grid], axis=-1),
        align_corners=True,
        mode = mode,
    )