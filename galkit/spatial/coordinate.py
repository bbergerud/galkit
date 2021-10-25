'''
This module contains functions for dealing with the spatial coordinates
in an image. It is primarily designed for dealing with projected disk
geometries of galaxies.

Functions
---------
cartesian(grid, h0, w0, scale, flip_lr, flip_ud)
    Generates a cartesian coordinate system based on the input grid and specified
    parameters.

cartesian_to_grid(h, w, h0, w0, scale, flip_lr, flip_ud)
    Converts cartesian coordinates to the original grid coordinates.

cartesian_to_polar(h, w, pa, q, q0, p)
    Converts cartesian coordinates to polar coordinates.

polar(grid, h0, w0, scale, flip_lr, flip_ud, pa, q, q0, p)
    Generates a polar coordinate system based on the input grid and specified
    parameters.

polar_to_cartesian(θ, r, pa, q, q0, p)
    Converts polar coordinates to cartesian coordinates.

polar_to_grid(θ, r, pa, q, q0, h0, w0, scale, flip_lr, flip_ud) 
    Converts polar coordinates to the original grid coordinates.

θ_ellipse(x, y, q, q0)
    Returns the azimuthal coordinates in radians, which is defined as θ = arctan2(y, q'*x)
    where q' is the ellipsoidally corrected q value.   

r_ellipse(x, y, q, q0, p)
    Returns the radial coordinates, which is defined as rᵖ = |x|ᵖ + |y/q'|ᵖ,
    where q' is the ellipsoidally corrected q value.

rotate(x, y, angle)
    Rotates (x,y) coordinates by the angle `angle`. When dealing with position angles,
    angle = -pa.

q_correction(q, q0, device)
    Computes the ellipsoidally corrected value (q')² ≡ cos²(i) = [(b/a)² - q0²] / [1 - q0²]
    where q ≡ b/a is the observed semi-minor to semi-major axis ratio.
'''

import torch
from typing import Optional, Tuple
from ..utils import to_tensor, safe_divisor

def cartesian(
    grid    : Tuple[torch.Tensor, torch.Tensor], 
    h0      : Optional[torch.Tensor] = None, 
    w0      : Optional[torch.Tensor] = None, 
    scale   : Optional[torch.Tensor] = None, 
    flip_lr : bool = True, 
    flip_ud : bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates a cartesian coordinate system based on the input grid and specified
    parameters.

    Parameters
    ----------
    grid : Tuple[Tensor, Tensor]
        Iterable containing the (h,w) coordinates of the grid system.

    h0: Tensor, optional
        The height (vertical) center coordinate in the grid system. If not a
        tensor, it will be converted to one.

    w0: Tensor, optional
        The width (horizontal) center coordinate in the grid system. If not a
        tensor, it will be converted to one.

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

    Returns
    -------
    h : Tensor
        The height (vertical) coordinates.

    w : Tensor
        The width (horizontal) coordinates.

    Examples
    --------
    import matplotlib.pyplot as plt
    from galkit.spatial import grid, coordinate

    h, w = coordinate.cartesian(
        grid = grid.pytorch_grid(100,100, dense=True),
        h0 = [0.5, 0.0, -0.5],
        w0 = [0.5, 0.0, -0.5],
        scale = [0.5, 1, 1.5],
        flip_lr = True,
        flip_ud = False
    )

    def foo(i):
        vmax = max(h[i].abs().max(), w[i].abs().max())

        fig, ax = plt.subplots(ncols=2)
        c0 = ax[0].imshow(h[i], vmin=-vmax, vmax=vmax, cmap=plt.cm.bwr)
        c1 = ax[1].imshow(w[i], vmin=-vmax, vmax=vmax, cmap=plt.cm.bwr)
        fig.colorbar(c0, ax=ax[0], orientation='horizontal')
        fig.colorbar(c1, ax=ax[1], orientation='horizontal')    
        fig.tight_layout()
        fig.show()

    foo(0)
    """
    h, w = grid
    if h0 is not None:
        h = h - to_tensor(h0, device=h.device)
    if w0 is not None:
        w = w - to_tensor(w0, device=h.device)
    if scale is not None:
        scale = to_tensor(scale, device=h.device)
        h = h * scale
        w = w * scale
    return -h if flip_ud else h, -w if flip_lr else w

def cartesian_to_grid(
    h       : torch.Tensor, 
    w       : torch.Tensor, 
    h0      : Optional[torch.Tensor] = None, 
    w0      : Optional[torch.Tensor] = None, 
    scale   : Optional[torch.Tensor] = None, 
    flip_lr : bool = True, 
    flip_ud : bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Converts cartesian coordinates to the original grid coordinates.

    Parameters
    ----------
    h : Tensor
        The height (vertical) coordinates.

    w : Tensor
        The width (horizontal) coordinates.

    h0: Tensor, optional
        The height (vertical) center coordinate in the grid system. If not a
        tensor, it will be converted to one.

    w0: Tensor, optional
        The width (horizontal) center coordinate in the grid system. If not a
        tensor, it will be converted to one.

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

    Returns
    -------
    h : Tensor
        The height (vertical) grid coordinates.

    w : Tensor
        The width (horizontal) grid coordinates.

    Examples
    --------
    import matplotlib.pyplot as plt
    from galkit.spatial import grid, coordinate

    kwargs = {
        'h0': [0.1, 0.0, -0.1],
        'w0': [0.2, -0.1, 0.1],
        'scale': [0.5, 1., 1.5],
        'flip_lr': True,
        'flip_ud': True
    }

    g = grid.pytorch_grid(100,100, dense=True)
    h, w = coordinate.cartesian(grid=g, **kwargs)
    h, w = coordinate.cartesian_to_grid(h=h, w=w, **kwargs)

    def foo(i):
        fig, ax = plt.subplots(ncols=2)
        c0 = ax[0].imshow((h[i] - g[0]).squeeze())
        c1 = ax[1].imshow((w[i] - g[1]).squeeze())
        fig.colorbar(c0, ax=ax[0], orientation='horizontal')
        fig.colorbar(c1, ax=ax[1], orientation='horizontal')    
        fig.tight_layout()    
        fig.show()

    foo(0)
    """
    if flip_lr: w = -w
    if flip_ud: h = -h
    if scale is not None:
        scale = to_tensor(scale, device=h.device)
        h = h / scale
        w = w / scale
    if h0 is not None:
        h = h + to_tensor(h0, device=h.device)
    if w0 is not None:
        w = w + to_tensor(w0, device=h.device)
    return h, w

def cartesian_to_polar(
    h  : torch.Tensor, 
    w  : torch.Tensor, 
    pa : torch.Tensor, 
    q  : torch.Tensor, 
    q0 : Optional[torch.Tensor] = None,
    p  : torch.Tensor = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Converts cartesian coordinates to polar coordinates.

    Parameters
    ----------
    h : Tensor
        The height (vertical) grid coordinates.

    w : Tensor
        The width (horizontal) grid coordinates.

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

    Returns
    -------
    θ : Tensor
        The azimuthal coordinates in radians

    r : Tensor
        The radial coordinates

    Examples
    --------
    from galkit.spatial import grid, coordinate
    from numpy import deg2rad
    import matplotlib.pyplot as plt

    h, w = coordinate.cartesian(
        grid = grid.pytorch_grid(100,100),
        h0 = 0.1, w0=0.2
    )

    θ, r = coordinate.cartesian_to_polar(
        h  = h,
        w  = w,
        q  = [0.5, 0.75, 1.0],
        pa = [0, deg2rad(45), deg2rad(90)],
        p  = [1, 2, 3]
    )

    def foo(i):
        fig, ax = plt.subplots()
        ax.imshow(θ[i].squeeze())
        ax.contour(r[i].squeeze())
        fig.show()

    foo(0)
    """
    if not isinstance(pa, torch.Tensor):
        pa = to_tensor(pa, device=h.device)
    x, y = rotate(h, w, angle=-pa)
    q = q_correction(q, q0, device=h.device)
    θ = θ_ellipse(x, y, q=q)
    r = r_ellipse(x, y, q=q, p=p)
    return θ, r

def polar(
    grid    : Tuple[torch.Tensor, torch.Tensor],
    h0      : Optional[torch.Tensor] = None, 
    w0      : Optional[torch.Tensor] = None, 
    scale   : Optional[torch.Tensor] = None,
    flip_lr : bool = True,
    flip_ud : bool = False,
    pa      : torch.Tensor = 0,
    q       : torch.Tensor = 1,
    q0      : Optional[torch.Tensor] = None,
    p       : torch.Tensor = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates a polar coordinate system based on the input grid and specified
    parameters.

    Parameters
    ----------
    grid : Tuple[Tensor]
        Iterable containing the (h,w) coordinates of the grid system.

    h0: Tensor, optional
        The height (vertical) center coordinate in the grid system. If not a
        tensor, it will be converted to one.

    w0: Tensor, optional
        The width (horizontal) center coordinate in the grid system. If not a
        tensor, it will be converted to one.

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

    Returns
    -------
    θ : Tensor
        The azimuthal coordinates in radians

    r : Tensor
        The radial coordinates

    Examples
    --------
    from galkit.spatial import grid, coordinate
    from numpy import deg2rad
    import matplotlib.pyplot as plt

    θ, r = coordinate.polar(
        grid = grid.pytorch_grid(100,100),
        h0 = [-0.2, 0.2],
        w0 = [-0.2, 0.2],
        pa = [deg2rad(45), deg2rad(-45)],
        q = [0.33, 0.67],
        p = [1.5, 2.5]
    )

    def foo(i):
        fig, ax = plt.subplots()
        ax.imshow(θ[i])
        ax.contour(r[i])
        fig.show()

    foo(0)
    """
    h, w = cartesian(grid, h0=h0, w0=w0, scale=scale, flip_lr=flip_lr, flip_ud=flip_ud)
    θ, r = cartesian_to_polar(h=h, w=w, pa=pa, q=q, q0=q0, p=p)
    return θ, r


def polar_to_cartesian(
    θ  : torch.Tensor, 
    r  : torch.Tensor, 
    pa : torch.Tensor = 0, 
    q  : torch.Tensor = 1,
    q0 : Optional[torch.Tensor] = None, 
    p  : torch.Tensor = 2
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Converts polar coordinates to cartesian coordinates.

    Parameters
    ----------
    θ : Tensor
        The azimuthal coordinates in radians

    r : Tensor
        The radial coordinates

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

    Returns
    -------
    h : Tensor
        The height (vertical) grid coordinates.

    w : Tensor
        The width (horizontal) grid coordinates.

    Examples
    --------
    from galkit.spatial import grid, coordinate
    from numpy import deg2rad
    import matplotlib.pyplot as plt

    ckeys = {
        'h0': 0,
        'w0': 0,
        'scale' : 1,
        'flip_lr' : True,
        'flip_ud' : False,
    }

    pkeys = {
        'pa': deg2rad(0),
        'q' : 0.25,
        'q0': 0.20,
        'p' : 2.5
    }

    h, w = coordinate.cartesian(grid = grid.pytorch_grid(100,100), **ckeys)
    θ, r = coordinate.cartesian_to_polar(h=h, w=w, **pkeys)
    hp, wp = coordinate.polar_to_cartesian(θ=θ, r=r, **pkeys)

    fig, ax = plt.subplots(ncols=2)
    c0 = ax[0].imshow(h.squeeze(0) - hp.squeeze())
    c1 = ax[1].imshow(w.squeeze(0) - wp.squeeze())
    fig.colorbar(c0, ax=ax[0], orientation='horizontal')
    fig.colorbar(c1, ax=ax[1], orientation='horizontal') 
    fig.tight_layout()
    fig.show()
    """
    p = to_tensor(p, device=r.device)
    q = q_correction(q, q0, device=r.device) 
    if not isinstance(pa, torch.Tensor) or pa.ndim != 3:
        pa = to_tensor(pa, device=r.device)

    if (p == 2).all():
        y = r * torch.sin(θ) * q
        x = r * torch.cos(θ)
    else:
        tanθ = θ.tan()             # tan(θ) = y / (q⋅x)
        rp = r.pow(p)              # rᵖ = |x|ᵖ + |y/q|ᵖ
        x  = (rp / (1 + tanθ.abs().pow(p))).pow(1/p) * (θ.cos().sign())
        y  = x * q * tanθ

    x, y = rotate(x, y, angle=pa)
    return x, y

def polar_to_grid(
    θ       : torch.Tensor, 
    r       : torch.Tensor, 
    pa      : torch.Tensor = 0, 
    q       : torch.Tensor = 1,
    q0      : Optional[torch.Tensor] = None,
    p       : torch.Tensor = 2, 
    h0      : Optional[torch.Tensor] = None, 
    w0      : Optional[torch.Tensor] = None, 
    scale   : Optional[torch.Tensor] = None, 
    flip_lr : torch.Tensor = True, 
    flip_ud : torch.Tensor = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Converts polar coordinates to the original grid coordinates.

    Parameters
    ----------
    θ : Tensor
        The azimuthal coordinates in radians

    r : Tensor
        The radial coordinates

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

    h0: Tensor, optional
        The height (vertical) center coordinate in the grid system. If not a
        tensor, it will be converted to one.

    w0: Tensor, optional
        The width (horizontal) center coordinate in the grid system. If not a
        tensor, it will be converted to one.

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

    Returns
    -------
    h : Tensor
        The height (vertical) grid coordinates.

    w : Tensor
        The width (horizontal) grid coordinates.

    Examples
    --------
    from galkit.spatial import grid, coordinate
    from numpy import deg2rad
    import matplotlib.pyplot as plt

    ckeys = {
        'h0': 0,
        'w0': 0,
        'scale' : 1,
        'flip_lr' : True,
        'flip_ud' : False,
    }

    pkeys = {
        'pa': deg2rad(15),
        'q' : 0.5,
        'p' : 2.5,
        'q0': 0.2
    }

    h, w = grid.pytorch_grid(100,100)
    θ, r = coordinate.polar(grid=(h,w), **pkeys, **ckeys)
    hp, wp = coordinate.polar_to_grid(θ=θ, r=r, **pkeys, **ckeys)

    def foo(i):
        fig, ax = plt.subplots(ncols=2)
        ax[0].imshow(h[0] - hp[i])
        ax[1].imshow(w[0] - wp[i])
        fig.show()

    foo(0)
    """
    h, w = polar_to_cartesian(θ=θ, r=r, pa=pa, q=q, p=p, q0=q0)
    h, w = cartesian_to_grid(h=h, w=w, h0=h0, w0=w0, scale=scale, flip_ud=flip_ud, flip_lr=flip_lr)
    return h, w

def θ_ellipse(
    x  : torch.Tensor, 
    y  : torch.Tensor, 
    q  : torch.Tensor = 1,
    q0 : Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Returns the azimuthal coordinates in radians, which is defined as

        θ = arctan2(y, q'*x)

     where q' is the ellipsoidally corrected q value.   

    Parameters
    ----------
    x : Tensor
        The coordinate along the semi-major axis

    y : Tensor
        The coordinate along the semi-minor axis

    q : Tensor
        Flattening parameter representing the ratio of the semi-minor to semi-
        major axis (b/a).

    q0 : Tensor, optional
        Axis ratio correction for an obolate spheroid disk. Some typical values
        are 0.13 or 0.20. Default is None.

    Returns
    -------
    θ : Tensor
        The azimuthal coordinates in radians

    Examples
    --------
    import matplotlib.pyplot as plt
    import numpy
    from galkit.spatial import grid, coordinate

    ckeys = {
        'h0': [-0.25, 0, 0.25],
        'w0': [0.25, 0, -0.25],
        'scale': [0.5, 1., 1.5],
        'flip_lr': True,
        'flip_ud': False
    }

    q  = [0.5, 0.75, 1.0]
    pa = numpy.deg2rad(numpy.array([45, 135, 0]))

    h, w = coordinate.cartesian(
        grid=grid.pytorch_grid(100,100),
        **ckeys
    )
    x, y = coordinate.rotate(h, w, angle=-pa)
    θ = coordinate.θ_ellipse(x=x, y=y, q=q)
    r = coordinate.r_ellipse(x=x, y=y, q=q)

    def foo(i):
        fig, ax = plt.subplots()
        ax.imshow(θ[i])
        ax.contour(r[i])
        fig.show()

    foo(0)
    """
    return torch.atan2(y, q_correction(q, q0, device=x.device)*x)

def r_ellipse(
    x   : torch.Tensor, 
    y   : torch.Tensor, 
    q   : torch.Tensor = 1,
    q0  : Optional[torch.Tensor] = None,
    p   : torch.Tensor = 2,
) -> torch.Tensor:
    """
    Returns the radial coordinates, which is defined as

        rᵖ = |x|ᵖ + |y/q'|ᵖ

    where q' is the ellipsoidally corrected q value.

    Parameters
    ----------
    x : Tensor
        The coordinate along the semi-major axis

    y : Tensor
        The coordinate along the semi-minor axis

    q : Tensor
        Flattening parameter representing the ratio of the semi-minor to semi-
        major axis (b/a).

    q0 : Tensor, optional
        Axis ratio correction for an obolate spheroid disk. Some typical values
        are 0.13 or 0.20. Default is None.

    p : Tensor
        Generalized ellipse parameter, rᵖ = |x|ᵖ + |y/q|ᵖ. If not a tensor, it
        will be converted to one.

    Returns
    -------
    r : Tensor
        The radial coordinates

    Examples
    --------
    import matplotlib.pyplot as plt
    from galkit.spatial import grid, coordinate

    ckeys = {
        'h0': 0,
        'w0': 0,
        'scale': 1,
        'flip_lr': True,
        'flip_ud': False
    }

    q  = 0.5
    q0 = 0.2
    pa = 0
    p  = [1, 2, 3]

    h, w = coordinate.cartesian(
        grid=grid.pytorch_grid(100,100),
        **ckeys
    )
    x, y = coordinate.rotate(h, w, angle=-pa)
    θ = coordinate.θ_ellipse(x=x, y=y, q=q, q0=q0)
    r = coordinate.r_ellipse(x=x, y=y, q=q, q0=q0, p=p)

    fig, ax = plt.subplots(ncols=len(p))
    for i,ps in enumerate(p):
        ax[i].imshow(θ.squeeze())
        ax[i].contour(r[i])
        ax[i].set_title(f'p = {ps}')
    fig.tight_layout()
    fig.show()
    """
    p = to_tensor(p, device=x.device)
    q = q_correction(q, q0, device=x.device)
    return (                            # For backpropagation issues
        safe_divisor(x).abs().pow(p) + \
        safe_divisor(y).abs().div(q).pow(p)
    ).pow(1/p)

def rotate(
    x     : torch.Tensor, 
    y     : torch.Tensor, 
    angle : torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Rotates (x,y) coordinates by the angle `angle`. When dealing with position angles,
    angle = -pa.

    Parameters
    ----------
    x : Tensor
        The coordinate along the semi-major axis

    y : Tensor
        The coordinate along the semi-minor axis

    angle : Tensor
        Angle to rotate by.

    Returns
    -------
    x' : array
        The x-coordinates in the rotated frame

    y' : array
        The y-coordinates in the rotated frame

    Examples
    --------
    import matplotlib.pyplot as plt
    from numpy import deg2rad
    from galkit.spatial import grid, coordinate

    h, w = grid.pytorch_grid(100,100)
    x, y = coordinate.rotate(h, w, deg2rad(45))

    fig, ax = plt.subplots(ncols=2)
    c0 = ax[0].imshow(x.squeeze())
    c1 = ax[1].imshow(y.squeeze())
    fig.colorbar(c0, ax=ax[0], orientation='horizontal')
    fig.colorbar(c1, ax=ax[1], orientation='horizontal')    
    fig.tight_layout()   
    fig.show()
    """
    if not isinstance(angle, torch.Tensor) or angle.ndim != 3:
        angle = to_tensor(angle, device=x.device)
    cos = angle.cos()
    sin = angle.sin()
    x_new = x*cos - y*sin
    y_new = x*sin + y*cos
    return x_new, y_new

def q_correction(
    q  : torch.Tensor,
    q0 : Optional[torch.Tensor] = None,
    device : Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Computes the ellipsoidally corrected value

        (q')² ≡ cos²(i) = [(b/a)² - q0²] / [1 - q0²]

    where q ≡ b/a is the observed semi-minor to semi-major axis ratio.

    Parameters
    ----------
    q : Tensor
        Flattening parameter representing the ratio of the semi-minor to semi-
        major axis (b/a).

    q0 : Tensor, optional
        Axis ratio correction for an obolate spheroid disk. Some typical values
        are 0.13 or 0.20. Default is `None`, so no correction is applied.

    device : torch.device, optional
        The device to generate the data on. Only used if q is not a tensor.

    Returns
    -------
    q' : float
        The ellipsoidally corrected inclination parameter such that cos(i) = q'
        holds.

    Examples
    --------
    from galkit.spatial.coordinate import q_correction

    b = 0.2
    a = 1.0
    q0 = 0.13

    q = q_correction(b/a, q0)
    print(q)
    """
    if not isinstance(q, torch.Tensor) or q.ndim != 3:
        q = to_tensor(q, device=device)
    if q0 is not None:
        if not isinstance(q0, torch.Tensor):
            q0 = to_tensor(q0, device=q.device)
        q0_sq = q0.pow(2)
        q  = (q.pow(2) - q0_sq).div(1 - q0_sq).sqrt()
    return q