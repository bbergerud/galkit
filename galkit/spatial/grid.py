'''
This module contains functions for generating a base coordinate grid as
well as functions for converting from one coordinate grid to another.
The functions in this module serve as a base for inputting into functions
from the coordinate module.

Implemented grids are pixel, pytorch, and normalized.

Classes
-------
Grid
    Interface to the base grid with methods for converting
    from one coordinate system to another.

NormalizedGrid
    Interface to the normalized grid. Inherits the Grid class.

PixelGrid
    Interface to the pixel grid. Inherits the Grid class.

PytorchGrid
    Interface to the pytorch grid. Inherits the Grid class.

Functions
---------
_parse_grid(grid, dense)
    Function for generating a sparse or dense grid representation
    of the base 1D tensors.

normalized_grid(*shape, dense, device)
    Returns the pixel coordinates of the image normalized so that the top left
    is (0,0) and the bottom right is (+1,+1).

normalized_to_pixel_grid(grid, shape)
    Converts a normalized grid to its pixel equivalent.

normalized_to_pytorch_grid(grid, shape)
    Converts a normalized grid to its pytorch equivalent.

pixel_grid(*shape, dense, device)
    Returns the pixel coordinates of the image, with (0,0) corresponding to the
    top left.

pixel_to_normalized_grid(grid, shape)
    Converts a pixel grid to its normalized equivalent.

pixel_to_pytorch_grid(grid, shape)
    Converts a pixel grid to its pytorch equivalent.

pytorch_grid(*shape, dense, device)
    Returns the pixel coordinates of the image normalized so that the top left
    is (-1,-1) and the bottom right (+1,+1).

pytorch_to_pixel_grid(grid, shape)
    Converts a pytorch grid to its pixel equivalent.

pytorch_to_normalized_grid(grid, shape)
    Converts a pytorch grid to its normalized equivalent.
'''
import torch
from dataclasses import dataclass
from typing import Optional, Tuple
from ..utils import flatten

def _parse_grid(
    grid  : Tuple[torch.Tensor], 
    dense : bool
) -> Tuple[torch.Tensor]:
    """
    Function for generating a sparse or dense grid representation
    of the base 1D tensors.

    Parameters
    ----------
    grid : Tuple[Tensor]
        A sequence of 1D tensors containing the grid values along
        each dimension.

    dense : bool
        Boolean indicating whether to return the dense representation (True)
        or the sparse representation (False).

    Returns
    -------
    grid : Tuple[Tensors]
        The sparse or dense representation of the input tensors. An additional
        dimension has been added to the beginning of each tensor to represent
        the channels.

    Examples
    --------
    from galkit.spatial import grid
    import torch

    shape = [5, 10, 15]
    x = [torch.linspace(0,1,s) for s in shape]

    print('Sparse Grid')
    sparse = grid._parse_grid(x, dense=False)
    for gi in sparse:
        print(gi.shape)

    print('Dense Grid')
    dense = grid._parse_grid(x, dense=True)
    for gi in dense:
        print(gi.shape)
    """
    if dense:
        grid = torch.meshgrid(*grid)
    else:
        n = len(grid)
        grid = tuple(
            t.view(*(-1 if j==i else 1 for j in range(n)))
            for i,t in enumerate(grid)
        )
    return tuple(t.unsqueeze(0) for t in grid)

def normalized_grid(
    *shape,
    dense  : bool = False,
    device : Optional[torch.device] = None,
) -> Tuple[torch.Tensor]:
    """
    Returns the pixel coordinates of the image normalized so that the top left
    is (0,0) and the bottom right is (+1,+1).

    Parameters
    ----------
    shape : Iterable[int]
        A collection of integer objects representing the shape parameters. Can
        be a list-like collection or a sequence of integer inputs.

    dense : bool
        Boolean indicating whether to return the dense representation (True)
        or the sparse representation (False).

    device : torch.device
        The device to generate the tensors on.

    Returns
    -------
    grid : Tuple[Tensors]
        A sequence of arrays representing the grid coordinates

    Examples
    --------
    from galkit.spatial import grid
    import matplotlib.pyplot as plt

    h, w = grid.normalized_grid(100, 100, dense=True)
    x, y, z = grid.normalized_grid(100, 100, 100, dense=False)

    fig, ax = plt.subplots(ncols=2)
    c0 = ax[0].imshow(h[0])
    c1 = ax[1].imshow(w[0])
    fig.colorbar(c0, ax=ax[0], orientation='horizontal')
    fig.colorbar(c1, ax=ax[1], orientation='horizontal')
    fig.tight_layout()
    fig.show()

    print(x.shape, y.shape, z.shape)
    """
    shape = tuple(flatten(shape))
    grid  = tuple(torch.linspace(0,1,s,device=device) for s in shape)
    return _parse_grid(grid=grid, dense=dense)

def normalized_to_pixel_grid(
    grid  : Tuple[torch.Tensor],
    shape : Tuple[int]
) -> Tuple[torch.Tensor]:
    """
    Converts a normalized grid to its pixel equivalent.

    Parameters
    ----------
    grid: Tuple[Tensor]
        An iterable object that contains the grid coordinates

    shape: Tuple[int]
        An iterable object that contains the grid dimensions

    Returns
    -------
    grid : Tuple[Tensor]
        A sequence of arrays representing the pixel grid coordinates

    Examples
    --------
    from galkit.spatial import grid

    center = [0.5, 0.5]
    shape  = (100, 100)

    center_in_pixels = grid.normalized_to_pixel_grid(grid=center, shape=shape)
    print(center_in_pixels)
    """
    return tuple(c*(s-1) for c,s in zip(grid, shape))

def normalized_to_pytorch_grid(
    grid  : Tuple[torch.Tensor],
    shape : Tuple[int]
) -> Tuple[torch.Tensor]:
    """
    Converts a normalized grid to its pytorch equivalent.

    Parameters
    ----------
    grid: Tuple[Tensor]
        An iterable object that contains the grid coordinates

    shape: Tuple[int]
        An iterable object that contains the grid dimensions

    Returns
    -------
    grid : Tuple[Tensor]
        A sequence of arrays representing the pytorch grid coordinates

    Examples
    --------
    from galkit.spatial import grid

    center = [0.5, 0.5]
    shape  = (100, 100)

    center_in_pixels = grid.normalized_to_pytorch_grid(grid=center, shape=shape)
    print(center_in_pixels)
    """
    return tuple(2*c-1 for c,s in zip(grid, shape))

def pixel_grid(
    *shape,
    dense  : bool = False,
    device : Optional[torch.device] = None,
) -> Tuple[torch.Tensor]:
    """
    Returns the pixel coordinates of the image, with (0,0) corresponding to the
    top left.

    Parameters
    ----------
    shape : Iterable[int]
        A collection of integer objects representing the shape parameters. Can
        be a list-like collection or a sequence of integer inputs.

    dense : bool
        Boolean indicating whether to return the dense representation (True)
        or the sparse representation (False).

    device : torch.device
        The device to generate the tensors on.

    Returns
    -------
    grid : Tuple[Tensors]
        A sequence of arrays representing the grid coordinates

    Examples
    --------
    from galkit.spatial import grid
    import matplotlib.pyplot as plt

    h, w = grid.pixel_grid(100, 100, dense=True)
    x, y, z = grid.pixel_grid(100, 100, 100, dense=False)

    fig, ax = plt.subplots(ncols=2)
    c0 = ax[0].imshow(h[0])
    c1 = ax[1].imshow(w[0])
    fig.colorbar(c0, ax=ax[0], orientation='horizontal')
    fig.colorbar(c1, ax=ax[1], orientation='horizontal')
    fig.tight_layout()
    fig.show()

    print(x.shape, y.shape, z.shape)
    """
    shape = tuple(flatten(shape))
    grid  = tuple(torch.arange(s, device=device) for s in shape)
    return _parse_grid(grid, dense=dense)

def pixel_to_normalized_grid(
    grid  : Tuple[torch.Tensor],
    shape : Tuple[int]
) -> Tuple[torch.Tensor]:
    """
    Converts a pixel grid to its normalized equivalent.

    Parameters
    ----------
    grid: Tuple[Tensor]
        An iterable object that contains the grid coordinates

    shape: Tuple[int]
        An iterable object that contains the grid dimensions

    Returns
    -------
    grid : Tuple[Tensor]
        A sequence of arrays representing the normalized grid coordinates

    Examples
    --------
    from galkit.spatial import grid

    center = [49.5, 49.5]
    shape  = (100, 100)

    center_in_normalized = grid.pixel_to_normalized_grid(grid=center, shape=shape)
    print(center_in_normalized)
    """
    return tuple(x/(s-1) for x,s in zip(grid, shape))

def pixel_to_pytorch_grid(
    grid  : Tuple[torch.Tensor],
    shape : Tuple[int]
) -> Tuple[torch.Tensor]:
    """
    Converts a pixel grid to its pytorch equivalent.

    Parameters
    ----------
    grid: Tuple[Tensor]
        An iterable object that contains the grid coordinates

    shape: Tuple[int]
        An iterable object that contains the grid dimensions

    Returns
    -------
    grid : Tuple[Tensor]
        A sequence of arrays representing the pytorch grid coordinates

    Examples
    --------
    from galkit.spatial import grid

    center = [49.5, 49.5]
    shape  = (100, 100)

    center_in_pytorch = grid.pixel_to_pytorch_grid(grid=center, shape=shape)
    print(center_in_pytorch)
    """
    return tuple(2*(x/(s-1) - 0.5) for x,s in zip(grid, shape))

def pytorch_grid(
    *shape, 
    dense  : bool = False,
    device : Optional[torch.device] = None,
) -> Tuple[torch.Tensor]:
    """
    Returns the pixel coordinates of the image normalized so that the top left
    is (-1,-1) and the bottom right (+1,+1).

    Parameters
    ----------
    shape : Iterable[ints]
        A collection of integer objects representing the shape parameters. Can
        be a list-like collection or a sequence of integer inputs.

    dense : boolean [Optional]
        Boolean indicating whether to return the dense grid with the dimensions
        specified by shape, or to return a sparse representation. Default is False.

    device : torch.device
        The device to generate the tensors on.

    Returns
    -------
    grid : Tuple[Tensor]
        A sequence of Tensors representing the grid coordinates.

    Examples
    --------
    from galkit.spatial import grid
    import matplotlib.pyplot as plt

    # Plot a dense representation
    h, w = grid.pytorch_grid((100, 100), dense=True)

    fig, ax = plt.subplots(ncols=2)
    c0 = ax[0].imshow(h.squeeze())
    c1 = ax[1].imshow(w.squeeze())
    fig.colorbar(c0, ax=ax[0], orientation='horizontal')
    fig.colorbar(c1, ax=ax[1], orientation='horizontal')
    fig.tight_layout()
    fig.show()

    x, y, z = grid.pytorch_grid(10,10,10)
    for a in (x,y,z):
        print(a.shape)
    """
    shape = tuple(flatten(shape))
    grid  = tuple(torch.linspace(-1,1,s, device=device) for s in shape)
    return _parse_grid(grid, dense=dense)

def pytorch_to_normalized_grid(
    grid  : Tuple[torch.Tensor],
    shape : Tuple[int]
) -> Tuple[torch.Tensor]:
    """
    Converts a pytorch grid to its normalized equivalent.

    Parameters
    ----------
    grid: Tuple[Tensor]
        An iterable object that contains the grid coordinates

    shape: Tuple[int]
        An iterable object that contains the grid dimensions

    Returns
    -------
    grid : Tuple[Tensor]
        A sequence of tensors representing the normalized grid coordinates

    Examples
    --------
    from galkit.spatial import grid

    center = [0.0, 0.0]
    shape  = (100, 100)

    center_in_pixels = grid.pytorch_to_normalized_grid(grid=center, shape=shape)
    print(center_in_pixels)
    """
    return tuple(0.5*(c+1) for c,s in zip(grid, shape))

def pytorch_to_pixel_grid(
    grid  : Tuple[torch.Tensor],
    shape : Tuple[int]
) -> Tuple[torch.Tensor]:
    """
    Converts a pytorch grid to its pixel equivalent.

    Parameters
    ----------
    grid: Tuple[Tensor]
        An iterable object that contains the grid coordinates

    shape: Tuple[int]
        An iterable object that contains the grid dimensions

    Returns
    -------
    grid : Tuple[Tensor]
        A sequence of tensors representing the pixel grid coordinates

    Examples
    --------
    from galkit.spatial import grid

    center = [0.0, 0.0]
    shape  = (100, 100)

    center_in_pixels = grid.pytorch_to_pixel_grid(grid=center, shape=shape)
    print(center_in_pixels)
    """
    return tuple((c/2+0.5)*(s-1) for c,s in zip(grid, shape))

@dataclass
class Grid:
    """
    Interface to the base grid with methods for converting
    from one coordinate system to another.

    Parameters
    ----------
    base_grid
        A function for generating the base coordinate grid.

    to_normalized_grid
        A function that converts from the base coordinate
        grid to the normalized coordinate grid.
    
    to_pixel_grid
        A function that converts from the base coordinate
        grid to the pixel coordinate grid.

    to_pytorch_grid
        A function that converts from the base coordinate
        grid to the pytorch coordinate grid.

    from_normalized_grid
        A function that converts from the normalized grid
        to the base coordinate grid.

    from_pixel_grid
        A function that converts from the pixel grid
        to the base coordinate grid.

    from_pytorch_grid
        A function that converts from the pytorch grid
        to the base coordinate grid.

    Methods
    -------
    __call__(self, *args, **kwargs)
        Interface to the base_grid function.
    """
    base_grid            : callable
    to_normalized_grid   : callable
    to_pixel_grid        : callable
    to_pytorch_grid      : callable
    from_normalized_grid : callable
    from_pixel_grid      : callable
    from_pytorch_grid    : callable

    def __call__(self, *args, **kwargs):
        return self.base_grid(*args, **kwargs)

class NormalizedGrid(Grid):
    def __init__(self):
        self.base_grid = normalized_grid
        self.to_normalized_grid = lambda grid, shape: grid
        self.to_pixel_grid = normalized_to_pixel_grid
        self.to_pytorch_grid = normalized_to_pytorch_grid
        self.from_normalized_grid = lambda grid, shape: grid
        self.from_pixel_grid = pixel_to_normalized_grid
        self.from_pytorch_grid = pytorch_to_normalized_grid

class PixelGrid(Grid):
    def __init__(self):
        self.base_grid = pixel_grid
        self.to_normalized_grid = pixel_to_normalized_grid
        self.to_pixel_grid = lambda grid, shape: grid
        self.to_pytorch_grid = pixel_to_pytorch_grid
        self.from_normalized_grid = normalized_to_pixel_grid
        self.from_pixel_grid = lambda grid, shape: grid
        self.from_pytorch_grid = pytorch_to_pixel_grid

class PytorchGrid(Grid):
    def __init__(self):
        self.base_grid = pytorch_grid
        self.to_normalized_grid = pytorch_to_normalized_grid
        self.to_pixel_grid = pytorch_to_pixel_grid
        self.to_pytorch_grid = lambda grid, shape: grid
        self.from_normalized_grid = normalized_to_pytorch_grid
        self.from_pixel_grid = pixel_to_pytorch_grid
        self.from_pytorch_grid = lambda grid, shape: grid