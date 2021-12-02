"""
Functions that perform basic operations for parsing data.

Classes
-------
FunctionWrapper
    Function wrapper that stores default parameter values.

HiddenPrints
    Method for preventing a function from printing values to the terminal.

Methods
-------
flatten(items, *exceptions)
    Takes a list of objects and returns a generator containg the flattened
    expression. Any dictionary objects are set to return the values rather
    than the keys.

parse_parameter(input, length)
    Takes a value or list-like collection of values and parses
    the input to make sure it has the proper size.

round_up_to_odd_integer(x)
    Rounds the input up to the nearest odd integer.

safe_divisor(input, buffer)
    Buffers the input by the specified amount. Where the input values
    are negative, the negative of the buffer factor is added.

to_tensor(input, device, view)
    Casts the input into a tensor with shape `view`.

unravel_index(index, shape)
    Unravels the index location based on the provided shape of the tensor.
"""

import os, sys
import numpy
import torch
from typing import Generator, Iterable, Optional, Tuple, Union

class FunctionWrapper:
    """
    Function wrapper that stores default parameter values.

    Attributes
    ----------
    function : callable
        The function to wrap.

    kwargs
        Additional keyword arguments to pass into the function
        upon calling.

    Methods
    -------
    __call__(*args, **kwargs)
        Interface to the function __call__ method. The arguments
        stored in the attribute `kwargs` are alse passed.
    """
    def __init__(self, function:callable, **kwargs):
        """
        Parameters
        ----------
        function : callable
            The function to wrap.

        **kwargs
            Additional keyword arguments to pass into the function
            upon calling.
        """
        self.function = function
        self.kwargs = kwargs
    
    def __call__(self, *args, **kwargs):
        """
        Interface to the function __call__ method. The arguments
        stored in the attribute `kwargs` are alse passed.

        Parameters
        ----------
        *args, **kwargs
            Additional arguments to pass into the function.      
        """
        return self.function(*args, **kwargs, **self.kwargs)

class HiddenPrints:
    """
    Method for preventing a function from printing values to the terminal.

    Examples
    --------
    from galkit.utils import HiddenPrints

    def foo():
        print(1)

    with HiddenPrints():
        for i in range(3):
            foo()
    """
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def flatten(
        items:Iterable, 
        *exceptions
    ) -> Generator:
    """
    Takes a list of objects and returns a generator containg the flattened
    expression. Any dictionary objects are set to return the values rather
    than the keys.

    Parameters
    ----------
    items : Iterable
        A sequence of items to flatten.

    *exceptions
        Object types to avoid further flattening. String and Byte types are
        automatically yielded.

    Returns
    -------
    output : Generator
        A generator for returning a flattened expression containing the values
        in `items`. These values can be extracted from the generator by passing
        the output through a function like `list` or `tuple`.

    Examples
    --------
    from galkit.utils import flatten

    items = [1, [2, 3, 4], {'a': 5, 'b': 6}, (7, 8), 9]
    print(tuple(flatten(items)))
    """
    for item in items:
        if isinstance(item, dict):
            yield from flatten(item.values(), *exceptions)
        elif isinstance(item, Iterable) and not isinstance(item, (str, bytes, *exceptions)):
            yield from flatten(item, *exceptions)
        else:
            yield item

def parse_parameter(
    input  : Union[int, float, Tuple], 
    length : int
) -> Tuple:
    """
    Takes a value or list-like collection of values and parses
    the input to make sure it has the proper size.

    Parameter
    ---------
    input : int, float, tuple
        The input values to parse. Should either be a single value
        or a collection of values.
    
    length : int
        The length of the output value.

    Returns
    -------
    output : tuple
        If input is a single value, then the value is returned as a tuple 
        with length `length`. Otherwise, the input value is returned if
        no exception is raised.

    Raises
    ------
    ValueError
        If input is an iterable but not of the proper length, then an
        exception is raised.

    Examples
    --------
    from galkit.utils import parse_parameter

    x = parse_parameter(1, 5)
    print(x)

    y = parse_parameter((1,2,3,4), 4)
    print(y)

    # exception
    z = parse_parameter([1,2,3], 4)
    """
    if isinstance(input, (list, tuple)):
        if len(input) != length:
            raise ValueError(f"len({input}) != {length}")
        return input
    else:
        return (input,) * length

def round_up_to_odd_integer(x:float) -> int:
    """
    Rounds the input up to the nearest odd integer.

    Parameters
    ----------
    x : int
        The value to round.
    
    Returns
    -------
    output : float
        The nearest integer rounded up.

    Examples
    --------
    from galkit.utils import round_up_to_odd_integer

    print(round_up_to_odd_integer(4.5))
    """
    return int(x+1) // 2 * 2 + 1

def safe_divisor(
    input  : torch.Tensor,
    eps    : float = 1e-8,
) -> torch.Tensor:
    """
    Buffers the input by the specified amount. Where the input values
    are negative, the negative of the buffer factor is added.

    Parameters
    ----------
    input : Tensor
        The input tensor to pad to avoid numerical issues
    
    eps : float
        The buffer factor to add to the input

    Returns
    -------
    output : Tensor
        The padded

    Examples
    --------
    import torch
    from galkit.utils import safe_divisor

    x = torch.tensor([-1e-7, 0, 1e-7])
    s = safe_divisor(x, eps=1e-7)
    print(s)
    """
    where = torch.where if isinstance(input, torch.Tensor) else numpy.where
    buffer = where(input >= 0, eps, -eps)
    return buffer + input

def to_tensor(
    input  : Union[float, Iterable],
    device : Optional[torch.device] = None,
    view   : Optional[tuple] = (-1,1,1)
) -> torch.Tensor:
    """
    Casts the input into a tensor with shape `view`.

    Parameters
    ----------
    input : Iterable
        The object to cast into a tensor

    device : torch.device, optional
        The device to generate the tensors on. Default is `None`.
    
    view : tuple, optional
        The desired output shape.

    Returns
    -------
    output : Tensor
        A tensor version of the input with shape `view`. If the input
        is not a tensor, then the data type is float32.
    
    Examples
    --------
    import torch
    from galkit.utils import to_tensor

    t = to_tensor([1,2,3], view=(-1,1,1))
    print(t, t.dtype)

    x = torch.tensor([1, 2, 3])
    t = to_tensor(x, view=(1,1,-1))
    print(t, t.dtype)
    """
    if not isinstance(input, torch.Tensor):
        input = torch.tensor(input, dtype=torch.float32, device=device)
    if view is not None:
        input = input.view(*view)
    return input

def unravel_index(
    index : torch.Tensor, 
    shape : Tuple,
) -> Tuple[torch.Tensor]:
    """
    Unravels the index location based on the provided shape of the tensor.

    Parameters
    ----------
    index : Tensor
        The index element. Should be an integer type.

    shape : Tuple
        The shape of the origin tensor

    Returns
    -------
    indices : Tuple
        A tuple containing the indices of the minimum value
        in the input tensor

    Examples
    --------
    import torch
    from galkit.utils import unravel_index

    loc = unravel_index(index=torch.tensor(7), shape=(3,3))
    print(loc)
    """
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = torch.div(index, dim, rounding_mode='floor')
    return tuple(reversed(out))