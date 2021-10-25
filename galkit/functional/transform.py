"""
Transformation operations.

Functions
---------
arcsinh_stretch(input, lower, upper, scale, eps)
    Transforms the input by the operation
        f(x) = arcsinh((x-lower)/scale) / arcsinh((upper-lower)/scale)
    Values outside the range [lower, upper] are clipped to the range (0,1).

fits2jpeg(input, lower, upper, scale, gamma, gain, desaturate, 
          sharpen, kernel_size, kernel_sigma, max_output)
    Converts a FITS image into a JPEG image in a similar manner to the SDSS algorithm.
"""
import kornia
import math
import numpy
import torch
from typing import Optional, Tuple, Union

def arcsinh_stretch(
    input : Union[numpy.ndarray, torch.Tensor],
    lower : Optional[Union[callable, float]] = None,
    upper : Optional[Union[callable, float]] = None,
    scale : Union[callable, float] = 0.75,
    eps   : float = 1e-7
) -> Union[numpy.ndarray, torch.Tensor]:
    """
    Transforms the input by the operation

        f(x) = arcsinh((x-lower)/scale) / arcsinh((upper-lower)/scale)

    Values outside the range [lower, upper] are clipped to [0,1].

    Note that the input values for lower, upper, and scale and percentiles,
    with the values in the above equation calculated based on the indicated
    percentile.

    Parameters
    ----------
    input : Array, Tensor (C × H × W)
        The input to transform. Should be channels first. Can have dimensions
        of either (B × C × H × W) or (C × H × W).

    lower : callable, float
        The percentile to take for the lower bound of the tensor. If set to
        None, then it is set to a value of zero. Can also be a callable function
        that takes no inputs and returns a value.

    upper : callable, float, optional
        The percentile to take for the upper bound of the tensor. If set to
        None, then it is set to the maximum value of the tensor. Can also be
        a callable function that takes no inputs and returns a value.

    scale : callable, float
        The percentile to take for the smoothing parameter.  Can also be
        a callable function that takes no inputs and returns a value.

    eps : float
        Padding factor to prevent a division by zero.

    Returns
    -------
    output : Array, Tensor (C × H × W)
        The transformed imaged. Has a channels first orientation.

    Examples
    --------
    import matplotlib.pyplot as plt
    import torch
    from galkit.functional.photometric import exponential
    from galkit.functional.transform import arcsinh_stretch
    from galkit.spatial import grid, coordinate

    _, r = coordinate.polar(
        grid = grid.pytorch_grid(100,100),
        q = 0.5,
    )

    # Flux image
    flux_g = exponential(r, amplitude=1.0, scale=0.25)
    flux_r = exponential(r, amplitude=1.5, scale=0.20)
    flux_i = exponential(r, amplitude=1.9, scale=0.15)
    flux = torch.cat([flux_i, flux_r, flux_g], dim=0)

    image = arcsinh_stretch(flux, lower=None, upper=None, scale=0.75)

    fig, ax = plt.subplots(ncols=2)
    ax[0].imshow(flux.squeeze().T)
    ax[0].set_title('Original Image')
    ax[1].imshow(image.squeeze().T)
    ax[1].set_title('Arcsinh Stretch')
    fig.tight_layout()
    fig.show()
    """
    is_tensor = isinstance(input, torch.Tensor)
    quantile = torch.quantile if is_tensor else numpy.quantile
    arcsinh  = torch.arcsinh if is_tensor else numpy.arcsinh
    ndim = input.ndim

    kwargs = {
        'input' if is_tensor else 'a'  : input.flatten(1),
        'dim' if is_tensor else 'axis' : 1 if ndim == 4 else None,
    }

    lower = 0 if lower is None else quantile(q = lower() if callable(lower) else lower, **kwargs)
    if upper is None:
        if ndim == 4:
            upper = input.flatten(1).max(1)
            if is_tensor:
                upper = upper.values
        else:
            upper = input.max()
    else:
        upper = quantile(q = upper() if callable(upper) else upper, **kwargs)
    scale = quantile(q = scale() if callable(scale) else scale, **kwargs) + eps

    num = arcsinh((input - lower) / scale)
    den = arcsinh((upper - lower) / scale) + eps
    return (num / den).clip(0,1)

def fits2jpeg(
    input        : Union[numpy.ndarray, torch.Tensor],
    lower        : Union[callable, float] = 0.1,
    upper        : Union[callable, float] = 10000,
    scale        : Union[callable, float] = 6,
    gamma        : Optional[Union[callable, tuple]] = None,
    gain         : Optional[tuple] = None,
    desaturate   : bool = False,
    sharpen      : bool = False,
    kernel_size  : Tuple[int,int] = (3,3),
    kernel_sigma : Tuple[float,float] = (1,1),
    max_output   : float = 1,
) -> Union[numpy.ndarray, torch.Tensor]:
    """
    Converts a FITS image into a JPEG image in a similar manner to the SDSS algorithm.

    Parameters
    ----------
    input : Array, Tensor
        The input to transform. Should be channels first. Can have dimensions
        of either (B × C × H × W) or (C × H × W).

    lower : callable, float
        The lower bound on the flux counts for setting values to 0.

    upper : callable, float
        The upper blound on the flux counts for setting values to 1.

    scale : callable, float
        The smoothing parameter.

    gamma : float
        The color correction factor. If the flux has been converted to counts
        via nMgyPerCount, then setting it to zero should be sufficient.

    gain : tuple, optional
        The gain factors for the r,g,b filters. Note that while the SDSS code
        uses gain=[0.9,1.1,1.8], if the flux has been converted to counts via
        nMgyPerCount, then setting it to one should be sufficient. If set to
        None, then no gain is applied. Default is None.

    desaturate : bool
        Boolean indicating to apply a convolution to smooth out the saturated
        regions. Default is False.

    sharpen : bool
        Boolean indicating to apply an unsharp mask. Default is False.

    kernel_size : Tuple[int,int]
        The kernel size of the convolution in pixels. Defaults is (3,3).

    kernel_sigma : Tuple[float,float]
        The standard deviation of the kernel in pixels. Default is (1,1).

    max_output : float
        The maximum output value to rescale the jpeg (0-255) values to.
        Default is 1.

    Returns
    -------
    image : Tensor
        The converted jpeg image of the input file. Note that a float type is
        still used while the values have been rescaled to the range (0,max_output).

    Examples
    --------
    import torch
    import matplotlib.pyplot as plt
    from galkit.functional.photometric import exponential
    from galkit.functional.transform import fits2jpeg
    from galkit.spatial import grid, coordinate

    _, r = coordinate.polar(
        grid = grid.pytorch_grid(100,100),
        q = 0.5,
    )

    # Flux image
    flux_g = exponential(r, amplitude=1.0, scale=0.25)
    flux_r = exponential(r, amplitude=1.5, scale=0.20)
    flux_i = exponential(r, amplitude=1.9, scale=0.15)
    flux = torch.cat([flux_i, flux_r, flux_g], dim=0)

    # Convert to counts
    fpc  = torch.tensor([0.0065, 0.005, 0.004]).view(-1,1,1)
    cnts = flux / fpc

    imag = fits2jpeg(cnts, gain=[1.1, 1.0, 0.9])

    fig, ax = plt.subplots(ncols=2)
    ax[0].imshow(flux.T)
    ax[0].set_title('Original Image')
    ax[1].imshow(imag.T)
    ax[1].set_title('fits2jpeg')
    fig.tight_layout()
    fig.show()
    """
    ndim = input.ndim
    is_tensor = isinstance(input, torch.Tensor)
    if not is_tensor:
        input = torch.tensor(input)

    # Define blur for use in desaturation
    blur = lambda x: kornia.filters.gaussian_blur2d(
        input       = x if x.ndim == 4 else x.expand(1,-1,-1,-1),
        kernel_size = kernel_size,
        sigma       = kernel_sigma,
        border_type = 'replicate'
    )

    image = blur(input).clip(min=0)

    # Mean value across the different channels per pixel
    channel_mean_per_pixel = image.mean(1, keepdims=True)

    # Scaling factors
    if callable(upper): upper = upper()
    if callable(lower): lower = lower()
    if callable(scale): scale = scale()

    slope = 255 / math.asinh((upper - lower) / scale)
    scaledMean = slope * torch.arcsinh((channel_mean_per_pixel - lower) / scale)
    scale = scaledMean / (channel_mean_per_pixel + 0.01)
    image = image * scale

    if gain is not None:
        gain = torch.as_tensor(gain, device=image.device).float().view(1,-1,1,1)
        image = image * gain

    if sharpen:
        image = kornia.filters.unsharp_mask(
            input       = image,
            kernel_size = kernel_size,
            sigma       = kernel_sigma,
            border_type = 'replicate'
        )

    if gamma is not None:
        image[:,0] += gamma * (image[:,0] - image[:,1])     # Δred
        image[:,2] += gamma * (image[:,2] - image[:,1])     # Δblue

    # Desaturate Option
    if desaturate:
        # Find the places where there is a pixel value above
        imax = image.max(1, keepdims=True).values

        mask = imax > 255
        if mask.any():
            mask = mask.repeat_interleave(3, 1)
            temp = image.clone()
            temp[mask] =  temp[mask] * (255 / imax).repeat_interleave(3,1)[mask]
            temp = blur(temp)
            image[mask] = temp[mask]

    # Cast to jpeg values and new range
    image = image.clip(min=0, max=255).round() * (max_output / 255)

    # Convert back to 3 dimensions if the input was such
    if ndim == 3:
        image = image.squeeze(0)

    return image if is_tensor else image.numpy()