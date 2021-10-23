"""
This module contains functions useful for downloading imaging data from SDSS.

Functions
---------
fits_cutout(ra, dec, fov, shape, plate_scale, filter_bands)
    Interface to the HIPS2FITS service for SDSS FITS imaging. Returns an HDUL
    object containing the photometric data for each filter band.

jpeg_cutout(ra, dec, shape, scale, fov, dr, flip_lr, flip_ud)
    Interface to the SDSS SkyServer ImgCutout service. Returns a JPEG image of
    the sky over the desired region.
"""
import numpy
import urllib
from astropy.io import fits
from skimage.io import imread
from typing import Iterable, Optional, Tuple, Union
from ..hips import hips2fits_cutout

def fits_cutout(
    ra           : float,
    dec          : float,
    fov          : float,
    shape        : Optional[Union[int, Tuple[int,int]]] = None,
    plate_scale  : Optional[float] = None,
    filter_bands : Iterable = 'ugriz',
) -> fits.HDUList:
    """
    Interface to the HIPS2FITS service for SDSS FITS imaging. Returns an HDUL
    object containing the photometric data for each filter band.

    Parameters
    ----------
    ra : float
        The right ascension of the image center in degrees.

    dec : float
        The declination of the image center in degrees.

    fov : float
        The field of view in arcseconds.

    shape : int, Tuple[int, int]
        The desired image shape. Can be either a single integer, or an Iterable
        containing the image dimensions. May optional set the `plate_scale`
        parameter instead, which calculates the desired image size.

    plate_scale : float
        The pixel scale of the image (arcseconds/pixel). The image size is 
        calculated as round(fov / plate_scale) if passed. For SDSS, 0.396
        arcseconds/pixel is the original scale.

    filter_bands : iterable
        The filter bands to return cutout images for. Valid options are:
        {u, g, r, i, z}

    Returns
    -------
    data : HDUList
        An HDUList object containing the data for each imaging band. Each
        array is stored in its own extension with the name being the filter
        band.

    Examples
    --------
    import matplotlib.pyplot as plt
    from galkit.data.sdss import fits_cutout
    from numpy import arcsinh

    data = fits_cutout(
        ra  = 202.4696,
        dec = 47.1953,
        shape = (128,156),
        fov = 700,
        filter_bands = 'r'
    )

    fig, ax = plt.subplots()
    ax.imshow(arcsinh(data['r'].data))
    fig.show()
    """
    if (plate_scale is not None) and (shape is None):
        shape = int(round(fov/plate_scale))
    elif (plate_scale is None) and (shape is None):
        raise Exception("Cannot set both plate_scale and shape to be None")
    elif (plate_scale is not None) and (shape is not None):
        raise Exception("Cannot set both plate_scale and shape to be not None")

    hdul = fits.HDUList([fits.PrimaryHDU()])
    for band in filter_bands:
        hdu = hips2fits_cutout(ra=ra, dec=dec, fov=fov, shape=shape, hips=f'CDS/P/SDSS9/{band}', format='fits')
        hdu = fits.ImageHDU(data=hdu[0].data, header=hdu[0].header, name=band)
        hdul.append(hdu)
    return hdul

def jpeg_cutout(
    ra      : float, 
    dec     : float, 
    shape   : Union[int, Tuple[int,int]], 
    scale   : Optional[float] = None, 
    fov     : Optional[float] = None, 
    dr      : int = 16, 
    flip_lr : bool = False, 
    flip_ud : bool = False,
) -> numpy.ndarray:
    """
    Interface to the SDSS SkyServer ImgCutout service. Returns a JPEG image of
    the sky over the desired region.

    Parameters
    ----------
    ra : float
        The right ascension of the image center in degrees.

    dec : float
        The declination of the image center in degrees.

    shape : int, Tuple[int, int]
        The desired image shape. Can be either a single integer, or an Iterable
        containing the image dimensions.

    scale : float, optional
        The desired image scale (arcseconds / pixel).

    fov : float
        The field of view in arcseconds. If passed, then the scale is calculated
        as scale = fov / max(shape)

    dr : int
        Data release version of SDSS

    flip_lr : boolean
        Boolean indicating whether to flip the image left-right.

    flip_ud : boolean
        Boolean indicating whether to flip the image up-down. If trying to align
        the image with MaNGA / CALIFA data, then this should be set to True.

    Returns
    -------
    image : array
        A JPEG image array of the desired region on the sky

    Examples
    --------
    import matplotlib.pyplot as plt
    from galkit.data.sdss import jpeg_cutout

    jpeg = jpeg_cutout(
        ra  = 202.4696,
        dec = 47.1953,
        shape = (128,156),
        fov = 700,
    )

    fig, ax = plt.subplots()
    ax.imshow(jpeg)
    fig.show()
    """
    if (scale is None) and (fov is None):
        raise Exception("both scale and fov cannot be None")
    if (scale is not None) and (fov is not None):
        raise Exception("both scale and fov cannot be not None")
    if fov is not None:
        scale = fov / numpy.max(shape)  # Use numpy to avoid error with max(integer_value)

    query_string = urllib.parse.urlencode(dict(
        ra     = ra,
        dec    = dec,
        height = shape if isinstance(shape, int) else shape[0],
        width  = shape if isinstance(shape, int) else shape[1],
        scale  = scale
    ))

    url = f'http://skyserver.sdss.org/dr{dr}/SkyServerWS/ImgCutout/getjpeg'
    url = url + '?' + query_string

    image = imread(url)
    if flip_ud: image = numpy.flipud(image)
    if flip_lr: image = numpy.fliplr(image)
    return image