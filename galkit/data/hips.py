"""
This module contains methods useful for working with the HiPS cutout service.

Functions
---------
hips2fits_cutout(ra, dec, shape, fov, hips, format, projection, cache, **kwargs)
    Interface to the hips2fits cutout service.
"""
import numpy
from astropy.io import fits
from skimage.io import imread
from typing import Tuple, Union
from urllib.parse import urlencode
from ..utils import HiddenPrints

def hips2fits_cutout(
    ra         : float,
    dec        : float,
    shape      : Union[int, Tuple[int,int]],
    fov        : float,
    hips       : str,
    format     : str,
    projection : str = 'TAN',
    cache      : bool = False,
    **kwargs
) -> Union[fits.HDUList, numpy.ndarray]:
    """
    Interface to the hips2fits cutout service.

    Parameters
    ----------
    ra : float
        The right ascension of the image center in degrees.

    dec : float
        The declination of the image center in degrees.

    shape : int, Iterable[int,int]
        The desired image shape. Can be either a single integer, or an Iterable
        containing the image dimensions.

    fov : float
        The field of view in arcseconds.

    hips: str
        Keyword identifying the HiPS to use. See the "HiPS sky maps" at 
        https://aladin.u-strasbg.fr/hips/list for the different possibilities.

    format: str
        The image format. See the link under `hips` for the available formats based
        on the designated `hips`. Note that while "jpeg" may be listed as an appropriate
        format, it should be passed in as "jpg" rather than "jpeg".

    projection : str
        The name of the projection to use.

    cache : bool
        Boolean to indicate whether to cache the downloaded data if returning a FITS
        file.

    **kwargs
        Any additional parameters that can be passed into the api

    Returns
    -------
    image : Union[HDUList, ndarray]
        The data image. The format will be an HDUList if requesting
        a FITS file, while a basic image format will be returned as
        a numpy array.

    Notes
    -----
    See http://alasky.u-strasbg.fr/hips-image-services/hips2fits for more
    information.

    Examples
    --------
    import matplotlib.pyplot as plt
    from galkit.data.hips import hips2fits_cutout
    from numpy import flipud

    # Image parameters
    kwargs = {
        'ra': 202.4696,
        'dec': 47.1953,
        'shape': (128,156),
        'fov': 700,
    }

    # JPEG image of M51
    jpeg = hips2fits_cutout(
        hips='CDS/P/SDSS9/color-alt',
        format='jpg',
        **kwargs,
    )

    # GALEX Near-UV FITS image of M51
    hdu = hips2fits_cutout(
        hips='CDS/P/GALEXGR6/AIS/NUV',
        format='fits',          
        **kwargs
    )

    fig, ax = plt.subplots(ncols=2)
    ax[0].imshow(flipud(jpeg))
    ax[1].imshow(hdu[0].data)
    fig.show()
    """
    query_string = urlencode(dict(
        ra=ra,
        dec=dec,
        height=shape if isinstance(shape, int) else shape[0],
        width=shape if isinstance(shape, int) else shape[1],
        hips=hips,
        projection=projection,
        fov=fov/3600,   # Interface wants it in degrees
        format=format,
        **kwargs
    ))

    url = 'http://alasky.u-strasbg.fr/hips-image-services/hips2fits'
    url = url + '?' + query_string

    if format == 'fits':
        # Disable printing of the download status. Any bad urls will still be
        # caught and a message passed to the terminal.
        with HiddenPrints():
            image = fits.open(url, cache=cache)
    else:
        image = imread(url)

    return image
