# Galkit
This is a library that holds routines that are commonly used by some of the more specialized libraries related to identifying spiral arms in galaxies. The library is designed to be friendly to both numpy arrays and pytorch tensors, but the spatial modules are currently only set up to work with pytorch tensors. The structure is organized as follows.

```
.
└── galkit
    ├── data
    │   ├── hips.py
    │   └── sdss
    │       ├── imaging.py
    │       └── meta.py
    ├── functional
    │   ├── angle.py
    │   ├── convolution.py
    │   ├── magnitude.py
    │   ├── photometric.py
    │   └── transform.py
    ├── spatial
    │   ├── coordinate.py
    │   ├── grid.py
    │   └── resample.py
    └── utils.py
```

## Data
This section is designed to work with astronomical data. Currently it has methods that are an interface to the HIPS2FITS service and the SDSS SkyServer jpeg cutout service, along with some metadata about SDSS (Sloan Digital Sky Survey)

### [galkit/data/hips.py](./galkit/data/hips.py)
```
This module contains methods useful for working with the HiPS cutout service.

hips2fits_cutout(ra, dec, shape, fov, hips, format, projection, cache, **kwargs)
    Interface to the hips2fits cutout service.
```

### [galkit/data/sdss/imaging.py](./galkit/data/sdss/imaging.py)
```
This module contains functions useful for downloading imaging data from SDSS.

Functions
---------
fits_cutout(ra, dec, fov, shape, plate_scale, filter_bands)
    Interface to the HIPS2FITS service for SDSS FITS imaging. Returns an HDUL
    object containing the photometric data for each filter band.

jpeg_cutout(ra, dec, shape, scale, fov, dr, flip_lr, flip_ud)
    Interface to the SDSS SkyServer ImgCutout service. Returns a JPEG image of
    the sky over the desired region.
```

### [galkit/data/sdss/meta.py](./galkit/data/sdss/meta.py)
```
Storage of metadata related to SDSS

Variables
---------
sdss_b : Dict[str, float]
    Dictionary whose key represents the SDSS filter band and whose value
    contains the `b` parameter used when constructing the arcsinh magnitude.

sdss_nMgyPerCount : Dict[str, float]
    Dictionary whose key represents the SDSS filter band and whose value
    is the mean nanomaggies / count.

sdss_λ / sdss_wavelengths : Dict[str, float]
    Dictionary whose key represents the SDSS filter band and whose value
    contains the effective wavelength of the filter band in meters.
```
## Functional
Collection of various functions.

### [galkit/functional/angle.py](./galkit/functional/angle.py)
```
Methods for working with angles.

Functions
---------
angular_distance(x1, x2, absolute)
    Computes the angular distance between the azimuthal angles x1 and x2

mod_angle(θ, Δθ, deg)
    Adds the angles and returns the modulus such that -π ≤ θ < π
    for radians and -180° ≤ θ < 180° for degrees.
```
### [galkit/functional/convolution.py](./galkit/functional/convolution.py)
```
Profiles for modeling the point-spread function (PSF).

Classes
-------
Gaussian
    Class interface to the gaussian function. Also stores the
    FWHM of the profile.

Moffat
    Class interface to the moffat function. Also stores the
    FWHM of the profile.

DoubleGaussianPowerlaw
    Class interface to the double_guassian_powerlaw function.
    Also stores the FWHM of the profile.

Functions
---------
get_kernel2d(kernel_size, profile, device, oversample)
    Returns a 2D kernel representation of the provided profile.

get_kernel2d_fwhm(profile, default_size, device, oversample)
    Returns a 2D kernel representation of the provided profile
    out to the indicated number of FWHM.

double_gaussian_powerlaw(r, b, beta, p0, sigma1, sigma2, sigmaP, normalize)
    Double gaussian + powerlaw profile used by SDSS for modeling the PSF,
        f(r) = [exp(-r²/(2σ₁²)) + b⋅exp(-r²/(2σ₂²)) + p₀[1 + r²/(β⋅σₚ²)]^(-β/2)] / (1 + b + p0)

gaussian(r, sigma, normalize)
    Gaussian profile,
        1/sqrt(2πσ²) ⋅ exp(-r² / (2σ²))

moffat(r, core_width, power_index, normalize)
    Moffat Profile,
        f(r;α,β) = (β-1)/(πα²)⋅[1 + (r²/α²)]⁻ᵝ
    where α is the core width of the profile and β the power index.
```


### [galkit/functional/magnitude.py](./galkit/functional/magnitude.py)
```
Methods for converting between flux and magnitude.

Functions
---------
pogson_flux2mag(flux, filter_band, m0)
    Converts flux to Pogson magnitudes.

pogson_mag2flux(magnitude, filter_band, m0)
    Converts Pogson magnitudes to flux.

sdss_flux2mag(flux, filter_band)
    Converts nanomaggies to arcsinh magnitudes.

sdss_mag2flux(magnitude, filter_band)
    Converts arcsinh magnitudes to nanomaggies.
```

### [galkit/functional/photometric.py](./galkit/functional/photometric.py)
```
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
```

### [galkit/functional/transform.py](./galkit/functional/transform.py)
```
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

normalize(input, loc, scale, eps)
    Normalizes the input by subtracting off the mean and dividing by the
    standard deviation. The resulting output is then multiplied by the
    scale and shifted to have a mean `loc`.

rescale(input, lower, upper)
    Rescales the input to have values within the range [lower, upper]

sigmoid(input, loc, scale)
    Logistic sigmoid function,
        output = 1 / (1 + exp(-z))
    where z = (input - loc) / scale.

to_tricolor(input, palette, transform)
    Converts a multichannels input into a tri-color image using
    the provided palette. Useful for converting instance masks to
    tri-color images.
```
## Spatial
Collection of methods for generating projected geometries. The grid module is used to allow for flexible base grids, such as a pixel-based grid or pytorch's grid.

### [galkit/spatial/coordinate.py](./galkit/spatial/coordinate.py)
```
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
```

### [galkit/spatial/grid.py](./galkit/spatial/grid.py)
```
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
```

### [galkit/spatial/resample.py](./galkit/spatial/resample.py)
```
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

to_cartesian(input, h0, w0, pa, q, q0, p, scale, flip_lr, flip_ud,
        transform, untransform, x_min, x_max, return_grid, dense_grid,
        alpha_shift, theta_shift, mode)
    Reprojects the input tensor from a polar coordinate system to a cartesian
    coordinate system.

to_polar(input, h0, w0, pa, q, q0, p, scale, flip_lr, flip_ud,
        transform, untransform, x_min, x_max, return_grid, dense_grid,
        alpha_shift, theta_shift, mode)
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
```

# [galkit/utils.py](./galkit/utils.py)
```
Functions that perform basic operations for parsing data.

Classes
-------
HiddenPrints()
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
```