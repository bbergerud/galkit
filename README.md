# Galkit
This is a library that holds routines that are commonly used by some of the more specialized libraries related to identifying spiral arms in galaxies. The library is designed to be friendly to both numpy / pytorch tensors, but the spatial modules are currently only set up to work with pytorch tensors. The structure is organized as follows.

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