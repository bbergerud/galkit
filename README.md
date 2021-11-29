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

## Functional
Collection of various functions.

## Spatial
Collection of methods for generating projected geometries. The grid module is used to allow for flexible base grids, such as a pixel-based grid or pytorch's grid.