# GeoTIFF Utilities

## Overview
Working with geotiff files in Python is difficult due to a lack of proper support for interacting with them beyond basic functionality like reading their image data.
Even the documentation for GDAL, a library providing extensive functionality for interacting with geotiff files, specifically mentions that there is no official documentation for using the Python bindings.
This code is designed to provide users detailed functionality for working with geotiff files directly in Python without the need to use confusing low level APIs.
This repository is particularly intended for workloads involving machine learning and deep learning.
Functionality offered by this repository includes: 
1. Reading pixels from geotiff
1. Writing pixels to geotiff
1. Reading and writing pixels in chunks to enable the use of massive geotiffs too large to be loaded into RAM all at once
1. Creating a new geotiff containing metadata cloned from another geotiff file
1. Reading geotiff metadata
1. Writing geotiff metadata
1. Performing coordinate and geospatial transform calculations

## Environment
This repository uses [GDAL](https://gdal.org/index.html) internally to interact with geotiff files.
A suitable programming environment must be set up to use Python with GDAL.
Setting up GDAL with Python can be tricky, especially in Windows.
There are three methods you can try to set up a working environment:
1. You can try to set up your own environment using conda or pip and installing GDAL yourself.
2. You can try the using the included environment.yml file as a starting point and customize your environment from there.
3. On Windows (where GDAL can be particularly tricky) you could use one of the included whl files to install GDAL into your environment.


## Code
The geotiff utilities are provided in geotiff_utils.py.
All of the functions have documentation explaining their inputs and outputs.
If you want to verify your environment is working correctly...
1. Download the test dataset from [here](https://drive.google.com/drive/u/3/folders/1OxkkxI5o6K4Jlur5UI7Pw0GmSVLv1irw)
2. Place its contents in tests/test_dataset (no subdirectories)
3. Run **python tests/tests_geotiff_utils.py** (from the top level directory of this repository)
