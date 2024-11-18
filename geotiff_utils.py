"""
This file provides functionality for working with geotiff files using gdal.

Reference 1: https://stackoverflow.com/questions/33537599/how-do-i-write-create-a-geotiff-rgb-image-file-in-python
Reference 2: https://gis.stackexchange.com/questions/164853/reading-modifying-and-writing-a-geotiff-with-gdal-in-python
Reference 3: https://gis.stackexchange.com/questions/172666/optimizing-python-gdal-readasarray
Reference 4: https://gis.stackexchange.com/questions/53617/how-to-find-lat-lon-values-for-every-pixel-in-a-geotiff-file
Reference 5: https://stackoverflow.com/questions/60954617/how-to-build-internal-overviews-with-python-gdal
-buildoverviews

Since GDAL is hard to use on windows, the most reliable installation method has been to install using pip from a whl
file here: https://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal
Or here: https://github.com/cgohlke/geospatial-wheels/releases/tag/v2023.7.16
"""


import inspect
import os
from time import sleep

from alive_progress import alive_bar
from collections import defaultdict
import numpy as np
from osgeo import gdal, ogr, osr
import torch

from utils.general_utils import check_required_and_keyword_args
from utils.numpy_utils import create_np_array_from_geotiff_bands, rescale_array, normalize_array, \
    convert_np_array_to_torch_format, convert_torch_tensor_to_np_array_format


def check_arguments(func):
    def wrapper(*args, **kwargs):

        check_required_and_keyword_args(func, args, kwargs)

        keywords = inspect.signature(func).parameters.keys()
        for keyword, arg in zip(keywords, args):
            kwargs[keyword] = arg

        check_keywords = ['geotiff_filepath',
                          'read_geotiff_filepath',
                          'write_geotiff_path',
                          'shape_filepath']
        for check_keyword in check_keywords:
            if check_keyword in kwargs:
                arg = kwargs[check_keyword]
                if not isinstance(arg, str):
                    raise AttributeError(f'Passed value for "{check_keyword}" was "{arg}" which is not a string.')
                if not os.path.exists(arg):
                    raise FileNotFoundError(
                        f'Passed value for "{check_keyword}" was "{arg}" which is a filepath that does not exist.')

        check_keyword = 'reference_geotiff_filepath'
        if check_keyword in kwargs:
            arg = kwargs[check_keyword]
            if arg is not None:
                if not isinstance(arg, str):
                    raise AttributeError(f'Passed value for "{check_keyword}" was "{arg}" which is not a string or '
                                         f'"None".')
                if not os.path.exists(arg):
                    raise FileNotFoundError(
                        f'Passed value for "{check_keyword}" was "{arg}" which is a filepath that does not exist.')
            else:
                required_args = ['width', 'height', 'num_bands', 'no_data_values', 'geo_transform', 'projection']
                for required_arg in required_args:
                    if kwargs[required_arg] is None:
                        raise AttributeError(f'If you do not pass a value for "{check_keyword}", you must pass an '
                                             f'argument for "{required_arg}".')

        check_keywords = ['create_geotiff_filepath', 'create_shape_filepath']
        for check_keyword in check_keywords:
            if check_keyword in kwargs:
                arg = kwargs[check_keyword]
                if not isinstance(arg, str):
                    raise AttributeError(f'Passed value for "{check_keyword}" was "{arg}" which is not a string.')
                if os.path.exists(arg):
                    raise FileNotFoundError(
                        f'Passed value for "{check_keyword}" was "{arg}" which is a filepath that already exists.')
                if not os.path.exists(os.path.dirname(arg)):
                    raise FileNotFoundError(
                        f'Passed value for "{check_keyword}" has directory path "{arg}" is "{os.path.dirname(arg)}" '
                        f'which already not exists.')

        check_keyword = 'data_array'
        if check_keyword in kwargs:
            arg = kwargs[check_keyword]
            if arg is not None and not (isinstance(arg, np.ndarray) or isinstance(arg, torch.Tensor)):
                raise AttributeError(
                    f'Passed value for "{check_keyword}" was "{arg}" which should be either "None", '
                    f'a numpy array with the color channel in the third position, '
                    f'or a pytorch tensor with the color channel in the first position.')
            if isinstance(arg, np.ndarray) and not len(arg.shape) == 3:
                raise AttributeError(f'When passing a numpy array for "{check_keyword}", it should have '
                                     f'three dimensions.\n'
                                     f'Instead, the numpy array you passed has "{len(arg.shape)}" dimensions.')
            if isinstance(arg, np.ndarray) and (arg.shape[2] > arg.shape[0] or arg.shape[2] > arg.shape[1]):
                raise AttributeError(f'When passing a numpy array for "{check_keyword}", the color channel must '
                                     f'be in the third dimension.\n'
                                     f'It looks like your color channel is not in the third dimension.\n'
                                     f'The shape of the passed numpy array was "{arg.shape}".')
            if isinstance(arg, torch.Tensor) and not len(arg.size()) == 3:
                raise AttributeError(f'When passing a torch tensor for "{check_keyword}", it should have '
                                     f'three dimensions.\n'
                                     f'Instead, the torch tensor you passed has "{len(arg.shape)}" dimensions.')
            if isinstance(arg, torch.Tensor) and (arg.size()[0] > arg.size()[1] or arg.size()[0] > arg.size()[2]):
                raise AttributeError(f'When passing a torch tensor for "{check_keyword}", the color channel must '
                                     f'be in the first dimension.\n'
                                     f'It looks like your color channel is not in the first dimension.\n'
                                     f'The shape of the passed pytorch tensor was "{arg.size()}".')

        check_keywords = ['coordinates',
                          'array_coordinates',
                          'grid_coordinates',
                          'size',
                          'top_left_coordinates',
                          'bottom_right_coordinates']
        for check_keyword in check_keywords:
            if check_keyword in kwargs:
                arg = kwargs[check_keyword]
                if arg is not None and not (isinstance(arg, tuple)
                                            and len(arg) == 2
                                            and all(isinstance(n, int) for n in arg)):
                    raise AttributeError(f'The passed value for "{check_keyword}" was "{arg}" which is invalid.\n'
                                         f'The value for "{check_keyword}" should "None" or '
                                         f'a tuple containing two integers.')

        check_keywords = ['lat_lon', 'top_left_lat_lon', 'bottom_right_lat_lon']
        for check_keyword in check_keywords:
            if check_keyword in kwargs:
                arg = kwargs[check_keyword]
                if not (isinstance(arg, tuple) and len(arg) == 2 and all(isinstance(n, (int, float)) for n in arg)):
                    raise AttributeError(f'Passed value for "{check_keyword}" was "{arg}".\n'
                                         f'It must be a tuple of length two containing either ints or floats.')

        check_keywords = ['convert_to_pytorch_tensor_format', 'verbose']
        for check_keyword in check_keywords:
            if check_keyword in kwargs:
                arg = kwargs[check_keyword]
                if not isinstance(arg, bool):
                    raise AttributeError(f'Passed value for "{check_keyword}" must be a boolean.')

        check_keywords = ['replace_no_data_value', 'replace_nan_with']
        for check_keyword in check_keywords:
            if check_keyword in kwargs:
                arg = kwargs[check_keyword]
                if not (arg is None or isinstance(arg, (int, float))):
                    raise AttributeError(f'Passed value for "{check_keyword}" was "{arg}".\n'
                                         f'The value must be either "None", an integer, or a float.')

        check_keyword = 'rescale'
        if check_keyword in kwargs:
            arg = kwargs[check_keyword]
            if not (arg is None or
                    isinstance(arg, (int, float)) or
                    isinstance(arg, tuple) and len(arg) == 2 and all(isinstance(n, int) for n in arg)):
                raise AttributeError(f'Passed value for "{check_keyword}" was "{arg}".\n'
                                     f'The value must be either "None", '
                                     f'an int or float, '
                                     f'or a tuple containing two integers.')

        check_keyword = 'normalize'
        if check_keyword in kwargs:
            arg = kwargs[check_keyword]
            if not (arg is None or
                    isinstance(arg, bool) or
                    isinstance(arg, tuple) and len(arg) == 2 and all(isinstance(n, (int, float)) for n in arg)):
                raise AttributeError(f'Passed value for "{check_keyword}" was "{arg}".\n'
                                     f'The value must be either "None", '
                                     f'a boolean, '
                                     f'or a tuple containing two ints or floats.')

        check_keywords = ['chunk_size', 'num_tries']
        for check_keyword in check_keywords:
            if check_keyword in kwargs:
                arg = kwargs[check_keyword]
                if not isinstance(arg, int):
                    raise AttributeError(f'Value for "{check_keyword}" was "{arg}" which is not an integer.')
                if not arg > 0:
                    raise AttributeError(f'Value for "{check_keyword}" was "{arg}" which is not greater than 0.')

        check_keywords = ['width', 'height', 'num_bands']
        for check_keyword in check_keywords:
            if check_keyword in kwargs:
                arg = kwargs[check_keyword]
                if arg is not None:
                    if not isinstance(arg, int):
                        raise AttributeError(f'Value for "{check_keyword}" was "{arg}" which is not an integer.')
                    if not arg > 0:
                        raise AttributeError(f'Value for "{check_keyword}" was "{arg}" which is not greater than 0.')

        check_keyword = 'geo_transform'
        if check_keyword in kwargs:
            arg = kwargs[check_keyword]
            if not (arg is None or
                    (isinstance(arg, tuple) and len(arg) == 6 and all(isinstance(n, float) for n in arg))):
                raise AttributeError(f'Value for "{check_keyword}" must be a tuple containing 6 floats.')

        check_keyword = 'projection'
        if check_keyword in kwargs:
            arg = kwargs[check_keyword]
            if not (arg is None or isinstance(arg, str)):
                raise AttributeError(f'Value for "{check_keyword}" should be a string.')

        check_keyword1 = 'num_bands'
        check_keyword2 = 'no_data_values'
        check_keyword1_exists = check_keyword1 in kwargs and kwargs[check_keyword1] is not None
        check_keyword2_exists = check_keyword2 in kwargs and kwargs[check_keyword2] is not None
        if check_keyword1_exists or check_keyword2_exists:
            if not (check_keyword1_exists and check_keyword2_exists):
                if check_keyword1_exists:
                    raise AttributeError(f'If a value is passed for "{check_keyword1}", you must pass a list for the '
                                         f'"{check_keyword2}" as a list of no data values at each band.')
                if check_keyword2_exists:
                    raise AttributeError(f'If a value is passed for "{check_keyword2}", you must pass am integer for '
                                         f'"{check_keyword1}" indicating the number of raster bands.')
            else:
                arg = kwargs[check_keyword2]
                if not (len(arg) == kwargs[check_keyword1] and all(isinstance(n, int) for n in arg)):
                    raise AttributeError(f'The value for "{check_keyword2}" must be a list of integers whose length is '
                                         f'equal to the "{check_keyword1}" argument.')

        if 'top_left_coordinates' in kwargs and 'bottom_right_coordinates' in kwargs:
            top_left_coordinates = kwargs['top_left_coordinates']
            bottom_right_coordinates = kwargs['bottom_right_coordinates']
            if top_left_coordinates is not None and bottom_right_coordinates is not None:
                if not top_left_coordinates[0] < bottom_right_coordinates[0] \
                        and top_left_coordinates[1] < bottom_right_coordinates[1]:
                    raise AttributeError(f'The passed top left coordinates at "{top_left_coordinates}" were not to the '
                                         f'top left of the top right coordinates at "{bottom_right_coordinates}".')

        if 'top_left_lat_lon' in kwargs and 'bottom_right_lat_lon' in kwargs:
            geotiff_filepath = None
            if 'read_geotiff_filepath' in kwargs:
                geotiff_filepath = kwargs['read_geotiff_filepath']
            elif 'write_geotiff_filepath' in kwargs:
                geotiff_filepath = kwargs['write_geotiff_filepath']
            elif 'geotiff_filepath' in kwargs:
                geotiff_filepath = kwargs['geotiff_filepath']
            if geotiff_filepath:
                top_left_coordinates = kwargs['top_left_lat_lon']
                bottom_right_coordinates = kwargs['bottom_right_lat_lon']
                top_left_grid_coordinates = get_grid_coordinates_from_lat_lon(geotiff_filepath, top_left_coordinates)
                bottom_right_grid_coordinates = \
                    get_grid_coordinates_from_lat_lon(geotiff_filepath, bottom_right_coordinates)
                top_left_array_coordinates = \
                    get_array_coordinates_from_grid_coordinates(geotiff_filepath, top_left_grid_coordinates)
                bottom_right_array_coordinates = \
                    get_array_coordinates_from_grid_coordinates(geotiff_filepath, bottom_right_grid_coordinates)
                if not top_left_array_coordinates[0] < bottom_right_array_coordinates[0] \
                        and top_left_array_coordinates[1] < bottom_right_array_coordinates[1]:
                    raise AttributeError(f'The passed top left coordinates at "{top_left_coordinates}" were not to the '
                                         f'top left of the top right coordinates at "{bottom_right_coordinates}".')

        if 'coordinates' in kwargs or 'size' in kwargs:
            if 'coordinates' not in kwargs:
                check_coords = (0, 0)
            else:
                check_coords = kwargs['coordinates']
            if not (check_coords[0] >= 0 and check_coords[1] >= 0):
                raise AttributeError(f'Coordinates should be greater than or equal to zero. '
                                     f'Instead got "{check_coords}".')
            if 'size' in kwargs:
                geotiff_filepath = None
                if 'read_geotiff_filepath' in kwargs:
                    geotiff_filepath = kwargs['read_geotiff_filepath']
                elif 'write_geotiff_filepath' in kwargs:
                    geotiff_filepath = kwargs['write_geotiff_filepath']
                elif 'geotiff_filepath' in kwargs:
                    geotiff_filepath = kwargs['geotiff_filepath']
                check_size = get_geotiff_size(geotiff_filepath)
                if not (kwargs['size'][0] >= 0 and kwargs['size'][1] >= 0):
                    raise AttributeError(f'Size should be greater than or equal to zero. '
                                         f'Instead got "{check_size}".')
                bottom_right_coords = (check_coords[0] + kwargs['size'][0], check_coords[1] + kwargs['size'][1])
                if bottom_right_coords[0] > check_size[0] or bottom_right_coords[1] > check_size[1]:
                    raise AttributeError(f'Value of coordinates + size must be less than the total size of the raster.'
                                         f'The passed coordinates + size is "{bottom_right_coords}" but the bottom '
                                         f'right of the raster is "{(check_size[0], check_size[1])}".')

        return func(**kwargs)

    return wrapper


@check_arguments
def read_from_geotiff(read_geotiff_filepath,
                      coordinates=None,
                      size=None,
                      convert_to_pytorch_tensor_format=True,
                      replace_no_data_value_with=None,
                      replace_nan_with=None,
                      rescale=None,
                      normalize=None):
    """
    Reads the data from the passed geotiff file at the passed raster band, coordinates, and size (if specified).
    The coordinates and size parameters should be (x, y) tuples. Either both or neither of these parameters should be
    None.
    If "convert_to_pytorch_tensor_format" is True, a pytorch tensor will be returned instead of a numpy array,
    and the color
    channel will be automatically moved to the 1st dimension, otherwise returns a numpy array with the color channel
    in the third dimension.
    If "replace_no_data_value_with" is equal to a number, the value marked as the "no data" value by the geotiff
    file will be replaced with the passed value in the returned data array.
    If "replace_nan_with" is not None, any "nan" values in the read array will be replaced with the passed integer.
    If "rescale" in None, the data array will not be rescaled.
    If "rescale" is a floating point number, the data array will be scaled by the ratio indicated by that number.
    If "rescale" is a tuple, the data array will be scaled to the (width, height) indicated by those numbers.
    If "normalize" is None, the data array will not be normalized.
    If "normalize" is True, the data array will be normalized from 0 to 1.
    If "normalize" is set to a tuple of (min, max), those values will be used to calculate the normalized values.
    """

    # open the geotiff and find the number of bands
    read_arrays = []
    read_file = gdal.Open(read_geotiff_filepath)
    num_bands = read_file.RasterCount

    # read in each band as an array and append it to the list of read arrays, close the geotiff after done reading data
    for band_number in range(1, num_bands + 1):
        band = read_file.GetRasterBand(band_number)
        if not coordinates and not size:
            data_array = band.ReadAsArray()
        elif coordinates and size:
            data_array = band.ReadAsArray(*coordinates, *size)
        else:
            raise ValueError(f'Either both the coordinates and size parameter should have values or both should be set '
                             f'to "None".\n Instead, the coordinates parameters was set to: '
                             f'{coordinates} and size was set to: {size}.')
        read_arrays.append(data_array)
    del read_file

    # concatenate all the bands together to form the full data array
    data_array = create_np_array_from_geotiff_bands(read_arrays)

    # replace no data values in the read array as appropriate
    if replace_no_data_value_with is not None:
        no_data_value = get_geotiff_no_data_values(read_geotiff_filepath)
        data_array[data_array == no_data_value] = replace_no_data_value_with

    # replace nan values in the read array as appropriate
    if replace_nan_with is not None:
        data_array[np.isnan(data_array)] = replace_nan_with

    # rescale the data array as appropriate
    if rescale is not None:
        if isinstance(rescale, int):
            data_array = rescale_array(data_array, rescale_ratio=rescale, replace_nan_with=replace_nan_with)
        elif isinstance(rescale, tuple):
            data_array = rescale_array(data_array, dimensions=rescale, replace_nan_with=replace_nan_with)

    # normalize the data array as appropriate
    if normalize is not None:
        if isinstance(normalize, bool) and normalize:
            data_array = normalize_array(data_array)
        elif isinstance(normalize, tuple):
            data_array = normalize_array(data_array, min_max=normalize)

    # pytorch expects the color channel first, whereas the geotiff loads the color channel last, so we convert formats
    if convert_to_pytorch_tensor_format:
        data_array = convert_np_array_to_torch_format(data_array)

    return data_array


@check_arguments
def create_geotiff(create_geotiff_filepath,
                   reference_geotiff_filepath=None,
                   data_array=None,
                   top_left_coordinates=None,
                   bottom_right_coordinates=None,
                   width=None,
                   height=None,
                   num_bands=None,
                   no_data_values=None,
                   geo_transform=None,
                   projection=None):
    """
    Creates a geotiff file with and saves it with the passed file path.
    The projection and coordinates will be matched to the geotiff file at "reference_file".
    The parameter "data_array" is an optional numpy array or pytorch tensor that will be written to the file.
    The coordinates parameters are (x, y) tuples which indicates where to place the numpy array relative to the
    reference file.
    If only "top_left_coordinates" is specified, these coordinates will be used to determine where the data array is
    placed.
    If "bottom_right_coordinates" is also specified the passed data array will be rescaled so it can be written
    between the two sets of passed coordinates.
    """

    # set the gdal configuration so a single, large file can be written to the disk
    gdal.SetConfigOption('CHECK_DISK_FREE_SPACE', 'FALSE')

    # determine the parameters for the new geotiff using the reference file or overridden by any passed parameters
    if reference_geotiff_filepath is not None:
        reference_file = gdal.Open(reference_geotiff_filepath)
        ref_width, ref_height, ref_num_bands = get_geotiff_size(reference_geotiff_filepath)
        if width is None:
            width = ref_width
        if height is None:
            height = ref_height
        if num_bands is None:
            num_bands = ref_num_bands
        if no_data_values is None:
            no_data_values = []
            for band_number in range(get_geotiff_raster_band_count(reference_geotiff_filepath)):
                reference_band = reference_file.GetRasterBand(band_number + 1)
                reference_no_data_value = reference_band.GetNoDataValue()
                if reference_no_data_value:
                    reference_no_data_value = int(reference_no_data_value)
                else:
                    reference_no_data_value = 0
                no_data_values.append(reference_no_data_value)
        assert len(no_data_values) == num_bands
        if geo_transform is None:
            geo_transform = reference_file.GetGeoTransform()
        if projection is None:
            projection = reference_file.GetProjection()
        del reference_file

    # create the new geotiff
    driver = gdal.GetDriverByName("GTiff")
    create_file = driver.Create(create_geotiff_filepath, width, height, num_bands, gdal.GDT_Float32)
    create_file.SetGeoTransform(geo_transform)
    create_file.SetProjection(projection)
    for band_number, no_data_value in enumerate(no_data_values):
        create_band = create_file.GetRasterBand(band_number + 1)
        create_band.SetNoDataValue(no_data_value)
    create_file.FlushCache()
    del create_file

    # write any passed data to the new geotiff
    if data_array is not None:
        write_to_geotiff(create_geotiff_filepath,
                         data_array=data_array,
                         top_left_coordinates=top_left_coordinates,
                         bottom_right_coordinates=bottom_right_coordinates)


@check_arguments
def write_to_geotiff(write_geotiff_filepath,
                     data_array,
                     top_left_coordinates=None,
                     bottom_right_coordinates=None,
                     num_tries=3):
    """
    The parameter "write_geotiff_filepath" indicates the path to the geotiff file to be written to.
    The parameter "data_array" is an optional numpy array or pytorch tensor that will be written to the file.
    The coordinates parameters are (x, y) tuples which indicates where to place the numpy array relative to the
    reference file.
    If only "top_left_coordinates" is specified, these coordinates will be used to determine where the data array is
    placed.
    If "bottom_right_coordinates" is also specified the passed data array will be rescaled so it can be written
    between the two sets of passed coordinates.
    The parameter "num_tries" in an integer that indicates how many times to try writing to the geotiff file again
    if the file is currently in use. This option exists because sometimes gdal doesn't free up the file right away.
    """

    if isinstance(data_array, torch.Tensor):
        data_array = convert_torch_tensor_to_np_array_format(data_array.cpu())

    success = False
    while not success and num_tries > 0:
        try:
            write_file = gdal.Open(write_geotiff_filepath, gdal.GA_Update)
            for band_number in range(data_array.shape[2]):
                write_band = write_file.GetRasterBand(band_number + 1)
                if top_left_coordinates:
                    if bottom_right_coordinates:
                        width = bottom_right_coordinates[0] - top_left_coordinates[0]
                        height = bottom_right_coordinates[1] - top_left_coordinates[1]
                        data_array = rescale_array(data_array, dimensions=(width, height))
                    write_band.WriteArray(data_array[:, :, band_number], *top_left_coordinates)
                else:
                    write_band.WriteArray(data_array[:, :, band_number])
                write_band.FlushCache()
            del write_file
            success = True
        except AttributeError as e:
            num_tries -= 1
            if num_tries == 0:
                raise e
            else:
                print(f'Encountered error when attempting to write to "{write_geotiff_filepath}".\n'
                      f'This is probably because the file is still in use from a previous operation.\n'
                      f'Will try "{num_tries}" more times before raising exception.')
                sleep(1)


@check_arguments
def write_chunks_to_geotiff(write_geotiff_filepath,
                            read_geotiff_filepath,
                            top_left_coordinates=None,
                            bottom_right_coordinates=None,
                            chunk_size=10000,
                            num_tries=3,
                            verbose=False):
    """
    The parameter "write_geotiff_filepath" indicates the path to a geotiff file to which data will be written in chunks.
    The parameter "read_geotiff_filepath" indicates the path to a geotiff file to which data will be read in chunks.
    Each chunk is placed at the corresponding location in the geotiff file at the write filepath.
    The parameter "data_array" is an optional numpy array or pytorch tensor that will be written to the file.
    The coordinates parameters are (x, y) tuples which indicates where to place the numpy array relative to the
    reference file.
    If "top_left_coordinates" is specified, only data after it will be written. Otherise the top left coordinates will
    be the top left of the data array.
    If "bottom_right_coordinates" are specified, data below and to the right of the passed coordinates will not be
    written. Otherwise, the bottom right coordinates will the be bottom right of the entire data array.
    will be written to the new file.
    The parameter "chunk_size" indicates the width and height of the chunks that should be read from one geotiff and
    written to the other.
    The parameter "num_tries" in an integer that indicates how many times to try writing to the geotiff file again
    if the file is currently in use. This option exists because sometimes gdal doesn't free up the file right away.
    The "verbose" parameter indicates if text will be printed indicating the progress of iterating through the process
    of reading and writing the chunks.
    """

    generator = _iterate_geotiff(read_geotiff_filepath,
                                 top_left_coordinates=top_left_coordinates,
                                 bottom_right_coordinates=bottom_right_coordinates,
                                 chunk_size=chunk_size,
                                 verbose=verbose)

    for x, y, x_chunk_size, y_chunk_size in generator:
        tiled_data = read_from_geotiff(read_geotiff_filepath, coordinates=(x, y), size=(x_chunk_size, y_chunk_size))
        write_to_geotiff(write_geotiff_filepath, tiled_data, top_left_coordinates=(x, y), num_tries=num_tries)


@check_arguments
def read_geotiff_within_lat_lon(read_geotiff_filepath,
                                top_left_lat_lon,
                                bottom_right_lat_lon,
                                convert_to_pytorch_tensor_format=False,
                                replace_no_data_value_with=False,
                                replace_nan_with=None,
                                rescale=None,
                                normalize=None):
    """
    Reads the data from the passed geotiff file at the passed raster band between the passed latitude and longitude
    coordinates.
    The top left and bottom right latitude and longitudes are a tuples of (lat, lon).
    Raster band should be an integer from 1 to N where N represents the number of raster bands.
    If "convert_to_pytorch_tensor_format" is True, a pytorch tensor will be returned instead of a numpy array,
    and the color
    channel will be automatically moved to the 1st dimension, otherwise returns a numpy array with the color channel
    in the third dimension.
    If "replace_no_data_value_with" is equal to a number, the value marked as the "no data" value by the geotiff
    file will be replaced with the passed value in the returned data array.
    If "replace_nan_with" is not None, any "nan" values in the read array will be replaced with the passed integer.
    If "rescale" in None, the data array will not be rescaled.
    If "rescale" is a floating point number, the data array will be scaled by the ratio indicated by that number.
    If "rescale" is a tuple, the data array will be scaled to the (width, height) indicated by those numbers.
    If "normalize" is None, the data array will not be normalized.
    If "normalize" is True, the data array will be normalized from 0 to 1.
    If "normalize" is set to a tuple of (min, max), those values will be used to calculate the normalized values.
    """

    top_left_grid_coordinates = get_grid_coordinates_from_lat_lon(read_geotiff_filepath, top_left_lat_lon)
    bottom_right_grid_coordinates = get_grid_coordinates_from_lat_lon(read_geotiff_filepath, bottom_right_lat_lon)

    top_left_array_coordinates = \
        get_array_coordinates_from_grid_coordinates(read_geotiff_filepath, top_left_grid_coordinates)
    bottom_right_array_coordinates = \
        get_array_coordinates_from_grid_coordinates(read_geotiff_filepath, bottom_right_grid_coordinates)

    size = tuple([tup[1] - tup[0] for tup in zip(top_left_array_coordinates, bottom_right_array_coordinates)])

    return read_from_geotiff(read_geotiff_filepath,
                             coordinates=top_left_array_coordinates,
                             size=size,
                             convert_to_pytorch_tensor_format=convert_to_pytorch_tensor_format,
                             replace_no_data_value_with=replace_no_data_value_with,
                             replace_nan_with=replace_nan_with,
                             rescale=rescale,
                             normalize=normalize)


@check_arguments
def write_to_geotiff_within_lat_lon(write_geotiff_filepath,
                                    data_array,
                                    top_left_lat_lon,
                                    bottom_right_lat_lon,
                                    num_tries=3):
    """
    The parameter "write_geotiff_filepath" indicates the path to the geotiff file to be written to.
    The parameter "data_array" is an optional numpy array or pytorch tensor that will be written to the file.
    The coordinates parameters are (x, y) tuples which indicates where to place the data array relative to the
    reference file.
    The parameters "top_left_lat_lon" and "bottom_right_lat_lon" indicate the latitudes and longitudes between which
    the data should be written. The data will be automatically rescaled to fit between these coordinates.
    The parameter "num_tries" in an integer that indicates how many times to try writing to the geotiff file again
    if the file is currently in use. This option exists because sometimes gdal doesn't free up the file right away.
    """

    top_left_grid_coordinates = get_grid_coordinates_from_lat_lon(write_geotiff_filepath, top_left_lat_lon)
    bottom_right_grid_coordinates = get_grid_coordinates_from_lat_lon(write_geotiff_filepath, bottom_right_lat_lon)

    top_left_array_coordinates = get_array_coordinates_from_grid_coordinates(write_geotiff_filepath,
                                                                             top_left_grid_coordinates)
    bottom_right_array_coordinates = get_array_coordinates_from_grid_coordinates(write_geotiff_filepath,
                                                                                 bottom_right_grid_coordinates)

    write_to_geotiff(write_geotiff_filepath,
                     data_array,
                     top_left_coordinates=top_left_array_coordinates,
                     bottom_right_coordinates=bottom_right_array_coordinates,
                     num_tries=num_tries)


@check_arguments
def create_shapefile_from_geotiff(geotiff_filepath, create_shape_filepath):
    """
    Returns the path to a shapefile containing a polygon outlining the bounds of the raster data in the passed
    geotiff file.

    Modified from: https://www.programcreek.com/python/example/97859/osgeo.ogr.GetDriverByName
    """

    type_mapping = {gdal.GDT_Byte: ogr.OFTInteger,
                    gdal.GDT_UInt16: ogr.OFTInteger,
                    gdal.GDT_Int16: ogr.OFTInteger,
                    gdal.GDT_UInt32: ogr.OFTInteger,
                    gdal.GDT_Int32: ogr.OFTInteger,
                    gdal.GDT_Float32: ogr.OFTReal,
                    gdal.GDT_Float64: ogr.OFTReal,
                    gdal.GDT_CInt16: ogr.OFTInteger,
                    gdal.GDT_CInt32: ogr.OFTInteger,
                    gdal.GDT_CFloat32: ogr.OFTReal,
                    gdal.GDT_CFloat64: ogr.OFTReal}

    ds = gdal.Open(geotiff_filepath)
    prj = ds.GetProjection()
    srcband = ds.GetRasterBand(1)
    dst_layername = "Shape"
    drv = ogr.GetDriverByName("ESRI Shapefile")
    dst_ds = drv.CreateDataSource(create_shape_filepath)
    srs = osr.SpatialReference(wkt=prj)

    dst_layer = dst_ds.CreateLayer(dst_layername, srs=srs)
    raster_field = ogr.FieldDefn('id', type_mapping[srcband.DataType])
    dst_layer.CreateField(raster_field)
    gdal.Polygonize(srcband, srcband, dst_layer, 0, [], callback=None)
    del ds, srcband, dst_ds, dst_layer


@check_arguments
def dissolve_shapefile(shape_filepath, create_shape_filepath):
    """Saves a new shapefile which contains a single polygon surrounding all the polygons in the shape filepath."""

    def createDS(ds_name, ds_format, geom_type, srs):
        drv = ogr.GetDriverByName(ds_format)
        ds = drv.CreateDataSource(ds_name)
        lyr_name = os.path.splitext(os.path.basename(ds_name))[0]
        lyr = ds.CreateLayer(lyr_name, srs, geom_type)
        return ds, lyr

    def dissolve(input, output, multipoly=False):
        ds = ogr.Open(input)
        lyr = ds.GetLayer()
        out_ds, out_lyr = createDS(output, ds.GetDriver().GetName(), lyr.GetGeomType(), lyr.GetSpatialRef())
        defn = out_lyr.GetLayerDefn()
        multi = ogr.Geometry(ogr.wkbMultiPolygon)
        for feat in lyr:
            if feat.geometry():
                feat.geometry().CloseRings()  # this copies the first point to the end
                wkt = feat.geometry().ExportToWkt()
                multi.AddGeometryDirectly(ogr.CreateGeometryFromWkt(wkt))
        union = multi.UnionCascaded()
        if multipoly is False:
            for geom in union:
                poly = ogr.CreateGeometryFromWkb(geom.ExportToWkb())
                feat = ogr.Feature(defn)
                feat.SetGeometry(poly)
                out_lyr.CreateFeature(feat)
        else:
            out_feat = ogr.Feature(defn)
            out_feat.SetGeometry(union)
            out_lyr.CreateFeature(out_feat)
            out_ds.Destroy()
        ds.Destroy()
        return True

    dissolve(shape_filepath, create_shape_filepath)


@check_arguments
def get_geotiff_size(geotiff_filepath):
    """
    Returns a tuple indicating the width, height, and num_bands in the raster image.
    """

    geotiff_file = gdal.Open(geotiff_filepath)
    rows = geotiff_file.RasterYSize
    cols = geotiff_file.RasterXSize
    num_bands = geotiff_file.RasterCount
    del geotiff_file

    return cols, rows, num_bands


@check_arguments
def get_geotiff_raster_band_count(geotiff_filepath):
    """Returns an integer indicating how many raster bands are in the geotiff file."""

    geotiff_file = gdal.Open(geotiff_filepath)
    num_bands = geotiff_file.RasterCount
    del geotiff_file

    return num_bands


@check_arguments
def get_geotiff_grid_edge_locations(geotiff_filepath):
    """
    Returns a tuple indicating the upper left, upper right, lower left, and lower right coordinate in the geotiff grid.
    """

    geotiff_file = gdal.Open(geotiff_filepath)
    ulx, xres, xskew, uly, yskew, yres = geotiff_file.GetGeoTransform()
    lrx = ulx + (geotiff_file.RasterXSize * xres)
    lry = uly + (geotiff_file.RasterYSize * yres)
    del geotiff_file

    return (round(ulx), round(uly)), (round(lrx), round(lry))


@check_arguments
def get_geotiff_lat_lon_edge_locations(geotiff_filepath):
    """
    Returns a tuple indicating the latitude and longitude of the upper left and lower right locations in the geotiff
    grid.
    """

    (ulx, uly), (lrx, lry) = get_geotiff_grid_edge_locations(geotiff_filepath)
    edge_1 = get_lat_lon_from_grid_coordinates(geotiff_filepath, (ulx, uly))
    edge_2 = get_lat_lon_from_grid_coordinates(geotiff_filepath, (lrx, lry))

    return edge_1, edge_2


@check_arguments
def get_geotiff_epsg(geotiff_filepath):
    """Returns the EPSG code for the passed geotiff file."""

    geotiff_file = gdal.Open(geotiff_filepath)
    proj = osr.SpatialReference(wkt=geotiff_file.GetProjection())
    epsg = proj.GetAttrValue('AUTHORITY', 1)
    epsg = int(epsg)
    del geotiff_file

    return epsg


@check_arguments
def get_geotiff_no_data_values(geotiff_filepath):
    """Returns a list of the no data values for each raster bands the passed geotiff file."""

    no_data_values = []
    geotiff_file = gdal.Open(geotiff_filepath)
    num_bands = geotiff_file.RasterCount
    for band_number in range(num_bands):
        band = geotiff_file.GetRasterBand(band_number + 1)
        no_data_value = band.GetNoDataValue()
        if no_data_value:
            no_data_value = int(no_data_value)
        else:
            no_data_value = 0
        no_data_values.append(no_data_value)
    del geotiff_file

    return no_data_values


@check_arguments
def get_geotiff_data_types(geotiff_filepath):
    """
    Returns the type of the data in the array for the passed geotiff filepath and raster band.
    Raster band should be an integer from 1 to N where N represents the number of raster bands.
    """

    if not isinstance(geotiff_filepath, str):
        raise AttributeError(f'Passed filepath was: "{geotiff_filepath}" which is not a string.')
    if not os.path.exists(geotiff_filepath):
        raise FileNotFoundError(f'Passed filepath was: "{geotiff_filepath}" which does not exist.')

    data_types = []
    geotiff_file = gdal.Open(geotiff_filepath)
    num_bands = geotiff_file.RasterCount
    for band_number in range(num_bands):
        band = geotiff_file.GetRasterBand(band_number + 1)
        data_type = band.DataType
        data_types.append(data_type)
    del geotiff_file

    return data_types


@check_arguments
def get_geotiff_min_max_values(geotiff_filepath,
                               chunk_size=10000,
                               replace_no_data_value_with=np.nan,
                               replace_nan_with=None,
                               verbose=False):
    """
    The parameter "geotiff_filepath" indicates the path to a geotiff file from which the minimum and maximum value will
    be calculated.
    The parameter "chunk_size" indicates the width and height of the chunks that should be read from one geotiff and
    written to the other.
    If "replace_no_data_value_with" is equal to a number, the value marked as the "no data" value by the geotiff
    file will be replaced with the passed value.
    If "replace_nan_with" is not None, any "nan" values in the read array will be replaced with the passed integer.
    The "verbose" parameter indicates if text will be printed indicating the progress of iterating through the process
    of reading and writing the chunks.

    Note that the function "generate_statistics" returns a tuple containing multiple statistics for the geotiff
    including the minimum and maximum value, and probably does it faster than this function.
    The difference is that function will always ignore nans whereas this function lets you replace them with a value
    for the purposes of calculating the minimum and maximum.
    """

    generator = _iterate_geotiff(geotiff_filepath, chunk_size=chunk_size, verbose=verbose)
    current_min = np.nan
    current_max = np.nan

    for x, y, x_chunk_size, y_chunk_size in generator:
        tiled_data = read_from_geotiff(geotiff_filepath,
                                       coordinates=(x, y),
                                       size=(x_chunk_size, y_chunk_size),
                                       convert_to_pytorch_tensor_format=False,
                                       replace_no_data_value_with=replace_no_data_value_with,
                                       replace_nan_with=replace_nan_with)
        if not np.isnan(tiled_data).all():
            array_min = np.nanmin(tiled_data)
            array_max = np.nanmax(tiled_data)
            if np.isnan(current_min) or array_min < current_min:
                current_min = array_min
            if np.isnan(current_max) or array_max > current_max:
                current_max = array_max
        if verbose:
            print(f'Current minimum is: {current_min}.\n'
                  f'Current maximum is: {current_max}.')

    return float(current_min), float(current_max)


@check_arguments
def get_geotiff_value_counts(geotiff_filepath,
                             chunk_size=10000,
                             replace_no_data_value_with=None,
                             verbose=False):
    """
    The parameter "geotiff_filepath" indicates the path to a geotiff file from which the value counts will be
    calculated.
    The parameter "chunk_size" indicates the width and height of the chunks that should be read from one geotiff and
    written to the other.
    If "replace_no_data_value_with" is equal to a number, the value marked as the "no data" value by the geotiff
    file will be replaced with the passed value.
    The "verbose" parameter indicates if text will be printed indicating the progress of iterating through the process
    of reading and writing the chunks.
    Returns a dictionary where each key indicates a class that appears in the geotiff file and each value represents
    the count of times that class appears.
    """

    generator = _iterate_geotiff(geotiff_filepath, chunk_size=chunk_size, verbose=verbose)
    value_counts = defaultdict(lambda: 0)

    for x, y, x_chunk_size, y_chunk_size in generator:
        tiled_data = read_from_geotiff(geotiff_filepath,
                                       coordinates=(x, y),
                                       size=(x_chunk_size, y_chunk_size),
                                       convert_to_pytorch_tensor_format=False,
                                       replace_no_data_value_with=replace_no_data_value_with)
        int_data = tiled_data.astype(int)
        values, counts = np.unique(int_data, return_counts=True)
        for value, count in zip(values, counts):
            value_counts[value] += count
        if verbose:
            print(f'Current counts are: {dict(value_counts)}.')

    return dict(value_counts)


@check_arguments
def get_geotiff_resolution(geotiff_filepath):
    """
    Returns a tuple indicating the (xres, yres) of the geotiff at the passed filepath.
    """

    geotiff_file = gdal.Open(geotiff_filepath)
    ulx, xres, xskew, uly, yskew, yres = geotiff_file.GetGeoTransform()
    del geotiff_file

    return xres, yres


@check_arguments
def get_geotiff_transform(geotiff_filepath):
    """Returns the coordinate transform for the passed geotiff file."""

    geotiff_file = gdal.Open(geotiff_filepath)
    geo_transform = geotiff_file.GetGeoTransform()
    del geotiff_file

    return geo_transform


@check_arguments
def get_geotiff_projection(geotiff_filepath):
    """Returns the projection for the passed geotiff file."""

    geotiff_file = gdal.Open(geotiff_filepath)
    projection = geotiff_file.GetProjection()
    del geotiff_file

    return projection


@check_arguments
def get_geotiff_metadata(geotiff_filepath):
    """Returns the metadata for the passed geotiff file."""

    geotiff_file = gdal.Open(geotiff_filepath)
    metadata = geotiff_file.GetMetadata()
    del geotiff_file

    return metadata


@check_arguments
def get_geotiff_info(geotiff_filepath):
    """Returns the info for the passed geotiff file."""

    return gdal.Info(geotiff_filepath, deserialize=True)


@check_arguments
def get_lat_lon_from_grid_coordinates(geotiff_filepath,
                                      grid_coordinates):
    """
    Returns the longitude and latitude corresponding to the passed grid coordinates (as an (x, y) tuple)
    for the passed geotiff file.
    """

    # setup the source projection
    source_file = gdal.Open(geotiff_filepath)
    source = osr.SpatialReference()
    source.ImportFromWkt(source_file.GetProjection())

    # setup the target projection
    target = osr.SpatialReference()
    target.ImportFromEPSG(4326)

    # perform the transform
    transform = osr.CoordinateTransformation(source, target)
    lat, lon, alt = transform.TransformPoint(*grid_coordinates)

    del source_file

    return lat, lon


@check_arguments
def get_grid_coordinates_from_lat_lon(geotiff_filepath,
                                      lat_lon):
    """
    Returns the grid coordinates corresponding to the passed latitude and longitude (as a (lat, lon) tuple)
    for the passed geotiff file.
    """

    # set up the source projection
    source = osr.SpatialReference()
    source.ImportFromEPSG(4326)

    # set up the target projection
    target_file = gdal.Open(geotiff_filepath)
    target = osr.SpatialReference()
    target.ImportFromWkt(target_file.GetProjection())

    # perform the transform
    transform = osr.CoordinateTransformation(source, target)
    value = transform.TransformPoint(*lat_lon)

    del target_file

    return round(value[0]), round(value[1])


@check_arguments
def get_array_coordinates_from_grid_coordinates(geotiff_filepath,
                                                grid_coordinates):
    """
    Returns the array coordinates corresponding to the passed grid coordinates for the passed geotiff file.
    """

    (min_x, min_y), (_, _) = get_geotiff_grid_edge_locations(geotiff_filepath)
    xres, yres = get_geotiff_resolution(geotiff_filepath)
    array_coordinates = (round((grid_coordinates[0] - min_x) / xres), round((grid_coordinates[1] - min_y) / yres))

    return array_coordinates


@check_arguments
def get_grid_coordinates_from_array_coordinates(geotiff_filepath,
                                                array_coordinates):
    """
    Returns the grid coordinates corresponding to the passed array coordinates for the passed geotiff file.
    """

    (min_x, min_y), (_, _) = get_geotiff_grid_edge_locations(geotiff_filepath)
    xres, yres = get_geotiff_resolution(geotiff_filepath)
    grid_coordinates = (min_x + round(array_coordinates[0] * xres), round(min_y + array_coordinates[1] * yres))

    return grid_coordinates


@check_arguments
def generate_pyramids(geotiff_filepath):
    """Generates the pyramids for the passed geotiff file."""

    Image = gdal.Open(geotiff_filepath, 1)
    Image.BuildOverviews("AVERAGE", [2, 4, 8, 16, 32, 64])


@check_arguments
def generate_statistics(geotiff_filepath):
    """Generates the statistics for the passed geotiff file."""

    ds = gdal.Open(geotiff_filepath)
    stats = ds.GetRasterBand(1).GetStatistics(0, 1)
    return stats


@check_arguments
def make_arc_gis_ready(geotiff_filepath):
    """Generates the pyramids and statistics for geotiff file, which allows it to be displayed properly in ArcGIS."""

    generate_pyramids(geotiff_filepath)
    generate_statistics(geotiff_filepath)


@check_arguments
def _iterate_geotiff(geotiff_filepath,
                     top_left_coordinates=None,
                     bottom_right_coordinates=None,
                     chunk_size=10000,
                     verbose=False):
    """
    The parameter "geotiff_filepath" indicates the path to a geotiff file to which data will be written in chunks.
    The parameter "data_array" is an optional numpy array or pytorch tensor that will be written to the file.
    The coordinates parameters are (x, y) tuples which indicates where to place the numpy array relative to the
    reference file.
    If "top_left_coordinates" is specified, only coordinates below and to the right of the passed coordinates will be
    iterated through. Otherwise the top left coordinates will set to (0, 0) i.e. the top left of the geotiff's data
    array.
    If "bottom_right_coordinates" are specified, coordinates below and to the right of the passed coordinates will not
    be iterated through. Otherwise, the bottom right coordinates will the be bottom right of the geotiff's data array.
    The parameter "chunk_size" indicates the width and height of the chunks that should be read from one geotiff and
    written to the other.
    The "verbose" parameter indicates if text will be printed indicating the progress of iterating through the process
    of reading and writing the chunks.
    """

    dims = get_geotiff_size(geotiff_filepath)
    if top_left_coordinates is None:
        top_left_coordinates = (0, 0)
    if bottom_right_coordinates is None:
        bottom_right_coordinates = dims[0:2]
    num_writes = len(range(top_left_coordinates[0], bottom_right_coordinates[0], chunk_size)) * \
                 len(range(top_left_coordinates[1], bottom_right_coordinates[1], chunk_size))

    i = 0
    with alive_bar(num_writes, dual_line=True, title='Loading chunks:') as progressbar:
        progressbar.text = 'Since the geotiff file is large, it will be loaded in separate chunks (see progress above).'
        for x in range(top_left_coordinates[0], bottom_right_coordinates[0], chunk_size):
            for y in range(top_left_coordinates[1], bottom_right_coordinates[1], chunk_size):
                x_chunk_size = bottom_right_coordinates[0] - x \
                    if x + chunk_size >= bottom_right_coordinates[0] else chunk_size
                y_chunk_size = bottom_right_coordinates[1] - y \
                    if y + chunk_size >= bottom_right_coordinates[1] else chunk_size
                if verbose:
                    print(f'{round(i / num_writes * 100)} percent complete.')
                    print(f'Generator at coordinates "{(x, y)}" and chunk size "{(x_chunk_size, y_chunk_size)}" '
                          f'in geotiff at "{geotiff_filepath}".')
                progressbar()
                yield x, y, x_chunk_size, y_chunk_size
                i += 1
