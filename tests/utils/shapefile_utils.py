"""
This file contains code for interacting with shape files.
"""


import inspect
import os

import numpy as np
from alive_progress import alive_bar
import shapefile
from shapely.geometry.polygon import Polygon
from osgeo import osr

from utils.general_utils import check_required_and_keyword_args, get_lat_lon_diff_from_meters
from utils.geotiff_utils import write_to_geotiff, write_to_geotiff_within_lat_lon, get_grid_coordinates_from_lat_lon, \
    get_array_coordinates_from_grid_coordinates


def check_arguments(func):
    def wrapper(*args, **kwargs):

        check_required_and_keyword_args(func, args, kwargs)

        keywords = inspect.signature(func).parameters.keys()
        for keyword, arg in zip(keywords, args):
            kwargs[keyword] = arg

        check_keywords = ['shape_filepath',
                          'write_geotiff_path']
        for check_keyword in check_keywords:
            if check_keyword in kwargs:
                arg = kwargs[check_keyword]
                if not isinstance(arg, str):
                    raise AttributeError(f'Passed value for "{check_keyword}" was "{arg}" which is not a string.')
                if not os.path.exists(arg):
                    raise FileNotFoundError(
                        f'Passed value for "{check_keyword}" was "{arg}" which is a filepath that does not exist.')

        check_keywords = ['lat_lon_cells']
        for check_keyword in check_keywords:
            if check_keyword in kwargs:
                arg = kwargs[check_keyword]
                if not isinstance(arg, list):
                    raise AttributeError(f'Passed value for "{check_keyword}" must be a list.')
                valid = True
                for item in arg:
                    if valid:
                        if not isinstance(item, tuple) and len(item) == 2:
                            valid = False
                        for coord in item:
                            if not isinstance(coord, tuple) and len(coord) == 2:
                                valid = False
                            if valid:
                                if not all(isinstance(i, (int, float)) for i in coord):
                                    valid = False
                if not valid:
                    raise AttributeError(f'Passed value for "{check_keyword}" must be a list containing length two '
                                         f'tuples each of which contains two pairs of latitude longitude pairs.\n'
                                         f'For example: "[((lat, lon), (lat, lon)), ((lat, lon), (lat, lon))]".')

        check_keywords = ['grid_coordinates', 'meters_cell_size']
        for check_keyword in check_keywords:
            if check_keyword in kwargs:
                arg = kwargs[check_keyword]
                if arg is not None and not (isinstance(arg, tuple)
                                            and len(arg) == 2
                                            and all(isinstance(n, int) for n in arg)):
                    raise AttributeError(f'The passed value for "{check_keyword}" was "{arg}" which is invalid.\n'
                                         f'The value for "{check_keyword}" should "None" or '
                                         f'a tuple containing two integers.')

        check_keywords = ['lat_lon', 'lat_lon_cell_size']
        for check_keyword in check_keywords:
            if check_keyword in kwargs:
                arg = kwargs[check_keyword]
                if not (isinstance(arg, tuple) and len(arg) == 2 and all(isinstance(n, (int, float)) for n in arg)):
                    raise AttributeError(f'Passed value for "{check_keyword}" was "{arg}".\n'
                                         f'It must be a tuple of length two containing either ints or floats.')

        check_keywords = ['convert_to_lat_lon']
        for check_keyword in check_keywords:
            if check_keyword in kwargs:
                arg = kwargs[check_keyword]
                if not isinstance(arg, bool):
                    raise AttributeError(f'Passed value for "{check_keyword}" must be a boolean.')

        check_keywords = ['relative_cell_width']
        for check_keyword in check_keywords:
            if check_keyword in kwargs:
                arg = kwargs[check_keyword]
                if not isinstance(arg, float):
                    raise AttributeError(f'Value for "{check_keyword}" was "{arg}" which is not a float.')
                if not 0 < arg < 1:
                    raise AttributeError(f'Value for "{check_keyword}" was "{arg}" '
                                         f'which must be greater than 0 and less than 1.')

        check_keywords = ['grid_cell_size', 'meters_cell_size']
        for check_keyword in check_keywords:
            if check_keyword in kwargs:
                arg = kwargs[check_keyword]
                if not (isinstance(arg, tuple) and len(arg) == 2 and all(isinstance(n, int) for n in arg)):
                    raise AttributeError(f'Passed value for "{check_keyword}" was "{arg}".\n'
                                         f'It must be a tuple of length two containing ints.')
                if not all([v > 0 for v in arg]):
                    raise AttributeError(f'Passed list for "{check_keyword}" should only contain values greater '
                                         f'than 0.')

        check_keywords = ['shape_index']
        for check_keyword in check_keywords:
            if check_keyword in kwargs:
                arg = kwargs[check_keyword]
                if not isinstance(arg, int):
                    raise AttributeError(f'Value for "{check_keyword}" was "{arg}" which is not an integer.')
                if not arg >= 0:
                    raise AttributeError(f'Value for "{check_keyword}" was "{arg}" '
                                         f'which is not greater than or equal to 0.')

        return func(**kwargs)
    return wrapper


@check_arguments
def get_polygon_from_shapefile(shape_filepath, shape_index=0):
    sf = shapefile.Reader(shape_filepath)
    shape = sf.shape(shape_index)
    polygon = Polygon([p for p in shape.points])
    sf.close()

    return polygon


@check_arguments
def get_grid_cells_inside_shape(shape_filepath,
                                grid_cell_size=None,
                                lat_lon_cell_size=None,
                                meters_cell_size=None,
                                shape_index=0,
                                convert_to_lat_lon=True):
    """Returns a grid of cells within the shape at the passed index withing the shape file."""

    num_sizes_passed = 0
    for arg in [grid_cell_size, lat_lon_cell_size, meters_cell_size]:
        if arg:
            num_sizes_passed += 1
    if num_sizes_passed == 0:
        raise ValueError("You must pass the size for the cells either as a grid size, lat lon size, or meters size.")
    elif num_sizes_passed > 1:
        raise ValueError("You must pass the size for the cells either as a grid size, lat lon size, or meters size.\n"
                         "You can only choose one (you passed multiple).")

    # read the shape from the shapefile
    sf = shapefile.Reader(shape_filepath)
    shape = sf.shape(shape_index)
    sf.close()

    # convert the shape into a polygon and read a bounding box around the shape
    bbox = shape.bbox
    bbox_x1, bbox_y1, bbox_x2, bbox_y2 = [int(value) for value in bbox]
    polygon = Polygon([p for p in shape.points])

    # convert the meters cell size to the corresponding lat lon cell size if appropriate
    if meters_cell_size:
        lat_1, lon_1 = get_lat_lon_from_projection_grid_coordinates(shape_filepath, (bbox_x1, bbox_y1))
        lat_lon_cell_size = get_lat_lon_diff_from_meters((lat_1, lon_1), meters_cell_size[0])

    # convert the lat lon cell size to grid cell size if appropriate
    if lat_lon_cell_size:
        lat_1, lon_1 = get_lat_lon_from_projection_grid_coordinates(shape_filepath, (bbox_x1, bbox_y1))
        lat_2, lon_2 = lat_1 + lat_lon_cell_size[0], lon_1 + lat_lon_cell_size[1]
        grid_x1, grid_y1 = get_grid_coordinates_from_projection_lat_lon(shape_filepath, (lat_1, lon_1))
        grid_x2, grid_y2 = get_grid_coordinates_from_projection_lat_lon(shape_filepath, (lat_2, lon_2))
        x_diff = int(abs(grid_x1 - grid_x2))
        y_diff = int(abs(grid_y1 - grid_y2))
        grid_cell_size = x_diff, y_diff

    # iterate through all the cells within the bounding box and make a list of any cells within the polygon
    valid_cells = []
    x_cell_size, y_cell_size = grid_cell_size
    num_cells = len(range(bbox_x1, bbox_x2, x_cell_size)) * len(range(bbox_y1, bbox_y2, y_cell_size))
    with alive_bar(num_cells, dual_line=True, title='Calculating cells:') as progressbar:
        progressbar.text = f'Calculating cells within shapefile at: {shape_filepath} (see progress above).'
        for x in range(bbox_x1, bbox_x2, x_cell_size):
            for y in range(bbox_y1, bbox_y2, y_cell_size):
                cell = Polygon([(x, y),
                                (x + x_cell_size, y),
                                (x + x_cell_size, y - y_cell_size),
                                (x, y - y_cell_size)])
                if polygon.contains(cell):
                    valid_cells.append(((x, y), (x + x_cell_size, y - y_cell_size)))
                progressbar()

    # convert the cells to latitude and longitude coordinates if applicable
    if convert_to_lat_lon:
        valid_cells = [(get_lat_lon_from_projection_grid_coordinates(shape_filepath, c1),
                        get_lat_lon_from_projection_grid_coordinates(shape_filepath, c2)) for c1, c2 in valid_cells]

    return valid_cells


@check_arguments
def write_lat_lon_grid_cells_to_geotiff(write_geotiff_filepath,
                                        lat_lon_cells,
                                        relative_cell_width=0.1,
                                        color=(255, 255, 255)):
    cell_data_size = 100
    cell_width = int(cell_data_size * relative_cell_width)
    cell_data = np.empty((cell_data_size, cell_data_size, 3)).astype('float32')
    cell_data[:, :, :] = 0
    cell_data[cell_width:cell_data_size - cell_width, cell_width:cell_data_size-cell_width, :] = color
    for (lat1, lon1), (lat2, lon2) in lat_lon_cells:
        write_to_geotiff_within_lat_lon(write_geotiff_filepath, cell_data, (lat1, lon1), (lat2, lon2))


@check_arguments
def write_lat_lon_grid_cells_to_geotiff_manual(write_geotiff_filepath, lat_lon_cells, relative_cell_width=0.1):
    for (lat1, lon1), (lat2, lon2) in lat_lon_cells:
        gx1, gy1 = get_grid_coordinates_from_lat_lon(write_geotiff_filepath, (lat1, lon1))
        x1, y1 = get_array_coordinates_from_grid_coordinates(write_geotiff_filepath, (gx1, gy1))
        gx2, gy2 = get_grid_coordinates_from_lat_lon(write_geotiff_filepath, (lat2, lon2))
        x2, y2 = get_array_coordinates_from_grid_coordinates(write_geotiff_filepath, (gx2, gy2))

        rows = abs(x1 - x2)
        cols = abs(y1 - y2)
        data = np.ones((rows, cols, 1))
        cell_width = int((rows + cols) / 2 * relative_cell_width)
        data[cell_width:rows - cell_width, cell_width:cols - cell_width, :] = 0
        write_to_geotiff(write_geotiff_filepath, data, top_left_coordinates=(x1, y1))


@check_arguments
def get_lat_lon_from_projection_grid_coordinates(shape_filepath, grid_coordinates):

    # find the path to the .prj file corresponding to the shape file
    prj_filepath = os.path.join(os.path.splitext(shape_filepath)[0] + '.prj')

    # setup the source projection
    with open(prj_filepath, "r") as f:
        prj_text = f.read()
    source = osr.SpatialReference()
    source.ImportFromWkt(prj_text)

    # setup the target projection
    target = osr.SpatialReference()
    target.ImportFromEPSG(4326)

    # perform the transform
    transform = osr.CoordinateTransformation(source, target)
    lat, lon, alt = transform.TransformPoint(*grid_coordinates)

    del source

    return lat, lon


@check_arguments
def get_grid_coordinates_from_projection_lat_lon(shape_filepath, lat_lon):

    # find the path to the .prj file corresponding to the shape file
    prj_filepath = os.path.join(os.path.splitext(shape_filepath)[0] + '.prj')

    # set up the source projection
    source = osr.SpatialReference()
    source.ImportFromEPSG(4326)

    # set up the target projection
    with open(prj_filepath, "r") as f:
        prj_text = f.read()
    target = osr.SpatialReference()
    target.ImportFromWkt(prj_text)

    # perform the transform
    transform = osr.CoordinateTransformation(source, target)
    value = transform.TransformPoint(*lat_lon)

    return round(value[0]), round(value[1])
