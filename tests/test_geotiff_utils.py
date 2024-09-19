"""
This file contains code for testing the geotiff utilities.
"""


import os
import random

import numpy as np

from utils.data_cache import DataCache
from utils.geotiff_utils import read_from_geotiff, create_geotiff, write_chunks_to_geotiff, \
    write_to_geotiff_within_lat_lon, get_geotiff_no_data_values, get_geotiff_epsg, get_geotiff_size, \
    get_geotiff_grid_edge_locations, get_geotiff_data_types, get_geotiff_raster_band_count, \
    get_geotiff_lat_lon_edge_locations, get_geotiff_resolution, get_geotiff_value_counts, get_geotiff_min_max_values, \
    get_grid_coordinates_from_lat_lon, get_grid_coordinates_from_array_coordinates, get_lat_lon_from_grid_coordinates, \
    get_array_coordinates_from_grid_coordinates, read_geotiff_within_lat_lon, get_geotiff_transform, \
    get_geotiff_projection, get_geotiff_metadata, get_geotiff_info
from utils.general_utils import approx_equal_tuple
from utils.numpy_utils import approximately_same, normalize_array


def run_geotiff_utils_tests():
    temp_geotiff_path = os.path.join(os.getcwd(), 'tests', 'test_dataset', 'temp.tif')
    if os.path.exists(temp_geotiff_path):
        os.remove(temp_geotiff_path)

    test_files = [os.path.join(os.getcwd(), 'tests', 'test_dataset', path) for path in
                  ['test.tif', 'test2.tif', 'multi_band_test.tif']]

    print("\n")

    # test creation of geotiffs using a reference geotiff file, from custom parameters, and a combination of both
    print("Testing the creation of geotiff files.")
    test_geo_transform = (429066.0566999996, 1.0, 0.0, 202889.0457000006, 0.0, -1.0)
    test_projection = "PROJCS[\"NAD83 / Vermont\",GEOGCS[\"NAD83\",DATUM[\"North_American_Datum_1983\"," \
                      "SPHEROID[\"GRS 1980\",6378137,298.257222101,AUTHORITY[\"EPSG\",\"7019\"]]," \
                      "AUTHORITY[\"EPSG\",\"6269\"]],PRIMEM[\"Greenwich\",0,AUTHORITY[\"EPSG\",\"8901\"]]," \
                      "UNIT[\"degree\",0.0174532925199433,AUTHORITY[\"EPSG\",\"9122\"]]," \
                      "AUTHORITY[\"EPSG\",\"4269\"]],PROJECTION[\"Transverse_Mercator\"]," \
                      "PARAMETER[\"latitude_of_origin\",42.5],PARAMETER[\"central_meridian\",-72.5]," \
                      "PARAMETER[\"scale_factor\",0.999964286],PARAMETER[\"false_easting\",500000]," \
                      "PARAMETER[\"false_northing\",0],UNIT[\"metre\",1,AUTHORITY[\"EPSG\",\"9001\"]]," \
                      "AXIS[\"Easting\",EAST],AXIS[\"Northing\",NORTH],AUTHORITY[\"EPSG\",\"32145\"]]"
    reference_geotiff_filepaths = [*test_files, None]
    widths = [None, 100, None, 10000]
    heights = [None, None, 500, 5000]
    num_bands = [None, 1, None, 3]
    no_data_values = [None, [-128], None, [0, 0, 0]]
    geo_transforms = [None, test_geo_transform, None, test_geo_transform]
    projections = [None, None, test_projection, test_projection]
    for reference_geotiff_filepath, width, height, bands, no_data_value_list, geo_transform, projection in \
            zip(reference_geotiff_filepaths, widths, heights, num_bands, no_data_values, geo_transforms, projections):
        create_geotiff(temp_geotiff_path,
                       reference_geotiff_filepath,
                       width=width,
                       height=height,
                       num_bands=bands,
                       no_data_values=no_data_value_list,
                       geo_transform=geo_transform,
                       projection=projection)
        if reference_geotiff_filepath:
            ref_width, ref_height, ref_num_bands = get_geotiff_size(reference_geotiff_filepath)
        if width is None and reference_geotiff_filepath:
            assert get_geotiff_size(temp_geotiff_path)[0] == ref_width
        else:
            assert get_geotiff_size(temp_geotiff_path)[0] == width
        if height is None and reference_geotiff_filepath:
            assert get_geotiff_size(temp_geotiff_path)[1] == ref_height
        else:
            assert get_geotiff_size(temp_geotiff_path)[1] == height
        if bands is None and reference_geotiff_filepath:
            assert get_geotiff_size(temp_geotiff_path)[2] == ref_num_bands
        else:
            assert get_geotiff_size(temp_geotiff_path)[2] == bands
        if no_data_value_list is None and reference_geotiff_filepath:
            assert get_geotiff_no_data_values(temp_geotiff_path) == get_geotiff_no_data_values(reference_geotiff_filepath)
        else:
            assert get_geotiff_no_data_values(temp_geotiff_path) == no_data_value_list
        if geo_transform is None and reference_geotiff_filepath:
            assert get_geotiff_transform(temp_geotiff_path) == get_geotiff_transform(reference_geotiff_filepath)
        else:
            assert get_geotiff_transform(temp_geotiff_path) == geo_transform
        if projection is None and reference_geotiff_filepath:
            # we only test the last 400 characters of the sting since the other characters sometimes change
            assert get_geotiff_projection(temp_geotiff_path)[-400:] == \
                   get_geotiff_projection(reference_geotiff_filepath)[-400:]
            assert get_geotiff_epsg(temp_geotiff_path) == get_geotiff_epsg(reference_geotiff_filepath)
        else:
            assert get_geotiff_projection(temp_geotiff_path) == projection
            assert get_geotiff_epsg(temp_geotiff_path) == int(projection[-8:-3])
        os.remove(temp_geotiff_path)
    print("Success.\n")

    # test that arrays read from geotiff have correct dimensions, no data values, can be rescaled properly, ect
    print("Testing reading from geotiff, replacing no data values and nans, rescaling, and normalizing.")
    x, y, width, height = 10000, 5000, 1000, 1000
    replace_value = -99
    test_args_list = [{'replace_no_data_value_with': replace_value},
                      {'replace_no_data_value_with': replace_value, 'replace_nan_with': replace_value},
                      {'rescale': 2},
                      {'normalize': True},
                      {'replace_no_data_value_with': replace_value, 'replace_nan_with': replace_value, 'rescale': 2}]
    for read_geotiff_path in test_files:
        basic_read_data = read_from_geotiff(read_geotiff_path,
                                            coordinates=(x, y),
                                            size=(width, height),
                                            convert_to_pytorch_tensor_format=False)
        no_data_value = get_geotiff_no_data_values(read_geotiff_path)
        for kwargs in test_args_list:
            read_data = read_from_geotiff(read_geotiff_path,
                                          coordinates=(x, y),
                                          size=(width, height),
                                          convert_to_pytorch_tensor_format=False,
                                          **kwargs)
            compare_data = np.copy(basic_read_data)
            if 'replace_no_data_value_with' in kwargs:
                compare_data[compare_data == no_data_value] = replace_value
            if 'replace_nan_with' in kwargs:
                compare_data[compare_data == np.nan] = replace_value
            if 'normalize' in kwargs:
                compare_data = normalize_array(compare_data)
            expected_width = width * (kwargs['rescale'] if 'rescale' in kwargs else 1)
            expected_height = height * (kwargs['rescale'] if 'rescale' in kwargs else 1)
            if 'replace_no_data_value_with' in kwargs and not replace_value == no_data_value:
                assert no_data_value not in read_data
            if 'replace_nan_with' in kwargs and not replace_value == np.nan:
                assert np.nan not in read_data
            if 'normalize' in kwargs:
                unique_values = list(np.unique(read_data))
                min_value = unique_values[0]
                max_value = unique_values[-1]
                if min_value == max_value:
                    assert np.amin(read_data) == 0
                    assert np.amax(read_data) == 0
                else:
                    assert np.amin(read_data) == 0
                    assert np.amax(read_data) == 1
            assert read_data.shape[:2] == (expected_height, expected_width)
            assert approximately_same(compare_data, read_data, auto_rescale=True)
    print("Success.\n")

    # test reading and writing a geotiff in single chunk, and test automatic rescaling when writing
    print("Testing reading from and writing to geotiff files in a single chunk with different rescale values.")
    x, y, width, height = 10000, 5000, 1000, 1000
    rescale_values = [1, 2, 0.5]
    for read_geotiff_path in test_files:
        for rescale_value in rescale_values:
            read_data = read_from_geotiff(read_geotiff_path,
                                          coordinates=(x, y),
                                          size=(width, height),
                                          convert_to_pytorch_tensor_format=False,
                                          rescale=rescale_value)
            create_geotiff(temp_geotiff_path,
                           read_geotiff_path,
                           read_data,
                           top_left_coordinates=(x, y),
                           bottom_right_coordinates=(x+width, y+height))
            read_written_data = read_from_geotiff(temp_geotiff_path,
                                                  coordinates=(x, y),
                                                  size=(width, height),
                                                  convert_to_pytorch_tensor_format=False)
            if rescale_value == 1:
                assert np.array_equal(read_data, read_written_data)
            else:
                assert approximately_same(read_data, read_written_data, auto_rescale=True)
            assert get_geotiff_epsg(read_geotiff_path) == get_geotiff_epsg(temp_geotiff_path) == 32145
            assert get_geotiff_size(read_geotiff_path) == get_geotiff_size(temp_geotiff_path)
            assert get_geotiff_grid_edge_locations(read_geotiff_path) == \
                   get_geotiff_grid_edge_locations(temp_geotiff_path)
            os.remove(temp_geotiff_path)
    print("Success.\n")

    # test reading and writing geotiffs in chunks
    print("Testing reading from and writing to geotiff files in chunks.")
    test_size = 10
    num_tests = 5
    for read_geotiff_path in test_files:
        create_geotiff(temp_geotiff_path, read_geotiff_path)
        write_chunks_to_geotiff(temp_geotiff_path, read_geotiff_path, chunk_size=10000, verbose=False)
        width, height, _ = get_geotiff_size(read_geotiff_path)
        min_x, min_y, max_x, max_y = 0, 0, width, height
        for i in range(num_tests):
            test_x = random.randint(min_x, max_x - test_size)
            test_y = random.randint(min_y, max_y - test_size)
            read_data = read_from_geotiff(read_geotiff_path, coordinates=(test_x, test_y), size=(test_size, test_size))
            read_written_data = read_from_geotiff(temp_geotiff_path,
                                                  coordinates=(test_x, test_y),
                                                  size=(test_size, test_size))
            assert np.array_equal(read_data, read_written_data)
        assert get_geotiff_epsg(read_geotiff_path) == get_geotiff_epsg(temp_geotiff_path) == 32145
        assert get_geotiff_size(read_geotiff_path) == get_geotiff_size(temp_geotiff_path)
        assert get_geotiff_grid_edge_locations(read_geotiff_path) == get_geotiff_grid_edge_locations(temp_geotiff_path)
        os.remove(temp_geotiff_path)
    print("Success.\n")

    # test the data types, band counts, epsgs, no data values, array sizes, grid edges, lat lon edges, resolutions,
    # value counts, and min and maxes in the test files matches the newly created file
    # for some of these the geotiff utils isn't designed to create an identical value in the created file, so for those
    # merely test that the test file values matches the value cached in the storage to make sure it is read correctly
    print("Testing data types, band counts, espgs, no data values, array sizes, grid edges, lat lon edges, "
          "resolutions, value counts, and min and maxes in created geotiff files.")
    base_path = os.path.join(os.getcwd(), 'tests', 'test_dataset')
    test_data_types = DataCache(os.path.join(base_path, 'test_data_types.dat'), create=False)
    test_band_counts = DataCache(os.path.join(base_path, 'test_band_counts.dat'), create=False)
    test_epsgs = DataCache(os.path.join(base_path, 'test_espgs.dat'), create=False)
    test_no_data_values = DataCache(os.path.join(base_path, 'test_no_data_values.dat'), create=False)
    test_resolutions = DataCache(os.path.join(base_path, 'test_resolutions.dat'), create=False)
    test_counts = DataCache(os.path.join(base_path, 'test_data_counts.dat'), create=False)
    test_min_maxes = DataCache(os.path.join(base_path, 'test_min_maxes.dat'), create=False)
    test_metadata_entries = DataCache(os.path.join(base_path, 'test_metadata.dat'), create=False)
    test_info_entries = DataCache(os.path.join(base_path, 'test_info.dat'), create=False)
    for test_filepath in test_files:
        create_geotiff(temp_geotiff_path, test_filepath)
        # assert get_geotiff_data_type(test_filepath) == get_geotiff_data_type(temp_geotiff_path) == data_type
        assert get_geotiff_data_types(test_filepath) == \
               test_data_types.load_from_storage(os.path.basename(test_filepath))
        assert get_geotiff_data_types(temp_geotiff_path) == [6] * get_geotiff_raster_band_count(test_filepath)
        assert get_geotiff_raster_band_count(test_filepath) == \
               get_geotiff_raster_band_count(temp_geotiff_path) == \
               test_band_counts.load_from_storage(os.path.basename(test_filepath))
        assert get_geotiff_epsg(test_filepath) == \
               get_geotiff_epsg(temp_geotiff_path) == \
               test_epsgs.load_from_storage(os.path.basename(test_filepath))
        assert get_geotiff_no_data_values(test_filepath) == \
               get_geotiff_no_data_values(temp_geotiff_path) == \
               test_no_data_values.load_from_storage(os.path.basename(test_filepath))
        assert get_geotiff_size(test_filepath) == \
               get_geotiff_size(temp_geotiff_path)
        assert get_geotiff_grid_edge_locations(test_filepath) == \
               get_geotiff_grid_edge_locations(temp_geotiff_path)
        assert get_geotiff_lat_lon_edge_locations(test_filepath) == \
               get_geotiff_lat_lon_edge_locations(temp_geotiff_path)
        assert get_geotiff_resolution(test_filepath) == \
               get_geotiff_resolution(temp_geotiff_path) == \
               test_resolutions.load_from_storage(os.path.basename(test_filepath))
        assert get_geotiff_value_counts(test_filepath) == test_counts.load_from_storage(os.path.basename(test_filepath))
        assert approx_equal_tuple(get_geotiff_min_max_values(test_filepath),
                                  test_min_maxes.load_from_storage(os.path.basename(test_filepath)))
        assert get_geotiff_metadata(test_filepath) == \
               test_metadata_entries.load_from_storage(os.path.basename(test_filepath))
        test_info = get_geotiff_info(test_filepath)
        cached_info = test_info_entries.load_from_storage(os.path.basename(test_filepath))
        assert test_info[test_info.index("Size is "):] == cached_info[cached_info.index("Size is "):]
        os.remove(temp_geotiff_path)
    print("Success.\n")

    # test the coordinate transforms in the test files
    print("Testing coordinate transforms in created geotiff files.")
    for test_filepath in test_files:

        # find the height, width, and edge coordinates
        width, height, _ = get_geotiff_size(test_filepath)
        edge_top_left_grid, edge_bottom_right_grid = get_geotiff_grid_edge_locations(test_filepath)
        edge_top_left_lat_lon, edge_bottom_right_lat_lon = get_geotiff_lat_lon_edge_locations(test_filepath)

        # transform from the array coordinates to grid coordinates to lat lon coordinates
        start_top_left_array = (0, 0)
        start_bottom_right_array = (width, height)
        start_top_left_grid = get_grid_coordinates_from_array_coordinates(test_filepath, start_top_left_array)
        start_bottom_right_grid = get_grid_coordinates_from_array_coordinates(test_filepath, start_bottom_right_array)
        top_left_lat_lon = get_lat_lon_from_grid_coordinates(test_filepath, start_top_left_grid)
        bottom_right_lat_lon = get_lat_lon_from_grid_coordinates(test_filepath, start_bottom_right_grid)

        # transform from lat lon coordinates to grid coordinates to array coordinates
        end_top_left_grid = get_grid_coordinates_from_lat_lon(test_filepath, top_left_lat_lon)
        end_bottom_right_grid = get_grid_coordinates_from_lat_lon(test_filepath, bottom_right_lat_lon)
        end_top_left_array = get_array_coordinates_from_grid_coordinates(test_filepath, end_top_left_grid)
        end_bottom_right_array = get_array_coordinates_from_grid_coordinates(test_filepath, end_bottom_right_grid)

        assert start_top_left_array == end_top_left_array
        assert start_bottom_right_array == end_bottom_right_array
        assert start_top_left_grid == end_top_left_grid == edge_top_left_grid
        assert start_bottom_right_grid == end_bottom_right_grid == edge_bottom_right_grid
        assert top_left_lat_lon == edge_top_left_lat_lon
        assert bottom_right_lat_lon == edge_bottom_right_lat_lon
    print("Success.\n")

    # test reading and writing within GPS coordinates of files with different resolutions
    print("Testing reading from and writing to within specific GPS coordinates in geotiff files.")
    test_coordinates_start_end = [(0.1, 0.2), (0.1, 0.2), (0.4, 0.5)]
    test_array_sizes = [(100, 100, 1), (100, 100, 1), (100, 100, 4)]
    for read_geotiff_path, start_end, array_size in zip(test_files, test_coordinates_start_end, test_array_sizes):
        test_array = np.random.rand(*array_size).astype('float32')
        start_lat_lon, end_lat_lon = get_geotiff_lat_lon_edge_locations(read_geotiff_path)
        start_lat = start_lat_lon[0] - (start_lat_lon[0] - end_lat_lon[0]) * start_end[0]
        start_lon = start_lat_lon[1] + (end_lat_lon[1] - start_lat_lon[1]) * start_end[0]
        end_lat = start_lat_lon[0] - (start_lat_lon[0] - end_lat_lon[0]) * start_end[1]
        end_lon = start_lat_lon[1] + (end_lat_lon[1] - start_lat_lon[1]) * start_end[1]
        create_geotiff(temp_geotiff_path, read_geotiff_path)
        write_to_geotiff_within_lat_lon(temp_geotiff_path, test_array, (start_lat, start_lon), (end_lat, end_lon))
        read_array = read_geotiff_within_lat_lon(temp_geotiff_path,
                                                 (start_lat, start_lon),
                                                 (end_lat, end_lon),
                                                 convert_to_pytorch_tensor_format=False,
                                                 rescale=array_size[:-1])
        assert approximately_same(test_array, read_array)
        os.remove(temp_geotiff_path)
    print("Success.\n")


if __name__ == "__main__":
    run_geotiff_utils_tests()
