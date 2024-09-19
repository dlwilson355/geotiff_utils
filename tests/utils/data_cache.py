"""This file contains code for saving and loading data to and from a cache file."""


import os.path
import pickle

from utils.file_utils import get_min_max_cache_filepath, get_cell_assignment_cache_filepath, \
    get_class_ratio_cache_filepath, get_test_min_maxes_filepath, get_test_counts_filepath, \
    get_test_data_types_filepath, get_test_band_counts_filepath, get_test_epsgs_filepath, \
    get_test_no_data_values_filepath, get_test_resolutions_filepath, get_test_metadata_filepath, get_test_info_filepath


def get_min_max_cache(create=True):
    """Returns a data cache object in which the minimum and maximum values for each image in the dataset is stored."""

    return DataCache(get_min_max_cache_filepath(), create=create)


def get_cell_assignment_cache(create=True):
    """Returns a data cache object in which cell assignments used by the dataset are stored."""

    return DataCache(get_cell_assignment_cache_filepath(), create=create)


def get_class_ratio_cache(create=True):
    """Returns a data cache object in which the class frequencies are stored."""

    return DataCache(get_class_ratio_cache_filepath(), create=create)


def get_test_min_maxes(create=False):
    """
    Returns a data cache object in which the minimum and maximum values for each image in the test dataset is stored.
    """

    return DataCache(get_test_min_maxes_filepath(), create=create)


def get_test_counts(create=False):
    """
    Returns a data cache object in which the number of counts with which certain values appear in the test dataset
    is stored.
    """

    return DataCache(get_test_counts_filepath(), create=create)


def get_test_data_types(create=False):
    """
    Returns a data cache object in which the data types used by the geotiff files in the test dataset are stored.
    """

    return DataCache(get_test_data_types_filepath(), create=create)


def get_test_band_counts(create=False):
    """
    Returns a data cache object in which the raster band counts for the geotiff files in the test dataset are stored.
    """

    return DataCache(get_test_band_counts_filepath(), create=create)


def get_test_epsgs(create=False):
    """Returns a data cache object containing the ESPG values for the geotiff files in the test dataset."""

    return DataCache(get_test_epsgs_filepath(), create=create)


def get_test_no_data_values(create=False):
    """Returns a data cache object containing the no data values for the geotiff files in the test dataset."""

    return DataCache(get_test_no_data_values_filepath(), create=create)


def get_test_resolutions(create=False):
    """Returns a data cache object containing the resolutions for the geotiff files in the test dataset."""

    return DataCache(get_test_resolutions_filepath(), create=create)


def get_test_metadata(create=False):
    """Returns a data cache object containing the metadata for the geotiff files in the test dataset."""

    return DataCache(get_test_metadata_filepath(), create=create)


def get_test_info(create=False):
    """Returns a data cache object containing the gdal info entries the geotiff files in the test dataset."""

    return DataCache(get_test_info_filepath(), create=create)


class DataCache:
    def __init__(self, filepath, create=True):
        self.filepath = filepath  # the filepath to which the cache file will be saved
        if create and not os.path.exists(filepath):
            print(f'Cache file at: "{filepath}" does not exist.\n'
                  f'An empty cache file will be created at this path.')
            self.save()

    def save(self, obj={}):
        """Saves the passed object to the cache file."""

        with open(self.filepath, 'wb') as f:
            pickle.dump(obj, f)

    def load(self):
        """Loads and returns the passed object to the cache file."""

        with open(self.filepath, 'rb') as f:
            obj = pickle.load(f)

        return obj

    def save_to_storage(self, key, obj):
        """
        Saves the passed object to the cache, after which it can be accessed using the passed key.
        """

        cache = self.load()
        cache[key] = obj
        self.save(cache)

    def load_from_storage(self, key):
        """
        Returns the object stored in the cache accessed from the passed key.
        If the key is not found "None" is returned.
        """

        cache = self.load()

        if key in cache:
            return cache[key]
        else:
            return None

    def exists(self):
        """Returns a boolean indicating if a cache file currently exists."""

        return os.path.exists(self.filepath)

    def get_path(self):
        """Returns the path at which the cache is stored."""

        return self.filepath
