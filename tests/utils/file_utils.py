"""
This file contains utilities for working with and manipulating files and filepaths.
"""


import glob
import os


# This is the main path pointing to the directory in which all the data is stored
# This is the only path that needs to be changed to run the code on another system
# BASE_DIRECTORY = r'E:\wetlands_data'

# These are the names of various subdirectories within the main directory
MAIN_DATASET_SUBDIR = 'main_dataset'
TEST_DATASET_SUBDIR = 'test_dataset'
OUTPUT_SUBDIR = 'output'
CHECKPOINT_SUBDIR = 'checkpoint'
CACHE_SUBDIR = 'cache'
DEBUG_IMAGES_SUBDIR = 'debug_images'  # processed images for debugging can be optionally saved here

# These are the name of various cache files used to store temporary data
MIN_MAX_CACHE_FILENAME = 'min_max_cache.dat'
CELL_ASSIGNMENT_CACHE_FILENAME = 'cell_assignments.dat'
CLASS_RATIO_CACHE_FILENAME = 'class_ratio_cache.dat'
TEST_MIN_MAXES_FILENAME = 'test_min_maxes.dat'
TEST_DATA_COUNTS_FILENAME = 'test_data_counts.dat'
TEST_DATA_TYPES_FILENAME = 'test_data_types.dat'
TEST_BAND_COUNTS_FILENAME = 'test_band_counts.dat'
TEST_ESPGS_FILENAME = 'test_espgs.dat'
TEST_NO_DATA_VALUES_FILENAME = 'test_no_data_values.dat'
TEST_RESOLUTIONS_FILENAME = 'test_resolutions.dat'
TEST_METADATA_FILENAME = 'test_metadata.dat'
TEST_INFO_FILENAME = 'test_info.dat'
TEMP_GEOTIFF_FILENAME = 'temp.tif'  # a temporary geotiff created when testing the geotiff utilities
TEMP_CHECKPOINT_FILENAME = 'temp.pth'  # a temporary checkpoint created when testing the pytorch utilities

# These are the filenames of the geotiff files in the dataset
INPUT_FILENAMES = [r'LeafOffImagery_OtterCreek.tif',
                   r'leafonimagery_ottercreek.tif',
                   r'CTI3m_OtterCreek.tif',
                   r'CTITexture1m_OtterCreek.tif',
                   r'nDSM1m_OtterCreek.tif',
                   r'TPI1m_OtterCreek.tif']
OUTPUT_FILENAMES = [r'Wetlands_OtterCreek_30June2022_AllClasses.tif',
                    r'Wetlands_OtterCreek_30June2022_GenericWetlandsOnly.tif',
                    r'Wetlands_OtterCreek_30June2022_GenericWetlandsWater.tif',
                    r'Wetlands_OtterCreek_30June2022_PrimaryClasses.tif']
SHAPE_FILENAME = r'SouthernOtterCreek_Boundary.shp'
OUTPUT_FILENAME_LOOKUP = {'all_classes': r'Wetlands_OtterCreek_30June2022_AllClasses.tif',
                          'generic_wetlands': r'Wetlands_OtterCreek_30June2022_GenericWetlandsOnly.tif',
                          'generic_water': r'Wetlands_OtterCreek_30June2022_GenericWetlandsWater.tif',
                          'primary_classes': r'Wetlands_OtterCreek_30June2022_PrimaryClasses.tif'}
TEST_FILENAMES = ['test.tif', 'test2.tif', 'multi_band_test.tif']
TEST_SHAPE_FILENAME = 'SouthernOtterCreek_Boundary.shp'


# def get_base_directory():
#     """Returns the path of the base directory."""
#
#     return BASE_DIRECTORY


def find_file(filename, directory):
    """
    Returns the full filepath to the passed filename if it exists in the passed directory.
    Otherwise raises an exception.
    """

    all_files = glob.glob(os.path.join(directory, '**'), recursive=True)
    for filepath in all_files:
        if os.path.basename(filepath) == filename:
            return filepath

    raise FileNotFoundError(f'Did not find file: "{filename}" in directory: "{directory}".')


def create_subdir_if_not_exists(directory, readme_string=None):
    """
    Creates the subdirectory for the passed directory if it doesn't exist.
    If readme_string is passed, a README.txt with the passed string will be placed in the directory if it doesn't exist.
    """

    if not os.path.exists(os.path.dirname(directory)):
        raise NotADirectoryError(f"Can not create: {directory}.\n"
                                 f"Did not find parent directory at: {os.path.dirname(directory)}.")

    if not os.path.exists(directory):
        os.mkdir(directory)

    readme_path = os.path.join(directory, 'README.txt')
    if readme_string and not os.path.exists(readme_path):
        with open(readme_path, 'w+') as f:
            f.write(readme_string)


def get_data_filepath(filename, create=True, search=True):
    """
    Searches the data directory for a file with the matching filename and returns the path if found.
    If create=True, the data directory and its readme will be created if it doesn't exist.
    If search=False, instead returns the filepath to the file in the data directory regardless of whether it exists.
    """

    parent_directory = os.path.join(BASE_DIRECTORY, MAIN_DATASET_SUBDIR)

    if create:
        readme_string = f'This directory contains the main dataset used to train, validation, and test the model.\n' \
                        f'The actual files used by the model are the ".tif" files, so those files must be present in ' \
                        f'this directory. Any other files may be useful for displaying the data in software such as ' \
                        f'ArcGIS, but are not required by the model.'
        create_subdir_if_not_exists(parent_directory, readme_string=readme_string)

    if search:
        filepath = find_file(filename, parent_directory)
    else:
        filepath = os.path.join(parent_directory, filename)

    return filepath


def get_test_data_filepath(filename, create=True, search=True):
    """
    Searches the test data directory for a file with the matching filename and returns the path if found.
    If search=False, instead returns the filepath to the file in the data directory regardless of whether it exists.
    """

    parent_directory = os.path.join(BASE_DIRECTORY, TEST_DATASET_SUBDIR)

    if create:
        readme_string = f'This directory contains a testing dataset used to run tests which verify the code is ' \
                        f'working properly.'
        create_subdir_if_not_exists(parent_directory, readme_string=readme_string)

    if search:
        filepath = find_file(filename, parent_directory)
    else:
        filepath = os.path.join(parent_directory, filename)

    return filepath


def get_test_data_filepaths():
    """Returns a list of full filepaths to each geotiff file in the test dataset."""

    return [get_test_data_filepath(path) for path in TEST_FILENAMES]


def get_output_filepath(filename, create=True):
    """
    Returns the full filepath to the passed filename in the output directory.
    If create is true it will create the parent directory if it doesn't exist.
    """

    if create:
        parent_directory = os.path.join(BASE_DIRECTORY, OUTPUT_SUBDIR)
        readme_string = f'This is the directory to which output geotiff files produced by the model is saved.'
        create_subdir_if_not_exists(parent_directory, readme_string=readme_string)

    return os.path.join(BASE_DIRECTORY, OUTPUT_SUBDIR, filename)


def get_checkpoint_filepath(filename, create=True):
    """
    Returns the full filepath to the passed filename in the model directory.
    If create is true it will create the parent directory if it doesn't exist.
    """

    if create:
        parent_directory = os.path.join(BASE_DIRECTORY, CHECKPOINT_SUBDIR)
        readme_string = f'This is the directory to which models are saved.'
        create_subdir_if_not_exists(parent_directory, readme_string=readme_string)

    return os.path.join(BASE_DIRECTORY, CHECKPOINT_SUBDIR, filename)


def get_cache_filepath(filename, create=True):
    """
    Returns the full filepath to the passed filename in the temporary directory.
    If create is true it will create the parent directory if it doesn't exist.
    """

    if create:
        parent_directory = os.path.join(BASE_DIRECTORY, CACHE_SUBDIR)
        readme_string = f'This directory contains temporary files generated automatically.\n' \
                        f'The file "{MIN_MAX_CACHE_FILENAME}" contains cached values for the minimum and maximum ' \
                        f'value in each geotiff, which is used to speed up the process of normalizing the returned ' \
                        f'data array when reading from geotiff files.\n' \
                        f'The directory "{DEBUG_IMAGES_SUBDIR}" contains sample inputs provided to the model. ' \
                        f'This can be useful for debugging.\n'
        create_subdir_if_not_exists(parent_directory, readme_string=readme_string)

    return os.path.join(BASE_DIRECTORY, CACHE_SUBDIR, filename)


def get_min_max_cache_filepath():
    """Returns the filepath to a cache file indicating minimum and maximum values for the geotiff files in the main."""

    return get_cache_filepath(MIN_MAX_CACHE_FILENAME)


def get_cell_assignment_cache_filepath():
    """
    Returns the filepath to a cache file indicating which cells are assigned to the training, validation, and
    testing data.
    """

    return get_cache_filepath(CELL_ASSIGNMENT_CACHE_FILENAME)


def get_class_ratio_cache_filepath():
    """
    Returns the filepath to a cache file indicating the ratios at which certain classes appear in the passed geotiff
    file.
    """

    return get_cache_filepath(CLASS_RATIO_CACHE_FILENAME)


def get_test_min_maxes_filepath():
    """
    Returns the filepath to a file indicating minimum and maximum values for the geotiff files in the test
    dataset.
    """

    return get_test_data_filepath(TEST_MIN_MAXES_FILENAME, search=False)


def get_test_counts_filepath():
    """
    Returns the filepath to a file containing a dictionary where each key is a filepath to a test dataset file.
    Each key accesses another dictionary in which each key represents a value that occurs in the geotiff file, and the
    corresponding value indicates how many times that key appears in the geotiff file.
    """

    return get_test_data_filepath(TEST_DATA_COUNTS_FILENAME, search=False)


def get_test_data_types_filepath():
    """Returns the filepath to the file specifying what data type each geotiff file in the test dataset holds."""

    return get_test_data_filepath(TEST_DATA_TYPES_FILENAME, search=False)


def get_test_band_counts_filepath():
    """
    Returns the filepath to the file specifying how many raster bands are present in each geotiff file in the
    test dataset.
    """

    return get_test_data_filepath(TEST_BAND_COUNTS_FILENAME, search=False)


def get_test_epsgs_filepath():
    """Returns the filepath to the file specifying the ESPG for each geotiff in the test dataset."""

    return get_test_data_filepath(TEST_ESPGS_FILENAME, search=False)


def get_test_no_data_values_filepath():
    """Returns the filepath to the file specifying which value represents "no data" in each file in the test dataset."""

    return get_test_data_filepath(TEST_NO_DATA_VALUES_FILENAME, search=False)


def get_test_resolutions_filepath():
    """Returns the filepath to the file specifying the resolution of each file in the test dataset."""

    return get_test_data_filepath(TEST_RESOLUTIONS_FILENAME, search=False)


def get_test_metadata_filepath():
    """Returns the filepath to the file specifying the metadata of each file in the test dataset."""

    return get_test_data_filepath(TEST_METADATA_FILENAME, search=False)


def get_test_info_filepath():
    """Returns the filepath to the file specifying the info for each file in the test dataset."""

    return get_test_data_filepath(TEST_INFO_FILENAME, search=False)


def get_temporary_geotiff_filepath():
    """
    Returns the filepath to which a temporary geotiff file can be saved.
    """

    return get_cache_filepath(TEMP_GEOTIFF_FILENAME)


def get_temporary_checkpoint_filepath():
    """
    Returns the filepath to which a temporary checkpoint file can be saved.
    """

    return get_cache_filepath(TEMP_CHECKPOINT_FILENAME)


def get_debug_images_save_directory(create=True):
    """
    Returns the path to the save directory for the preprocessed input images.
    If create is true it will create the parent directory if it doesn't exist.
    """

    image_cache_path = get_cache_filepath(DEBUG_IMAGES_SUBDIR)

    if create:
        readme_string = f'This directory contains data samples provided as inputs and outputs to the model for ' \
                        f'debugging.\n' \
                        f'The subdirectory indicates the index of the data sample in the dataset.\n' \
                        f'The images represent the pre-processed data sample being provided as input to the model.'
        create_subdir_if_not_exists(image_cache_path, readme_string=readme_string)

    return image_cache_path


def get_all_main_data_paths():
    """Returns a list of filepaths to all of the real geotiff files used in the dataset."""

    data_filepaths = get_input_data_filepaths() + get_output_data_paths()

    return data_filepaths


def get_input_data_filepaths():
    """Returns a list of input data filepaths in the main dataset."""

    data_filepaths = [get_data_filepath(path) for path in INPUT_FILENAMES]

    return data_filepaths


def get_output_data_paths():
    """Returns a list of output data filepaths in the main dataset."""

    data_filepaths = [get_data_filepath(path) for path in OUTPUT_FILENAMES]

    return data_filepaths
