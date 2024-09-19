"""
This file provides the dataset used to split and load the data samples for training and testing a model.

Useful reads:
https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
"""


import os.path
import shutil

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.data_cache import get_min_max_cache, get_class_ratio_cache
from utils.file_utils import get_debug_images_save_directory, create_subdir_if_not_exists
from utils.geotiff_utils import get_geotiff_min_max_values, get_geotiff_value_counts, read_geotiff_within_lat_lon, \
    create_geotiff, make_arc_gis_ready
from utils.numpy_utils import concatenate_torch_tensor_color_channel
from utils.pytorch_utils import one_hot_encode
from utils.shapefile_utils import get_grid_cells_inside_shape, write_lat_lon_grid_cells_to_geotiff


def get_class_ratios(geotiff_path, use_cache=True):
    """
    Returns a dictionary in which each class is a key and each value indicates the proportion of the total outputs
    with that class label.
    """

    def calculate_class_ratios():
        # count the occurrences of each class in the output
        class_counts = get_geotiff_value_counts(geotiff_path)

        # normalize the counts
        all_count = sum([count for count in list(class_counts.values())])
        class_weights = dict()
        for key in class_counts.keys():
            class_weights[key] = class_counts[key] / all_count

        return class_weights

    if use_cache:
        cache = get_class_ratio_cache()
        class_ratios = cache.load_from_storage(os.path.basename(geotiff_path))
        if class_ratios:
            return class_ratios
        else:
            print(f"No class ratio cache found for {geotiff_path}.\n"
                  f"The class weights will be calculated manually and then cached for future use.")
            class_ratios = calculate_class_ratios()
            cache.save_to_storage(os.path.basename(geotiff_path), class_ratios)
            return class_ratios
    else:
        return calculate_class_ratios()


def get_num_classes(geotiff_path, use_cache=True):
    """Returns an integer indicating how many unique classes are present in the output file."""

    return len(get_class_ratios(geotiff_path, use_cache=use_cache).keys())


def get_cell_assignments(shape_filepath,
                         cell_size_meters,
                         training_proportion=0.8,
                         validation_proportion=0.1,
                         testing_proportion=0.1,
                         cell_cache=None):
    """
    Calculates cells within the shapefile and assigns them to the training, validation, and testing subsets.
    Passing an optional DataCache object will allow the cell assignments to be cached and loaded faster in the future.
    """

    assert training_proportion + validation_proportion + testing_proportion == 1, \
        f"The proportion of training, validation, and testing data must add to 1."

    if cell_cache:
        key = (shape_filepath,
               training_proportion,
               validation_proportion,
               testing_proportion,
               cell_size_meters)
        cell_list = cell_cache.load_from_storage(key)
        if cell_list:
            return cell_list
        else:
            print(f'Cell assignments matching the parameters "{key}" were not found in the cell assignment cache.\n'
                  f'The cell assignments will be calculated manually and then cached for future use.')
            cell_list = get_grid_cells_inside_shape(shape_filepath=shape_filepath,
                                                    meters_cell_size=(cell_size_meters, cell_size_meters))
            cell_assignments = assign_cells_to_subsets(cell_list,
                                                       training_proportion,
                                                       validation_proportion,
                                                       testing_proportion)
            cell_cache.save_to_storage(key, cell_assignments)
            return cell_assignments
    else:
        cell_list = get_grid_cells_inside_shape(shape_filepath=shape_filepath,
                                                meters_cell_size=(cell_size_meters, cell_size_meters))
        return assign_cells_to_subsets(cell_list,
                                       training_proportion,
                                       validation_proportion,
                                       testing_proportion)


def assign_cells_to_subsets(cell_list, training_proportion, validation_proportion, testing_proportion):
    """
    Returns a dictionary where the passed list of cells are assigned to keys indicating the
    training, validation, and testing set.
    """

    assert training_proportion + validation_proportion + testing_proportion == 1, \
        f"The proportion of training, validation, and testing data must add to 1."

    # assign each cell to the training, validation, or testing set in the appropriate proportions
    target_proportions = {"training": training_proportion,
                          "validation": validation_proportion,
                          "testing": testing_proportion}
    cell_assignments = {"training": [], "validation": [], "testing": []}
    for cell in cell_list:
        proportion_differences = {}
        for key in cell_assignments.keys():
            proportion_differences[key] = len(cell_assignments[key]) / len(cell_list) - target_proportions[key]
        min_value = min(proportion_differences.values())
        min_key = [key for key in proportion_differences if proportion_differences[key] == min_value][0]
        cell_assignments[min_key].append(cell)

    # check to make sure the proportions assigned to each subset are correct
    difference_tolerance = 0.01
    assert len(cell_assignments["training"]) / len(cell_list) - training_proportion < difference_tolerance
    assert len(cell_assignments["validation"]) / len(cell_list) - training_proportion < difference_tolerance
    assert len(cell_assignments["testing"]) / len(cell_list) - training_proportion < difference_tolerance

    return cell_assignments


class GeotiffDataset(Dataset):
    def __init__(self,
                 input_paths,
                 output_path,
                 cell_list,
                 rescale_dimensions=(224, 224),
                 use_min_max_cache=True,
                 print_nan_warning=False,
                 save_input_images=False):

        self.input_data_paths = input_paths
        self.output_data_path = output_path
        self.cell_list = cell_list
        self.rescale_dimensions = rescale_dimensions
        self.use_min_max_cache = use_min_max_cache
        if self.use_min_max_cache:
            self.use_min_max_cache = get_min_max_cache(create=True)
        self.num_classes = get_num_classes(self.output_data_path)
        self.print_nan_warning = print_nan_warning
        self.save_input_images = save_input_images
        if self.save_input_images and os.path.exists(get_debug_images_save_directory(create=False)):
            shutil.rmtree(get_debug_images_save_directory(create=False))
        # for i in range(0, self.__len__()):
        #     print(i)
        #     self.__getitem__(i)

    def __len__(self):
        return len(self.cell_list)
        # return 20

    def __getitem__(self, idx):
        if not idx < len(self):
            raise ValueError(f'Index value must be smaller then length.\n'
                             f'Passed index was: "{idx}" and length is: "{self.__len__()}".')

        # sample coordinates and construct and input and output tensor
        coords = self.cell_list[idx]
        input = self.read_data_samples(self.input_data_paths, *coords, idx)
        output = self.read_data_samples([self.output_data_path], *coords, idx)
        output = one_hot_encode(output, self.num_classes)

        # handle the situation where there are "NANs" in tensor
        if self.print_nan_warning and torch.any(torch.isnan(input)) or torch.any(torch.isnan(output)):
            print(f'\nNAN data loaded at index: {idx} and coordinates: {coords}.')
            print(torch.any(torch.isnan(input)))
            print(torch.any(torch.isnan(output)))
            # matches = input == torch.nan
            # print(matches.nonzero(as_tuple=True)[0])
            # print(input.size()[0] * input.size()[1] * input.size()[2])
            # print(torch.sum(input == torch.nan))
            # print(torch.unique(matches, return_counts=True))
            # matches = output == torch.nan
            # print(matches.nonzero(as_tuple=True)[0])
            # print(output.size()[0] * output.size()[1] * output.size()[2])
            # print(torch.sum(output == torch.nan))
            # print(torch.unique(matches, return_counts=True))
            # print(torch.nonzero(torch.isnan(input.view(-1))))
            # print(torch.nonzero(torch.isnan(output.view(-1))))
            # print("counts above")
            # raise ValueError('Encountered NAN (see above).'
            #                  'Since "resample_nans" is "False", an exception is raised instead of resampling.')

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        input.to(device)
        output.to(device)

        return input, output

    def get_input_sample_depth(self):
        """Returns the depth of the input tensors."""

        input_depth = self.read_data_samples(self.input_data_paths,
                                             *self.cell_list[0],
                                             0).size()[0]

        return input_depth

    def get_num_classes(self, use_cache=True):
        """Returns the number of classes in the output geotiff file."""

        return get_num_classes(self.output_data_path, use_cache=use_cache)

    def get_min_max_values(self, geotiff_path):
        """Returns the minimum on maximum value inside the array for the passed geotiff."""

        if self.use_min_max_cache:
            min_max = self.use_min_max_cache.load_from_storage(os.path.basename(geotiff_path))
            # look to the geotiff path in the cache, calculate its min and max manually and add it if not available
            if min_max:
                return min_max
            else:
                print(f'Geotiff: "{geotiff_path}" not found in min_max cache.\n'
                      f'Its min max values will be calculated manually and then cached for future use.')
                min_max = get_geotiff_min_max_values(geotiff_path, replace_no_data_value_with=0, verbose=False)
                self.use_min_max_cache.save_to_storage(os.path.basename(geotiff_path), min_max)
                return min_max
        else:
            return get_geotiff_min_max_values(geotiff_path)

    def read_data_samples(self, geotiff_paths, top_left, bottom_right, idx):
        if not isinstance(geotiff_paths, list):
            raise AttributeError(f'Passed geotiff paths should be a list. '
                                 f'Received: "{type(geotiff_paths)}" object instead.')
        for geotiff_path in geotiff_paths:
            if not isinstance(geotiff_path, str):
                raise AttributeError(f'Passed geotiff paths in the list should be strings. '
                                     f'Received "{type(geotiff_path)}" object instead.')
        for coordinate in [top_left, bottom_right]:
            if coordinate is not None and not \
                    (isinstance(coordinate, tuple) and len(coordinate) == 2 and all(
                        isinstance(n, (float, int)) for n in coordinate)):
                raise AttributeError(f'Passed coordinate was: "{coordinate}" which is not a valid coordinate.')

        data_sample = []
        for data_path in geotiff_paths:
            min_max = self.get_min_max_values(data_path)
            data_array = read_geotiff_within_lat_lon(data_path,
                                                     top_left,
                                                     bottom_right,
                                                     convert_to_pytorch_tensor_format=True,
                                                     replace_no_data_value_with=0,
                                                     replace_nan_with=0,
                                                     rescale=self.rescale_dimensions,
                                                     normalize=min_max)
            if self.save_input_images:
                self.save_data_sample(data_array, data_path, idx, [top_left, bottom_right])
            data_sample.append(data_array)

        data_sample = concatenate_torch_tensor_color_channel(data_sample)

        return data_sample

    def save_data_sample(self, data_array, data_path, idx, coords):

        # find the save directory and create it if is doesn't exist
        save_data_index_dir = os.path.join(get_debug_images_save_directory(), str(idx))
        save_data_sample_path = os.path.join(save_data_index_dir,
                                             os.path.splitext(os.path.basename(data_path))[0]) + '.png'
        create_subdir_if_not_exists(save_data_index_dir,
                                    f'This file lists each data sample and its statistics.\n'
                                    f'This data sample was created from coordinates: {coords}.\n\n')

        # save the data array as an image
        temp = np.transpose(data_array, (1, 2, 0))
        save_data = np.zeros((224, 224, 3))
        for i in range(min(temp.shape[2], 2)):
            save_data[:, :, i] = temp[:, :, i]
        save_data = save_data * 255
        cv2.imwrite(save_data_sample_path, save_data)

        # record statistics from the data array
        num_channels = temp.shape[2]
        min_val = np.amin(save_data)
        max_val = np.amax(save_data)
        std = np.std(save_data)
        with open(os.path.join(save_data_index_dir, 'README.txt'), 'a') as f:
            f.write(f'Filename: {os.path.basename(data_path)}, '
                    f'Num Channels: {num_channels}, '
                    f'Min: {round(min_val, 2)}, '
                    f'Max: {round(max_val, 2)}, '
                    f'STD: {round(std, 2)}\n')

    def draw_cells_to_geotiff(self, color=(0, 0, 0)):
        """Draws the cells contained by the dataset to the passed geotiff file."""

        output_geotiff_filepath = os.path.join(get_debug_images_save_directory(), 'cells.tiff')

        if not os.path.exists(output_geotiff_filepath):
            create_geotiff(output_geotiff_filepath, self.output_data_path, num_bands=3, no_data_values=[-128, -128, -128])

        write_lat_lon_grid_cells_to_geotiff(output_geotiff_filepath, self.cell_list, relative_cell_width=0.1, color=color)

        make_arc_gis_ready(output_geotiff_filepath)
