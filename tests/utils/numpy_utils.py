"""
Provides utilities for working with numpy arrays.
"""


import cv2
import inspect
import numpy as np
import torch

from utils.general_utils import check_required_and_keyword_args


def check_arguments(func):
    def wrapper(*args, **kwargs):

        def check_numpy_array(np_arr, check_keyword):
            if not isinstance(np_arr, np.ndarray):
                raise AttributeError(
                    f'Passed value for "{check_keyword}" was "{np_arr}" which should be or a numpy array '
                    f'with the color channel in the third position.')
            if not (len(np_arr.shape) == 2 or len(np_arr.shape) == 3):
                raise AttributeError(f'When passing a numpy array for "{check_keyword}", it should have '
                                     f'two or three dimensions.\n'
                                     f'Instead, the numpy array you passed has "{len(np_arr.shape)}" dimensions.')
            if len(np_arr.shape) == 3 and (np_arr.shape[2] > np_arr.shape[0] or np_arr.shape[2] > np_arr.shape[1]):
                raise AttributeError(f'When passing a numpy array for "{check_keyword}", the color channel must '
                                     f'be in the third dimension.\n'
                                     f'It looks like your color channel is not in the third dimension.\n'
                                     f'The shape of the passed numpy array was "{arg.shape}".')

        def check_torch_tensor(torch_tensor, check_keyword):
            if not isinstance(torch_tensor, torch.Tensor):
                raise AttributeError(f'Passed value for "{check_keyword}" was "{torch_tensor}" which should be '
                                     f'a pytorch tensor with the color channel in the first position.')
            if not len(torch_tensor.size()) == 3:
                raise AttributeError(f'When passing a torch tensor for "{check_keyword}", it should have '
                                     f'three dimensions.\n'
                                     f'Instead, the torch tensor you passed has "{len(arg.shape)}" dimensions.')
            if torch_tensor.size()[0] > torch_tensor.size()[1] or torch_tensor.size()[0] > torch_tensor.size()[2]:
                raise AttributeError(f'When passing a torch tensor for "{check_keyword}", the color channel must '
                                     f'be in the first dimension.\n'
                                     f'It looks like your color channel is not in the first dimension.\n'
                                     f'The shape of the passed pytorch tensor was "{arg.size()}".')

        check_required_and_keyword_args(func, args, kwargs)

        keywords = inspect.signature(func).parameters.keys()
        for keyword, arg in zip(keywords, args):
            kwargs[keyword] = arg

        check_keyword = 'np_arr_list'
        if check_keyword in kwargs:
            arg_list = kwargs[check_keyword]
            for arg in arg_list:
                check_numpy_array(arg, check_keyword)

        check_keyword = 'tensor_list'
        if check_keyword in kwargs:
            arg_list = kwargs[check_keyword]
            for arg in arg_list:
                check_torch_tensor(arg, check_keyword)

        check_keywords = ['np_arr', 'np_arr1', 'np_arr2']
        for check_keyword in check_keywords:
            if check_keyword in kwargs:
                arg = kwargs[check_keyword]
                check_numpy_array(arg, check_keyword)

        check_keyword = 'torch_tensor'
        if check_keyword in kwargs:
            arg = kwargs[check_keyword]
            check_torch_tensor(arg, check_keyword)

        check_keywords = ['auto_rescale', 'verbose']
        for check_keyword in check_keywords:
            if check_keyword in kwargs:
                arg = kwargs[check_keyword]
                if not isinstance(arg, bool):
                    raise AttributeError(f'Passed value for "{check_keyword}" must be a boolean.')

        check_keyword = 'similarity_threshold'
        if check_keyword in kwargs:
            arg = kwargs[check_keyword]
            if not isinstance(arg, (int, float)):
                raise AttributeError(f'Passed value for "{check_keyword}" must be an int or float.')

        check_keyword = 'dimensions'
        if check_keyword in kwargs:
            arg = kwargs[check_keyword]
            if not (isinstance(arg, tuple) and len(arg) == 2 and all(isinstance(n, int) for n in arg)):
                raise AttributeError(f'The passed value for "{check_keyword}" was "{arg}" which is invalid.\n'
                                     f'The value for "{check_keyword}" should "None" or '
                                     f'a tuple containing two integers.')

        check_keyword = 'min_max'
        if check_keyword in kwargs:
            arg = kwargs[check_keyword]
            if not (isinstance(arg, tuple) and len(arg) == 2 and all(isinstance(n, (int, float)) for n in arg)):
                raise AttributeError(f'Passed value for "{check_keyword}" was "{arg}".\n'
                                     f'It must be a tuple of length two containing either ints or floats.')

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

        return func(**kwargs)
    return wrapper


@check_arguments
def create_np_array_from_geotiff_bands(np_arr_list):
    """Concatenates the passed list of 2D numpy arrays along their color channel."""

    for i, np_arr in enumerate(np_arr_list):
        if len(np_arr.shape) == 2:
            np_arr_list[i] = np.expand_dims(np_arr, axis=2)

    np_arr = np.concatenate(np_arr_list, axis=2)
    np_arr = np_arr.astype('float32')

    return np_arr


@check_arguments
def concatenate_torch_tensor_color_channel(tensor_list):
    """Concatenates the passed numpy arrays in torch format along its color channel."""

    return torch.cat(tensor_list, dim=0)


@check_arguments
def convert_np_array_to_torch_format(np_arr):
    """Converts the format of the numpy array to the format expected by pytorch as input to a model."""

    np_arr = np.transpose(np_arr, (2, 0, 1))

    np_arr = np_arr.astype('float32')

    torch_array = torch.from_numpy(np_arr).float()

    return torch_array


@check_arguments
def convert_torch_tensor_to_np_array_format(torch_tensor):
    """Converts the data format output by a pytorch model into the format expected by the geotiff library."""

    np_arr = torch_tensor.detach().numpy()

    np_arr = np.transpose(np_arr, (1, 2, 0))

    np_arr = np_arr.astype('float32')

    return np_arr


@check_arguments
def rescale_array(np_arr, rescale_ratio=None, dimensions=None, replace_nan_with=0, clip_to_original_min_max=True):
    """
    Rescales the numpy array of dimensions (rows, cols, channels) to a new size.
    Only a rescale ratio OR dimensions should be passed.
    If an integer rescale ratio is passed, the image dimensions will be rescaled to be multiplied by the ratio.
    If a (width, height) tuple of integer dimensions are passed, the image will be rescaled to those dimensions.
    Sometimes the openCV rescale function will create nans in the rescaled output. The "replace_nan_with" argument will
    cause nans and infinities to be replaced with the passed value. If "replace_nan_with" is None, nans and infinities
    will not be replaced in the resulting array.
    The "clip_to_original_min_max" boolean indicates if the rescaled values outside the range of the minimum and
    maximum of the original array will be clipped such that the rescaled array has the same minimum and maximum values
    as the rescaled array.
    """

    if not np_arr.dtype == 'float32':
        raise AttributeError(f'Numpy array must be float32 for openCV rescale function to work properly.')

    start_min = np.amin(np_arr)
    start_max = np.amax(np_arr)

    # the openCV rescale function will automatically remove the color channel if its size is 1, so we will add it back
    add_color_channel = np_arr.shape[2] == 1

    if not dimensions:
        dimensions = (round(np_arr.shape[1] * rescale_ratio), round(np_arr.shape[0] * rescale_ratio))

    np_arr = cv2.resize(np_arr, dsize=dimensions, interpolation=cv2.INTER_CUBIC)

    if add_color_channel:
        np_arr = np.expand_dims(np_arr, axis=2)

    if replace_nan_with is not None:
        np_arr[np.isnan(np_arr)] = replace_nan_with
        np_arr[np.isinf(np_arr)] = replace_nan_with

    if clip_to_original_min_max:
        np_arr = np.clip(np_arr, start_min, start_max)

    return np_arr


@check_arguments
def normalize_array(np_arr, min_max=None):
    """
    Normalizes the values in the passed numpy array to be from 0 to 1.
    If "min_max" is None, the minimum and maximum values in the array will be calculated automatically.
    If "min_max" is a tuple of (min, max), the passed values will be used to normalize the data array instead.
    """

    import warnings
    warnings.filterwarnings("error")

    if min_max is None:
        min = np.amin(np_arr)
        max = np.amax(np_arr)
    else:
        min, max = min_max

    # don't allow too large of a difference to prevent division by zero
    min_max_diff = max - min
    if min_max_diff < .1:
        min_max_diff = 1

    norm_arr = np.copy(np_arr)

    assert not np.isinf(np_arr).any()
    assert not np.isinf(norm_arr).any()

    norm_arr -= min
    norm_arr = norm_arr / min_max_diff

    return norm_arr


@check_arguments
def approximately_same(np_arr1, np_arr2, similarity_threshold=0.02, auto_rescale=False):
    """
    Returns a boolean indicating if the average absolute difference between corresponding values in two arrays of
    the same shape is less than a threshold.
    If rescale=True adn the two arrays are different sizes the larger array will be rescaled to the same size as the
    larger array for comparison.
    """

    if not auto_rescale and not np_arr1.shape == np_arr2.shape:
        raise AttributeError(f'Passed arrays should have the same dimensions unless rescale=True.\n'
                             f'Instead, dimensions were: "{np_arr1.shape}" and "{np_arr2.shape}".')
    elif auto_rescale and not np_arr1.shape == np_arr2.shape:
        if np_arr1.size <= np_arr2.size:
            np_arr2 = rescale_array(np_arr2, dimensions=(np_arr1.shape[1], np_arr1.shape[0]))
        else:
            np_arr1 = rescale_array(np_arr1, dimensions=(np_arr2.shape[1], np_arr2.shape[0]))

    assert np_arr1.shape == np_arr2.shape

    total_min = min(np.amin(np_arr1), np.amin(np_arr2))
    total_max = max(np.amax(np_arr2), np.amax(np_arr2))
    min_max_diff = total_max - total_min
    if min_max_diff < .01:  # prevent too small values causing division by zero or near zero
        min_max_diff = 1

    return np.sum(np.abs(np_arr1 - np_arr2)) / (np_arr1.size * min_max_diff) < similarity_threshold


if __name__ == "__main__":
    pass
