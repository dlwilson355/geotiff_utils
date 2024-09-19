"""
This file provides other general utilities that do not belong anywhere else.
"""


import inspect
import math

import numpy as np


def approx_equal_tuple(tuple1, tuple2, tolerance=0.01):
    assert len(tuple1) == len(tuple2)

    difference = 0
    for item1, item2 in zip(tuple1, tuple2):
        difference += abs(item1 - item2)

    return difference < tolerance


def get_lat_lon_diff_from_meters(reference_lat_lon, meters):
    ref_lat, ref_lon = reference_lat_lon
    r_earth = 6378000

    new_lat = ref_lat + (meters / r_earth) * (180 / math.pi)
    new_lon = ref_lon + (meters / r_earth) * (180 / math.pi) / math.cos(ref_lat * math.pi / 180)

    lat_diff = abs(ref_lat - new_lat)
    lon_diff = abs(ref_lon - new_lon)

    return lat_diff, lon_diff


def check_required_and_keyword_args(func, args, kwargs):
    """
    Checks that all the required arguments to a function were passed, and that all the keyword arguments are valid.
    """

    sig = inspect.signature(func)
    all_argument_names = [param.name for param in sig.parameters.values()]
    for kwarg in kwargs:
        if kwarg not in all_argument_names:
            raise ValueError(f'Function "{func.__name__}" does not have a parameter called "{kwarg}".')

    required_argument_names = [param.name for param in sig.parameters.values()
                               if param.default == param.empty]
    num_required_args = sum(1 for param in sig.parameters.values() if param.default == param.empty)
    num_passed_required_args = sum(1 for kwarg in kwargs if kwarg in required_argument_names) + len(args)
    if num_required_args > num_passed_required_args:
        raise ValueError(f'Function "{func.__name__}" expects {num_required_args} required argument(s) '
                         f'but got {num_passed_required_args} required argument(s) instead.\n'
                         f'Expected required parameters were "{required_argument_names}".')


def list_mean(list_object):
    """Calculates the mean of the values in the passed list."""

    return sum(list_object) / len(list_object)
