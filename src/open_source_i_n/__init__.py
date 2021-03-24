"""OpenSource I/N package."""
import os


def flatten_dict_to_list(some_dict):
    """Flatten a dict into a list, dropping any `None` values."""
    return [str(item) for pair in some_dict.items() for item in pair if item is not None]


# Prevent libraries from doing extra threading. Need to do this before they load.
os.environ['MKL_NUM_THREADS'] = "1"
os.environ['VECLIB_MAXIMUM_THREADS'] = "1"
os.environ['OPENBLAS_NUM_THREADS'] = "1"
