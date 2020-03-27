"""
Utility functions for the tests package.
"""
import random

import numpy as np


def read_contents_from_path(file_path):
    """Read the contents of a file as a string."""
    with open(file_path, "r") as rf:
        contents = rf.read()
    return contents


def seed_numpy_with_same_seed_as_random(seed=None):
    """Seed the numpy random generator to the same state as the Python one."""
    standard_random_generator = random.Random(seed)
    version, (*mt_state, position), gauss_next = standard_random_generator.getstate()
    new_numpy_state = ("MT19937", mt_state, position)
    # noinspection PyTypeChecker
    np.random.set_state(new_numpy_state)
