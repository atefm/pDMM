"""
Tests for reproducible randomness.
"""
import random
import unittest

import numpy as np

from .utils import seed_numpy_with_same_seed_as_random


class RandomTests(unittest.TestCase):

    def test_random_floats(self):
        """Test that the same random floats are generated."""
        seed_numpy_with_same_seed_as_random(1)
        observed_floats = np.random.random(4)
        expected_floats = np.array([0.13436424411240122, 0.8474337369372327, 0.763774618976614, 0.2550690257394217])
        self.assertTrue(np.all(observed_floats == expected_floats))

    def test_random_integers(self):
        """Test that the same random integers are generated."""
        seed_numpy_with_same_seed_as_random(1)
        observed_ints = np.random.randint(0, 1000, size=9)
        expected_ints = np.array([137, 582, 867, 821, 782, 64, 261, 120, 507])
        self.assertTrue(np.all(observed_ints == expected_ints))

    def test_random_uniform(self):
        """Test that the same uniformly distributed values are generated."""
        seed_numpy_with_same_seed_as_random(1)
        observed_values = np.random.uniform(0, 1000, size=4)
        expected_values = np.array([134.36424411240122, 847.4337369372327, 763.7746189766141, 255.0690257394217])
        self.assertTrue(np.all(observed_values == expected_values))
