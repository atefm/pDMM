"""
Tests for random sampling.
"""
import unittest

import numpy as np

from pdmm.utils import sample_from_cumulative_weights


class RandomTests(unittest.TestCase):

    def setUp(self):
        """Code to run before each test."""
        np.random.seed(19)
        self.weights = np.random.random(20)

        np.random.seed(5)
        self.random_numbers = np.random.random(20)
        self.expected_values = [1, 6, 1, 6, 6, 6, 6, 6, 1, 1, 0, 6, 4, 1, 6, 1, 4, 1, 6, 6]

    def test_random_sampling(self):
        """Test that the cumulative sampling is reproducible.."""
        sampled_values = []

        for random_number in self.random_numbers:
            sampled_value = sample_from_cumulative_weights(self.weights, random_number)
            sampled_values.append(sampled_value)

        self.assertListEqual(sampled_values, self.expected_values)
