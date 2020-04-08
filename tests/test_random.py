"""
Tests for random sampling. These will only break when built-in random functions have been affected by updates.
"""
import unittest

import numpy as np

import pdmm.utils


class RandomTests(unittest.TestCase):

    def setUp(self):
        """Code to run before each test."""
        np.random.seed(19)
        weights = np.random.random(20)
        self.cumulative_weights = weights.cumsum()

        np.random.seed(5)
        self.random_numbers = np.random.random(20)
        self.expected_values = [6, 17, 6, 18, 9, 11, 15, 9, 7, 5, 1, 14, 8, 4, 18, 7, 8, 7, 11, 11]

    def test_random_sampling(self):
        """Test that the cumulative sampling is reproducible.."""
        sampled_values = []

        for random_number in self.random_numbers:
            sampled_value = pdmm.utils.sample_from_cumulative_weights(self.cumulative_weights, random_number)
            sampled_values.append(sampled_value)

        self.assertListEqual(sampled_values, self.expected_values, "Sampled values were not as expected.")

    def test_multiple_sampling(self):
        """Test that sampling all at once produces the same results."""
        sampled_values = pdmm.utils.sample_many_from_cumulative_weights(self.cumulative_weights, self.random_numbers)
        self.assertListEqual(list(sampled_values), self.expected_values, "Sampled values were not as expected.")
