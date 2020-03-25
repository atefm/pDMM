"""
Tests for reproducible randomness.
"""
import random
import unittest


class RandomTests(unittest.TestCase):

    def test_random_floats(self):
        """Test that the same random floats are generated."""
        random.seed(1)
        observed_floats = [random.random() for __ in range(4)]
        expected_floats = [0.13436424411240122, 0.8474337369372327, 0.763774618976614, 0.2550690257394217]
        self.assertListEqual(observed_floats, expected_floats)

    def test_random_integers(self):
        """Test that the same random integers are generated."""
        random.seed(1)
        observed_ints = [random.randint(0, 1000) for __ in range(9)]
        expected_ints = [134, 848, 764, 255, 495, 449, 652, 789, 93]
        self.assertListEqual(observed_ints, expected_ints)
