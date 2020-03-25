"""
Utility functions for the tests package.
"""
import random


def read_contents_from_path(file_path):
    """Read the contents of a file as a string."""
    with open(file_path, "r") as rf:
        contents = rf.read()
    return contents


def python_2_randint(a, b):
    """Equivalent to random.randint in Python2."""
    return a + int(random.random() * (b + 1 - a))
