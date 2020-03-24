"""
Runs the tests.
"""
import unittest


def run_tests():
    test_loader = unittest.TestLoader()
    all_tests = test_loader.discover(".")
    test_runner = unittest.TextTestRunner()
    test_runner.run(all_tests)


if __name__ == "__main__":
    run_tests()
