"""
Runs the tests.
"""
import sys
import unittest

import coverage


def run_tests():
    """Run all of the tests."""
    cov = coverage.Coverage(source=["pdmm"], omit=["pdmm/utils.py"])
    cov.start()

    try:
        test_pattern_component = sys.argv[1]
    except IndexError:
        test_pattern_component = "*"

    test_pattern = "test_{}.py".format(test_pattern_component)

    test_loader = unittest.TestLoader()
    all_tests = test_loader.discover(".", pattern=test_pattern)
    test_runner = unittest.TextTestRunner()
    test_runner.run(all_tests)

    cov.stop()
    cov.exclude("if __name__ == .__main__.:")
    cov.html_report()


if __name__ == "__main__":
    run_tests()
