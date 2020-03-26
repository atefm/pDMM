"""
Runs the tests.
"""
import unittest

import coverage


def run_tests():
    """Run all of the tests."""
    cov = coverage.Coverage(source=["pdmm"], omit=["pdmm/utils.py"])
    cov.start()

    test_loader = unittest.TestLoader()
    all_tests = test_loader.discover(".")
    test_runner = unittest.TextTestRunner()
    test_runner.run(all_tests)

    cov.stop()
    cov.exclude("if __name__ == .__main__.:")
    cov.html_report()


if __name__ == "__main__":
    run_tests()
