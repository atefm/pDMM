"""
Runs the tests.
"""
import unittest

import coverage


def run_tests():
    test_loader = unittest.TestLoader()
    all_tests = test_loader.discover(".")
    test_runner = unittest.TextTestRunner()

    cov = coverage.Coverage(source=["pdmm"])
    cov.start()
    test_runner.run(all_tests)
    cov.stop()

    cov.html_report()


if __name__ == "__main__":
    run_tests()
