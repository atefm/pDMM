"""
Tests for the sampling module.
"""
import time
import unittest

from pdmm import Corpus, GibbsSamplingDMM


class TimingTests(unittest.TestCase):
    """Test the timing of the inference."""

    def test_speed_of_inference(self):
        """Test that the inference is fast enough."""
        corpus = Corpus.from_document_file("tests/data/sample_data")

        model = GibbsSamplingDMM(
            corpus,
            number_of_topics=20,
        )
        model.randomly_initialise_topic_assignment(seed=1)
        number_of_iterations = 200

        t0 = time.time()
        model.inference(number_of_iterations)
        t1 = time.time()

        average_seconds_per_iteration = (t1 - t0) / number_of_iterations
        expected_seconds_per_iteration = 0.02
        delta = 0.05

        if average_seconds_per_iteration < expected_seconds_per_iteration - delta:
            self.fail("Code was faster: actually took {:.5f} per iteration.".format(average_seconds_per_iteration))
        else:
            self.assertAlmostEqual(average_seconds_per_iteration, expected_seconds_per_iteration, delta=0.05)
