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

        model = GibbsSamplingDMM(corpus, number_of_topics=20)
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


class GenerationTestsWithReplacement(unittest.TestCase):

    def setUp(self):
        """Code to run before each test."""
        corpus = Corpus.from_document_file("tests/data/sample_data")
        self.model = GibbsSamplingDMM(corpus, number_of_topics=20)
        self.model.randomly_initialise_topic_assignment(seed=1)
        self.model.inference(100)
        self.generated_documents, self.chosen_topics = self.model.generate_synthetic_documents(10, seed=5)

    def test_generated_documents_lengths(self):
        """Test that the lengths of the generated documents are correct."""
        document_sizes = [len(document) for document in self.generated_documents]
        expected_sizes = [6, 3, 5, 5, 3, 3, 3, 1, 7, 7]
        self.assertListEqual(document_sizes, expected_sizes)

    def test_generated_topics(self):
        """Test that the same topics have been generated."""
        expected_topics = [17, 17, 17, 17, 17, 17, 17, 17, 17, 17]
        self.assertListEqual(list(self.chosen_topics), expected_topics)

    # noinspection SpellCheckingInspection
    def test_generated_documents(self):
        """Test the generation of sentences with replacement."""
        expected_documents = [
            ["retweets", "messed", "people", "good", "trip", "media"],
            ["show", "today", "facebook"],
            ["good", "facebook", "show", "good", "thing"],
            ["back", "sleep", "account", "account", "people"],
            ["dead", "good", "people"],
            ["pass", "omg", "point"],
            ["good", "love", "dead"],
            ["people"],
            ["make", "dead", "sharepoint", "seo", "show", "love", "sleep"],
            ["social", "shout", "show", "today", "facebook", "add", "fuck"]
        ]
        self.assertListEqual(self.generated_documents, expected_documents)

    def test_sampling_without_replacement(self):
        """Test that sampling without replacement is not implemented."""
        with self.assertRaises(NotImplementedError, msg="Method should throw error as it is not yet implemented."):
            self.model.generate_synthetic_documents(10, seed=5, replacement=False)
