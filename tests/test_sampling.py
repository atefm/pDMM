"""
Tests for the sampling module.
"""
import os
import tempfile
import time
import unittest

from pdmm import Corpus, GibbsSamplingDMM

from .utils import read_contents_from_path


class BasicTests(unittest.TestCase):

    def setUp(self):
        """Code to run before each test."""
        corpus = Corpus.from_document_file("tests/data/sample_data")
        self.model = GibbsSamplingDMM(corpus, number_of_topics=20)
        self.model.randomly_initialise_topic_assignment(seed=1)
        self.model.inference(50)

    def test_top_words(self):
        """Test that the top 10 words are correct."""
        topic_index = 3
        expected_top_words = ["siri", "iphone", "year", "word", "minutes", "doc", "blackberry", "sooo", "glad", "gave"]
        observed_top_words = self.model.get_top_words_for_topic(topic_index, number_of_top_words=10)
        self.assertListEqual(expected_top_words, observed_top_words, "Top words are not correct.")


class FileTests(unittest.TestCase):

    def setUp(self):
        """Code to run before each test."""
        corpus = Corpus.from_document_file("tests/data/sample_data")
        self.model = GibbsSamplingDMM(corpus, number_of_topics=20)
        self.model.randomly_initialise_topic_assignment(seed=1)
        self.model.inference(5)

        self.tempdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        """Code to run at the end of each test."""
        self.tempdir.cleanup()

    def test_saving_top_words(self):
        """Test that the topWords file is created properly."""
        file_path = os.path.join(self.tempdir.name, "topWords")
        self.model.save_top_topical_words_to_file(file_path)

        lines = []
        for topic_index in range(self.model.number_of_topics):
            top_words = self.model.get_top_words_for_topic(topic_index)
            joined_top_words = " ".join(top_words)
            line = "Topic {}: {} \n".format(topic_index, joined_top_words)
            lines.append(line)

        expected_saved_string = "".join(lines)
        observed_saved_string = read_contents_from_path(file_path)
        self.assertEqual(expected_saved_string, observed_saved_string)

    def test_saving_assignments(self):
        """Test that the topicAssignments file is created properly."""
        file_path = os.path.join(self.tempdir.name, "topicAssignments")
        self.model.save_topic_assignments_to_file(file_path)

        lines = []
        for document_index in range(self.model.corpus.number_of_documents):
            assignment = self.model.document_topic_assignments[document_index]
            line = "{}\n".format(assignment)
            lines.append(line)

        expected_saved_string = "".join(lines)
        observed_saved_string = read_contents_from_path(file_path)
        self.assertEqual(expected_saved_string, observed_saved_string)


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
