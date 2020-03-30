"""
Tests for generation after inference.
"""
import unittest

from pdmm import Corpus, GibbsSamplingDMM


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
        self.assertListEqual(self.chosen_topics, expected_topics)

    def test_generated_documents(self):
        """Test the generation of sentences with replacement."""
        expected_documents = [
            ["people", "facebook", "ill", "thing", "amp", "join"],
            ["show", "section", "retweets"],
            ["people", "good", "trip", "media", "show"],
            ["facebook", "good", "facebook", "show", "good"],
            ["back", "sleep", "account"],
            ["people", "dead", "good"],
            ["pass", "omg", "point"],
            ["love"],
            ["people", "make", "dead", "sharepoint", "seo", "show", "love"],
            ["social", "shout", "show", "today", "facebook", "add", "fuck"]
        ]
        self.assertListEqual(self.generated_documents, expected_documents)
