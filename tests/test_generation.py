"""
Tests for generation after inference.
"""
import unittest

from pdmm import Corpus, GibbsSamplingDMM


class GenerationTests(unittest.TestCase):

    def setUp(self):
        """Code to run before each test."""
        corpus = Corpus.from_document_file("tests/data/sample_data")
        self.model = GibbsSamplingDMM(
            corpus,
            number_of_topics=20,
        )
        self.model.randomly_initialise_topic_assignment(seed=1)
        self.model.inference(50)

    def test_generation_with_replacement(self):
        """Test the generation of sentences witgh replacement."""
        generated_documents = self.model.generate_synthetic_documents(10, seed=5)
        for document in generated_documents:
            print(document)