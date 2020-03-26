"""
Tests for the Corpus class.
"""
import os
import tempfile
import unittest

import numpy as np

from pdmm import Corpus


class CorpusTest(unittest.TestCase):

    def setUp(self):
        """Code to run before every test."""
        self.list_of_documents = [
            ["the", "quick", "brown", "fox"],
            ["jumped", "over"],
            ["the", "lazy", "lazy", "dog"]
        ]
        self.tempdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        """Code to run after every test."""
        self.tempdir.cleanup()

    def test_corpus_creation(self):
        """Test that the Corpus instance is properly created."""
        corpus = Corpus.from_documents_as_lists_of_words(self.list_of_documents)

        expected_documents = [[0, 1, 2, 3], [4, 5], [0, 6, 6, 7]]
        expected_occurrence_to_index_count = [[1, 1, 1, 1], [1, 1], [1, 1, 2, 1]]
        expected_word_counts_in_documents = np.array([
            [1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0],
            [1, 0, 0, 0, 0, 0, 2, 1]
        ])

        self.assertListEqual(expected_documents, corpus.documents, "The documents should be equal.")
        self.assertListEqual(expected_occurrence_to_index_count, corpus.occurrence_to_index_count)
        self.assertTrue(np.alltrue(expected_word_counts_in_documents == corpus.word_counts_in_documents))

    def test_number_of_document(self):
        """Test that the correct number of documents is reported."""
        corpus = Corpus.from_documents_as_lists_of_words(self.list_of_documents)
        self.assertEqual(corpus.number_of_documents, len(self.list_of_documents))

    def test_creation_from_file(self):
        """Test that a Corpus instance is properly created from a file."""
        file_path = os.path.join(self.tempdir.name, "file")
        with open(file_path, "w") as wf:
            for document in self.list_of_documents:
                line = " ".join(document) + "\n"
                wf.write(line)

        corpus_from_file = Corpus.from_document_file(file_path)
        corpus_from_documents = Corpus.from_documents_as_lists_of_words(self.list_of_documents)
        self.assertEqual(corpus_from_documents, corpus_from_file)

    def test_bad_equality(self):
        """Test that a Corpus instance is not equal to an integer."""
        corpus = Corpus.from_documents_as_lists_of_words(self.list_of_documents)
        with self.assertRaises(TypeError):
            self.assertFalse(corpus == 5)
