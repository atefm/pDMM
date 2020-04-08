"""
Tests for the Corpus class.
"""
import os
import tempfile
import unittest

import numpy as np

from pdmm import Corpus


class CreationTests(unittest.TestCase):

    def setUp(self):
        """Code to run before every test."""
        self.list_of_documents = [
            ["the", "quick", "brown", "fox"],
            ["jumped", "over"],
            ["the", "lazy", "lazy", "dog"]
        ]
        self.tempdir = tempfile.TemporaryDirectory()
        self.corpus = Corpus.from_documents_as_lists_of_words(self.list_of_documents)

    def tearDown(self):
        """Code to run after every test."""
        self.tempdir.cleanup()

    def test_correct_documents(self):
        """Test that the correct documents have been created."""
        expected_documents = [[0, 1, 2, 3], [4, 5], [0, 6, 6, 7]]
        self.assertListEqual(expected_documents, self.corpus.documents, "The documents should be equal.")

    def test_correct_occurrence_to_index_count(self):
        """Test that the correct occurrence to index count has been created."""
        expected_occurrence_to_index_count = [[1, 1, 1, 1], [1, 1], [1, 1, 2, 1]]
        self.assertListEqual(expected_occurrence_to_index_count, self.corpus.occurrence_to_index_count)

    def test_correct_word_counts(self):
        """Test that the correct word count matrix has been created."""
        expected_word_counts_in_documents = np.array([
            [1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0],
            [1, 0, 0, 0, 0, 0, 2, 1]
        ])
        self.assertTrue(np.alltrue(expected_word_counts_in_documents == self.corpus.word_counts_in_documents))

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


class AttributeTests(unittest.TestCase):

    def setUp(self):
        """Code to run before every test."""
        self.list_of_documents = [
            ["the", "quick", "brown", "fox"],
            ["jumped", "over"],
            ["the", "lazy", "lazy", "dog"]
        ]
        self.corpus = Corpus.from_documents_as_lists_of_words(self.list_of_documents)

    def test_number_of_documents(self):
        """Test that the correct number of documents is reported."""
        self.assertEqual(self.corpus.number_of_documents, len(self.list_of_documents))

    def test_mean_document_length(self):
        """Test that the correct mean document length is reported."""
        expected_mean = sum(len(document) for document in self.list_of_documents) / len(self.list_of_documents)
        observed_mean = self.corpus.get_mean_document_length()
        self.assertEqual(observed_mean, expected_mean, "Mean document lengths ais not correct.")

    def test_bad_equality(self):
        """Test that a Corpus instance is not equal to an integer."""
        corpus = Corpus.from_documents_as_lists_of_words(self.list_of_documents)
        with self.assertRaises(TypeError):
            bool(corpus == 5)
