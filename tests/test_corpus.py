"""
Tests for the Corpus class.
"""
import unittest

import numpy as np

from pdmm import Corpus


class CorpusTest(unittest.TestCase):

    def test_corpus_creation(self):
        """Test that the Corpus instance is properly created."""
        list_of_documents = [
            ["the", "quick", "brown", "fox"],
            ["jumped", "over"],
            ["the", "lazy", "lazy", "dog"]
        ]
        corpus = Corpus.from_documents_as_lists_of_words(list_of_documents)

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
