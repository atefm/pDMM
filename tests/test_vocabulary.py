"""
Tests for the Vocabulary class.
"""
import unittest

from pdmm import Vocabulary


class VocabularyTests(unittest.TestCase):

    def test_creation(self):
        """Test that the instance will be created properly."""
        id_to_word = {0: "Apple", 1: "Banana", 2: "Orange "}
        vocab = Vocabulary()

        for index in range(len(id_to_word)):
            word = id_to_word[index]
            vocab.get_id_from_word(word)

        for index, expected_word in id_to_word.items():
            observed_word = vocab.get_word_from_id(index)
            self.assertEqual(expected_word, observed_word)
