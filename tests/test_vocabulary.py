"""
Tests for the Vocabulary class.
"""
import os
import tempfile
import unittest

from pdmm import Vocabulary


class VocabularyTests(unittest.TestCase):

    def setUp(self):
        """Code to run at the start of every test."""
        self.list_of_words = ["apple", "banana", "orange"]
        self.tempdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        """Code to run after every test."""
        self.tempdir.cleanup()

    def test_creation(self):
        """Test that the instance will be created properly."""
        id_to_word = {index: word for index, word in enumerate(self.list_of_words)}
        vocab = Vocabulary()

        for index in range(len(id_to_word)):
            word = id_to_word[index]
            vocab.get_id_from_word(word)

        for index, expected_word in id_to_word.items():
            observed_word = vocab.get_word_from_id(index)
            self.assertEqual(expected_word, observed_word)

    def test_saving_and_loading(self):
        """Test that a file can be successfully saved and loaded."""
        file_path = os.path.join(self.tempdir.name, "file")
        vocab = Vocabulary.from_list_of_words(self.list_of_words)
        vocab.save_to_file(file_path)
        loaded_vocab = Vocabulary.load_from_file(file_path)
        self.assertEqual(vocab, loaded_vocab)


class EqualityTests(unittest.TestCase):

    def setUp(self):
        """Code to run at the start of every test."""
        self.list_of_words = ["apple", "banana", "orange"]
        self.vocab = Vocabulary.from_list_of_words(self.list_of_words)

    def test_bad_equality(self):
        """Test that a Vocabulary instance is not equal to an integer."""
        with self.assertRaises(TypeError):
            self.assertNotEqual(self.vocab, 5)

    def test_false_equality_size(self):
        """Test that two vocabularies of different sizes are not equal."""
        other_vocab = Vocabulary.from_list_of_words(self.list_of_words)
        other_vocab.get_id_from_word("grapes")
        self.assertNotEqual(self.vocab, other_vocab)

    def test_inequality(self):
        """Test that vocabs with different words and the same size are not equal."""
        other_list_of_words = self.list_of_words[:]
        other_list_of_words[0] = "grapes"
        other_vocab = Vocabulary.from_list_of_words(other_list_of_words)
        self.assertNotEqual(self.vocab, other_vocab)
