"""
Contains the Corpus class.
"""
from collections import Counter

import numpy as np

from .vocabulary import Vocabulary


class Corpus:
    """
    A class representing a document corpus.

    Attributes
    ----------
    documents : list[list[int]]
        A list of lists containing integers representing
        word indices.
    occurrence_to_index_count : list[list[int]]
        A list of lists containing integers representing
        relative occurrences of words within documents.
    word_counts_in_documents : np.ndarray[int, int]
        An array denoting the occurrences of each word
        within each document.
    vocabulary : pdmm.vocabulary.Vocabulary
        The vocabulary of the corpus.
    """
    def __init__(self, documents, occurrence_to_index_count, word_counts_in_documents, vocabulary):
        self.documents = documents
        self.occurrence_to_index_count = occurrence_to_index_count
        self.word_counts_in_documents = word_counts_in_documents
        self.vocab = vocabulary

    def __eq__(self, other):
        if not type(other) == type(self):
            raise TypeError("Can only compare to {} type.".format(type(self).__name__))

        return all([
            self.documents == other.documents,
            self.occurrence_to_index_count == other.occurrence_to_index_count,
            np.all(self.word_counts_in_documents == other.word_counts_in_documents),
            self.vocab == other.vocab
        ])

    def __iter__(self):
        for document in self.documents:
            yield document

    def __getitem__(self, item):
        return self.documents[item]

    @property
    def number_of_documents(self):
        """Return the number of documents in the corpus."""
        return len(self.documents)

    def get_mean_document_length(self):
        """Get the mean length of documents in the corpus."""
        sum_of_lengths = sum(len(document) for document in self)
        return sum_of_lengths / self.number_of_documents

    @classmethod
    def from_document_file(cls, file_path):
        """Create a Corpus instance from a document file."""
        with open(file_path, "r") as rf:
            list_of_documents = (line.rstrip().split() for line in rf.readlines())

        return cls.from_iterable_of_word_lists(list_of_documents)

    @classmethod
    def from_iterable_of_word_lists(cls, iterable_of_word_lists):
        """Create a Corpus instance from an iterable yielding lists of words."""
        vocab = Vocabulary()
        documents = []
        occurrence_to_index_count = []

        for list_of_words in iterable_of_word_lists:
            document = []
            word_occurrence_to_index_in_document_count = Counter()
            word_occurrence_to_index_in_document = []

            for word in list_of_words:
                word_id = vocab.get_id_from_word(word)
                document.append(word_id)
                word_occurrence_to_index_in_document_count[word] += 1

                word_occurrence_to_index_in_document.append(word_occurrence_to_index_in_document_count[word])

            documents.append(document)
            occurrence_to_index_count.append(word_occurrence_to_index_in_document)

        word_counts_in_documents = [np.bincount(document, minlength=vocab.size) for document in documents]
        word_counts_in_documents_as_array = np.array(word_counts_in_documents, dtype=np.int32)
        return cls(documents, occurrence_to_index_count, word_counts_in_documents_as_array, vocab)
