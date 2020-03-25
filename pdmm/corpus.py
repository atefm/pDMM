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

    @property
    def number_of_documents(self):
        """Return the number of documents in the corpus."""
        return len(self.documents)

    @classmethod
    def from_document_file(cls, file_path):
        """Create a Corpus instance from a document file."""
        vocab = Vocabulary()
        documents = []
        occurrence_to_index_count = []

        with open(file_path, "r") as rf:
            for line in rf.readlines():
                document = []
                word_occurrence_to_index_in_document_count = Counter()
                word_occurrence_to_index_in_document = []

                words = line.rstrip().split()
                for word in words:
                    word_id = vocab.get_id_from_word(word)
                    document.append(word_id)
                    word_occurrence_to_index_in_document_count += 1

                    word_occurrence_to_index_in_document.append(word_occurrence_to_index_in_document_count[word])

                documents.append(document)
                occurrence_to_index_count.append(word_occurrence_to_index_in_document)

        word_counts_in_documents = np.array([np.bincount(document, minlength=vocab.size) for document in documents])
        return cls(documents, occurrence_to_index_count, word_counts_in_documents, vocab)
