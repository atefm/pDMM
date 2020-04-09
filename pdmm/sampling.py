"""
Contains the GibbsSamplingDMM class.
"""
from collections import Counter
import logging

import numpy as np

from .utils import sample_from_cumulative_weights, sample_many_from_cumulative_weights


class GibbsSamplingDMM:
    """
    Gibbs sampler for Dirichlet multinomial mixture process.

    Attributes
    ----------
    corpus : pdmm.corpus.Corpus
            The corpus to be analysed.
    number_of_topics : int
        The number of topics.
    alpha : float
        The hyper-parameter alpha
    beta : float
        The hyper-parameter beta.
    document_topic_assignments : list[int]
        A list of the topic indexes, where the ith element
        is the topic index to which document i has been assigned.
    number_of_documents_in_each_topic : list[int]
        The number of documents in each topic.
    number_of_each_word_in_each_topic : list[list[int]]
        A list of lists containing the counts of each word
        within each topic.
    number_of_total_words_in_each_topic : list[int]
        The number of words total within each topic.
    topic_weights : list[float]
        The weights for each of the topics.
    logger : logging.Logger
        The logger for the class.
    """
    def __init__(self, corpus, number_of_topics=20, alpha=0.1, beta=0.001):
        """
        Initialise self.

        Parameters
        ----------
        corpus : pdmm.corpus.Corpus
            The corpus to be analysed.

        Optional Parameters
        -------------------
        number_of_topics : int, defaults to 20
            The number of topics.
        alpha : float, defaults to 0.1
            The hyper-parameter alpha
        beta : float, defaults to 0.001
            The hyper-parameter beta.
        """
        self.corpus = corpus
        self.number_of_topics = number_of_topics
        self.alpha = alpha
        self.beta = beta

        self.document_topic_assignments = np.zeros((self.corpus.number_of_documents,))
        self.number_of_documents_in_each_topic = np.zeros((self.number_of_topics,))

        self.number_of_each_word_in_each_topic = np.zeros((self.number_of_topics, self.corpus.vocab.size))
        self.number_of_total_words_in_each_topic = np.zeros((self.number_of_topics,))
        self.topic_weights = np.ones((self.number_of_topics,))

        self.logger = logging.getLogger(__name__)

    def randomly_initialise_topic_assignment(self, seed=None):
        """Randomly assign topics to each of the documents."""
        np.random.seed(seed)
        self.document_topic_assignments = np.random.randint(0, self.number_of_topics, self.corpus.number_of_documents)
        self.number_of_documents_in_each_topic = np.bincount(self.document_topic_assignments,
                                                             minlength=self.number_of_topics)

        for document_index, new_topic in enumerate(self.document_topic_assignments):
            self._assign_document_to_topic(document_index, new_topic)

    def inference(self, number_of_iterations):
        """
        Run inference for a number of iterations.

        Parameters
        ----------
        number_of_iterations : int
            The number of iterations to run.

        Notes
        -----
        - This implements the second 'for' loop from the algorithm
          in Yin's paper [1].
        """
        for iteration in range(1, number_of_iterations + 1):
            self.logger.debug("Sampling in iteration {} of {}".format(iteration, number_of_iterations))
            self._sample_in_single_iteration()

    def generate_synthetic_documents(self, number_of_documents, replacement=True, seed=None):
        """Generate new synthetic documents according to the model."""
        np.random.seed(seed)

        if not replacement:
            raise NotImplementedError("Not yet possible to sample without replacement.")

        mean_document_length_in_corpus = self.corpus.get_mean_document_length()
        document_lengths = np.random.poisson(mean_document_length_in_corpus, size=number_of_documents)
        documents = []

        cumulative_topic_weights = self.topic_weights.cumsum()
        cumulative_word_weights_for_all_topics = self.number_of_each_word_in_each_topic.cumsum(axis=1)

        random_numbers_for_topics = np.random.random(number_of_documents)
        topic_indices = sample_many_from_cumulative_weights(cumulative_topic_weights, random_numbers_for_topics)
        chosen_topics = topic_indices

        for i in range(number_of_documents):
            document_length = document_lengths[i]
            topic_index = topic_indices[i]
            cumulative_word_weights = cumulative_word_weights_for_all_topics[topic_index]
            random_numbers_for_words = np.random.random(document_length)
            word_indices = sample_many_from_cumulative_weights(cumulative_word_weights, random_numbers_for_words)
            words = [self.corpus.vocab.get_word_from_id(word_index) for word_index in word_indices]
            documents.append(words)
        return documents, chosen_topics

    def get_top_words_for_topic(self, topic_index, number_of_top_words=20):
        """
        Get a list of the most common words in a topic.

        Parameters
        ----------
        topic_index: int
            The index of the desired topic.

        Optional Parameters
        -------------------
        number_of_top_words : int
            The number of top words to return.

        Returns
        -------
        top_words : list[str]
            A list of the most common words as strings.
        """
        number_of_each_word_in_topic = self.number_of_each_word_in_each_topic[topic_index]
        word_counts = Counter(dict(enumerate(number_of_each_word_in_topic)))
        most_common_elements = word_counts.most_common(number_of_top_words)
        top_words = [self.corpus.vocab.get_word_from_id(word_id) for word_id, count in most_common_elements]

        return top_words

    def save_top_topical_words_to_file(self, file_path, number_of_top_words=20):
        """
        Save the top words in the topics to a file.

        Parameters
        ----------
        file_path : str
            The location at which to save the file.

        Optional Parameters
        -------------------
        number_of_top_words : int
            The number of top words from each topic to save.
        """
        with open(file_path, "w") as wf:
            for topic_index in range(self.number_of_topics):
                top_words = self.get_top_words_for_topic(topic_index, number_of_top_words=number_of_top_words)

                line = "Topic {}: {} \n".format(topic_index, " ".join(top_words))
                wf.write(line)

    def save_topic_assignments_to_file(self, file_path):
        """
        Save the topic assignments to a file.

        Parameters
        ----------
        file_path : str
            The location at which to save the file.
        """
        with open(file_path, "w") as wf:
            for document_index in range(self.corpus.number_of_documents):
                line = str(self.document_topic_assignments[document_index]) + "\n"
                wf.write(line)

    def _sample_in_single_iteration(self):
        """
        Sample in a single iteration.

        Notes
        -----
        - This implements the second 'for' loop from the algorithm
         in Yin's paper [1].
        - These steps MUST be done in series.
        """
        for document_index, document in enumerate(self.corpus.documents):
            current_topic_index = self.document_topic_assignments[document_index]
            self.number_of_documents_in_each_topic[current_topic_index] -= 1
            self._unassign_document_from_topic(document_index, current_topic_index)

            self._update_topic_weights_for_document(document_index)

            random_number = np.random.random()
            cumulative_weights = self.topic_weights.cumsum()
            new_topic_index = sample_from_cumulative_weights(cumulative_weights, random_number)

            self.number_of_documents_in_each_topic[new_topic_index] += 1
            self._assign_document_to_topic(document_index, new_topic_index)
            self.document_topic_assignments[document_index] = new_topic_index

    def _update_topic_weights_for_document(self, document_index):
        """Update the topic weights for a particular document."""
        document = self.corpus.documents[document_index]
        self.topic_weights = self.number_of_documents_in_each_topic + self.alpha
        occurrence_to_index_count_for_document = self.corpus.occurrence_to_index_count[document_index]

        numerators = (self.number_of_each_word_in_each_topic.take(document, axis=1) + self.beta +
                      occurrence_to_index_count_for_document - 1)

        range_mask = np.arange(len(document)).repeat(self.number_of_topics).reshape(
            len(document), self.number_of_topics)
        denominators = range_mask + self.number_of_total_words_in_each_topic + self.corpus.vocab.size * self.beta

        fractions = numerators / denominators.T
        self.topic_weights *= fractions.prod(axis=1)

    def _assign_document_to_topic(self, document_index, topic_index):
        """Assign a document to a topic."""
        document = self.corpus.documents[document_index]
        self.number_of_total_words_in_each_topic[topic_index] += len(document)
        self.number_of_each_word_in_each_topic[topic_index] += self.corpus.word_counts_in_documents[document_index]

    def _unassign_document_from_topic(self, document_index, topic_index):
        """Un-assign a document from a topic."""
        document = self.corpus.documents[document_index]
        self.number_of_total_words_in_each_topic[topic_index] -= len(document)
        self.number_of_each_word_in_each_topic[topic_index] -= self.corpus.word_counts_in_documents[document_index]
