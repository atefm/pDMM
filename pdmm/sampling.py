"""
Contains the GibbsSamplingDMM class.
"""
import logging
import random


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
    number_of_iterations : int
        The number of iterations of inference.
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
    def __init__(self, corpus, number_of_topics=20, alpha=0.1, beta=0.001, number_of_iterations=2000):
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
        number_of_iterations : int, defaults to 2000
            The number of iterations of inference.
        """
        self.corpus = corpus
        self.number_of_topics = number_of_topics
        self.alpha = alpha
        self.beta = beta
        self.number_of_iterations = number_of_iterations

        self.document_topic_assignments = []
        self.number_of_documents_in_each_topic = []
        self.number_of_each_word_in_each_topic = []
        self.number_of_total_words_in_each_topic = []
        self.topic_weights = []

        self.logger = logging.getLogger(__name__)

    def topic_assignment_initialise(self):
        self.number_of_documents_in_each_topic = [0 for __ in range(self.number_of_topics)]
        self.number_of_total_words_in_each_topic = [0 for __ in range(self.number_of_topics)]

        for __ in range(self.number_of_topics):
            self.number_of_each_word_in_each_topic.append([0 for __ in range(self.corpus.vocab.size)])

        for document_index in range(self.corpus.number_of_documents):
            topic = random.randint(0, self.number_of_topics - 1)
            self.number_of_documents_in_each_topic[topic] += 1

            for word_index in range(len(self.corpus.documents[document_index])):
                self.number_of_each_word_in_each_topic[topic][self.corpus.documents[document_index][word_index]] += 1
                self.number_of_total_words_in_each_topic[topic] += 1

            self.document_topic_assignments.append(topic)

    @staticmethod
    def next_discrete(a):
        b = 0.

        for i in range(len(a)):
            b += a[i]

        r = random.uniform(0., 1.) * b

        b = 0.
        for i in range(len(a)):
            b += a[i]
            if b > r:
                return i
        return len(a) - 1

    def sample_in_single_iteration(self, x):
        print("iteration: " + str(x))
        vocabulary_size = self.corpus.vocab.size
        for document_index in range(self.corpus.number_of_documents):
            topic = self.document_topic_assignments[document_index]
            self.number_of_documents_in_each_topic[topic] -= 1
            doc_size = len(self.corpus.documents[document_index])
            document = self.corpus.documents[document_index]

            for word_index in range(doc_size):
                word = document[word_index]
                self.number_of_each_word_in_each_topic[topic][word] -= 1
                self.number_of_total_words_in_each_topic[topic] -= 1

            for topic_index in range(self.number_of_topics):
                self.topic_weights[topic_index] = self.number_of_documents_in_each_topic[topic_index] + self.alpha

                for word_index in range(doc_size):
                    word = document[word_index]
                    self.topic_weights[topic_index] *= (self.number_of_each_word_in_each_topic[topic_index][word] + self.beta +
                                                        self.corpus.occurrence_to_index_count[document_index][
                        word_index] - 1) / (self.number_of_total_words_in_each_topic[topic_index] + word_index + vocabulary_size * self.beta)

            # print self.multiPros
            topic = self.next_discrete(self.topic_weights)
            # print topic

            self.number_of_documents_in_each_topic[topic] += 1

            for word_index in range(doc_size):
                word = document[word_index]
                self.number_of_each_word_in_each_topic[topic][word] += 1
                self.number_of_total_words_in_each_topic[topic] += 1

            self.document_topic_assignments[document_index] = topic

    def inference(self):
        self.topic_weights = [0 for __ in range(self.number_of_topics)]
        for iteration in range(self.number_of_iterations):
            self.sample_in_single_iteration(iteration)

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

    def get_top_words_for_topic(self, topic_index, number_of_top_words=20):
        """
        Get a list of the top words in a topic.

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
            A list of the top words as strings.
        """
        word_count = {w: self.number_of_each_word_in_each_topic[topic_index][w] for w in range(self.corpus.vocab.size)}
        top_words = []
        sorted_word_ids_iterator = iter(sorted(word_count, key=word_count.get, reverse=True))

        while len(top_words) < number_of_top_words:
            next_word_id = next(sorted_word_ids_iterator)
            next_word = self.corpus.vocab.get_word_from_id(next_word_id)
            top_words.append(next_word)

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
