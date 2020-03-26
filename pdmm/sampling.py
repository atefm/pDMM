import random


class GibbsSamplingDMM:

    def __init__(self, corpus, number_of_topics, alpha, beta,
                 number_of_iterations):
        self.corpus = corpus
        self.number_of_topics = number_of_topics
        self.alpha = alpha
        self.beta = beta
        self.number_of_iterations = number_of_iterations

        self.topic_assignments = []
        self.document_topic_count = []
        self.topic_word_count = []
        self.sum_topic_word_count = []
        self.multi_pros = []

    def topic_assignment_initialise(self):
        self.document_topic_count = [0 for __ in range(self.number_of_topics)]
        self.sum_topic_word_count = [0 for __ in range(self.number_of_topics)]

        for __ in range(self.number_of_topics):
            self.topic_word_count.append([0 for __ in range(self.corpus.vocab.size)])

        for document_index in range(self.corpus.number_of_documents):
            topic = random.randint(0, self.number_of_topics - 1)
            self.document_topic_count[topic] += 1

            for word_index in range(len(self.corpus.documents[document_index])):
                self.topic_word_count[topic][self.corpus.documents[document_index][word_index]] += 1
                self.sum_topic_word_count[topic] += 1

            self.topic_assignments.append(topic)

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
            topic = self.topic_assignments[document_index]
            self.document_topic_count[topic] -= 1
            doc_size = len(self.corpus.documents[document_index])
            document = self.corpus.documents[document_index]

            for word_index in range(doc_size):
                word = document[word_index]
                self.topic_word_count[topic][word] -= 1
                self.sum_topic_word_count[topic] -= 1

            for topic_index in range(self.number_of_topics):
                self.multi_pros[topic_index] = self.document_topic_count[topic_index] + self.alpha

                for word_index in range(doc_size):
                    word = document[word_index]
                    self.multi_pros[topic_index] *= (self.topic_word_count[topic_index][word] + self.beta +
                                                     self.corpus.occurrence_to_index_count[document_index][
                        word_index] - 1) / (self.sum_topic_word_count[topic_index] + word_index + vocabulary_size * self.beta)

            # print self.multiPros
            topic = self.next_discrete(self.multi_pros)
            # print topic

            self.document_topic_count[topic] += 1

            for word_index in range(doc_size):
                word = document[word_index]
                self.topic_word_count[topic][word] += 1
                self.sum_topic_word_count[topic] += 1

            self.topic_assignments[document_index] = topic

    def inference(self):
        self.multi_pros = [0 for __ in range(self.number_of_topics)]
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
                line = str(self.topic_assignments[document_index]) + "\n"
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
        word_count = {w: self.topic_word_count[topic_index][w] for w in range(self.corpus.vocab.size)}
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
