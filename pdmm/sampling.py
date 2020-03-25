import random


class GibbsSamplingDMM(object):

    def __init__(self, parameters):
        super(GibbsSamplingDMM, self).__init__()
        self.corpus = parameters.corpus
        self.output = parameters.output
        self.number_of_topics = int(parameters.ntopics)
        self.alpha = float(parameters.alpha)
        self.beta = float(parameters.beta)
        self.number_of_iterations = int(parameters.niters)
        self.number_of_top_words = int(parameters.twords)
        self.name = parameters.name

        self.number_of_documents = 0
        self.number_of_words_in_corpus = 0
        self.word_to_id = {}
        self.id_to_word = {}
        self.documents = []
        self.occurrence_to_index_count = []
        self.topic_assignments = []
        self.document_topic_count = []
        self.topic_word_count = []
        self.sum_topic_word_count = []
        self.multi_pros = []
        self.beta_sum = 0.

    def analyse_corpus(self):
        index_word = 0
        data = open(self.corpus, 'r')
        for doc in data:
            document = []
            word_occurrence_to_index_in_doc_count = {}
            word_occurrence_to_index_in_doc = []
            if doc.rstrip is not None:
                words = doc.rstrip().split()
                for word in words:

                    if word in self.word_to_id:
                        document.append(self.word_to_id[word])
                    else:
                        self.word_to_id[word] = index_word
                        self.id_to_word[index_word] = word
                        document.append(index_word)
                        index_word += 1

                    if word in word_occurrence_to_index_in_doc_count:
                        word_occurrence_to_index_in_doc_count[word] += 1
                    else:
                        word_occurrence_to_index_in_doc_count[word] = 1

                    word_occurrence_to_index_in_doc.append(word_occurrence_to_index_in_doc_count[word])

                self.number_of_words_in_corpus += len(document)
                self.number_of_documents += 1
                self.documents.append(document)
                self.occurrence_to_index_count.append(word_occurrence_to_index_in_doc)

        self.beta_sum = len(self.word_to_id) * self.beta

    def topic_assignment_initialise(self):
        self.document_topic_count = [0 for __ in range(self.number_of_topics)]
        self.sum_topic_word_count = [0 for __ in range(self.number_of_topics)]

        for __ in range(self.number_of_topics):
            self.topic_word_count.append([0 for __ in range(len(self.word_to_id))])

        for document_index in range(self.number_of_documents):
            topic = random.randint(0, self.number_of_topics - 1)
            self.document_topic_count[topic] += 1

            for word_index in range(len(self.documents[document_index])):
                self.topic_word_count[topic][self.documents[document_index][word_index]] += 1
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
        for document_index in range(self.number_of_documents):
            topic = self.topic_assignments[document_index]
            self.document_topic_count[topic] -= 1
            doc_size = len(self.documents[document_index])
            document = self.documents[document_index]

            for word_index in range(doc_size):
                word = document[word_index]
                self.topic_word_count[topic][word] -= 1
                self.sum_topic_word_count[topic] -= 1

            for topic_index in range(self.number_of_topics):
                self.multi_pros[topic_index] = self.document_topic_count[topic_index] + self.alpha

                for word_index in range(doc_size):
                    word = document[word_index]
                    self.multi_pros[topic_index] *= (self.topic_word_count[topic_index][word] + self.beta +
                                                     self.occurrence_to_index_count[document_index][
                        word_index] - 1) / (self.sum_topic_word_count[topic_index] + word_index + self.beta_sum)

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
            for document_index in range(self.number_of_documents):
                line = str(self.topic_assignments[document_index]) + "\n"
                wf.write(line)

    def write_top_topical_words(self):
        with open(self.output + self.name + ".topWords", "w") as wf:
            for topic_index in range(self.number_of_topics):
                word_count = {w: self.topic_word_count[topic_index][w] for w in range(len(self.word_to_id))}

                count = 0
                string = "Topic " + str(topic_index) + ": "

                for index in sorted(word_count, key=word_count.get, reverse=True):
                    string += self.id_to_word[index] + " "
                    count += 1
                    if count >= self.number_of_top_words:
                        wf.write(string + "\n")
                        # print string
                        break
