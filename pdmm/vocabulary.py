"""
Contains the Vocabulary class.
"""


class Vocabulary:
    """
    Represents a corpus vocabulary.
    """
    def __init__(self):
        self._word_to_id = {}
        self._id_to_word = {}
        self._count = 0

    def __eq__(self, other):
        if not type(other) == type(self):
            raise TypeError("Can only compare to {} type.".format(type(self).__name__))

        if self.size != other.size:
            return False

        for index in range(self._count):
            if self.get_word_from_id(index) != other.get_word_from_id(index):
                return False
        else:
            return True

    @property
    def size(self):
        """Get the size of the vocabulary."""
        return self._count

    def get_id_from_word(self, word):
        """Get a vocabulary id from a word."""
        try:
            word_id = self._word_to_id[word]
        except KeyError:
            word_id = self._add_new_word_and_return_id(word)
        return word_id

    def get_word_from_id(self, word_id):
        """Get a word from its vocabulary id."""
        return self._id_to_word[word_id]

    def save_to_file(self, file_path):
        """Save the vocabulary to a file."""
        with open(file_path, "w") as wf:
            for index in range(self._count):
                word = self._id_to_word[index]
                line = word + "\n"
                wf.write(line)

    @classmethod
    def from_list_of_words(cls, list_of_words):
        """Create a vocabulary from a list of words."""
        vocab = cls()
        for word in list_of_words:
            vocab._add_new_word_and_return_id(word)

        return vocab

    @classmethod
    def load_from_file(cls, file_path):
        """Load an instance from a vocabulary file."""
        vocab = cls()

        with open(file_path, "r") as rf:
            for line in rf.readlines():
                word = line.rstrip()
                vocab._add_new_word_and_return_id(word)

        return vocab

    def _add_new_word_and_return_id(self, word):
        """Add a new word to the vocabulary, and return the id."""
        new_word_id = self._count
        self._word_to_id[word] = new_word_id
        self._id_to_word[new_word_id] = word
        self._count += 1
        return new_word_id
