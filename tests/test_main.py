"""
Tests for the command line.
"""
import os
import random
import tempfile
import unittest

from pdmm.sampling import GibbsSamplingDMM
from pdmm.__main__ import check_arg

from .utils import read_contents_from_path


class CommandLineTests(unittest.TestCase):

    def test_overall_output(self):
        """Test that the output is correct."""
        random.seed(1)
        temp_dir = tempfile.gettempdir() + "/"
        arg_string = "--corpus {} --output {} --niters {}".format(
            "tests/data/sample_data",
            temp_dir,
            50,
        )
        arguments_components = arg_string.split()
        model = GibbsSamplingDMM(check_arg(arguments_components))
        model.analyse_corpus()
        model.topic_assignment_initialise()
        model.inference()

        model.write_top_topical_words()
        model.write_topic_assignments()

        expected_top_words = read_contents_from_path("tests/data/topWords")
        expected_topic_assignments = read_contents_from_path("tests/data/topicAssignments")

        top_words = read_contents_from_path(os.path.join(temp_dir, "model.topWords"))
        topic_assignments = read_contents_from_path(os.path.join(temp_dir, "model.topicAssignments"))

        self.assertEqual(top_words, expected_top_words, "Top words should be equal.")
        self.assertEqual(topic_assignments, expected_topic_assignments, "Topic assignments should be equal.")
