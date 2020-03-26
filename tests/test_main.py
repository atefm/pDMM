"""
Tests for the command line.
"""
import os
import random
import tempfile
import unittest

import pdmm.sampling
from pdmm.sampling import GibbsSamplingDMM
from pdmm.__main__ import check_arg

from .utils import read_contents_from_path, python_2_randint


class CommandLineTests(unittest.TestCase):

    def test_overall_output(self):
        """Test that the output is correct."""
        random.seed(1)
        pdmm.sampling.random.randint = python_2_randint
        temp_dir = tempfile.gettempdir() + "/"
        arg_string = "--corpus {} --output {} --niters {}".format(
            "tests/data/sample_data",
            temp_dir,
            50,
        )
        arguments_components = arg_string.split()
        parsed_args = check_arg(arguments_components)
        model = GibbsSamplingDMM(
            "tests/data/sample_data",
            temp_dir,
            20,
            0.1,
            0.001,
            50,
            20,
            "model"
        )
        model.analyse_corpus()
        model.topic_assignment_initialise()
        model.inference()

        model.save_top_topical_words_to_file(parsed_args.output + parsed_args.name + ".topWords")
        model.save_topic_assignments_to_file(parsed_args.output + parsed_args.name + ".topicAssignments")

        expected_top_words = read_contents_from_path("tests/data/topWords")
        expected_topic_assignments = read_contents_from_path("tests/data/topicAssignments")

        top_words = read_contents_from_path(os.path.join(temp_dir, "model.topWords"))
        topic_assignments = read_contents_from_path(os.path.join(temp_dir, "model.topicAssignments"))

        self.assertEqual(top_words, expected_top_words, "Top words should be equal.")
        self.assertEqual(topic_assignments, expected_topic_assignments, "Topic assignments should be equal.")
