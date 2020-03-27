"""
Contains the main function for running inference from the command line.
"""
import argparse
import logging
import os
import sys

from .corpus import Corpus
from .sampling import GibbsSamplingDMM


def main(parameters, seed=None):
    """Main function."""
    logger = logging.getLogger(__name__)

    corpus = Corpus.from_document_file(parameters.corpus_path)

    model = GibbsSamplingDMM(
        corpus,
        parameters.number_of_topics,
        parameters.alpha,
        parameters.beta,
    )
    model.randomly_initialise_topic_assignment(seed=seed)
    model.inference(parameters.number_of_iterations)

    if parameters.output_path:
        logger.debug("Writing results to file")
        model.save_top_topical_words_to_file(os.path.join(parameters.output_path, "topWords"))
        model.save_topic_assignments_to_file(os.path.join(parameters.output_path, "topicAssignments"))


def check_arg(args=None):
    parser = argparse.ArgumentParser(description='Topic Modelling')
    parser.add_argument('-c', '--corpus-file', help='Path to corpus file', dest="corpus_path", required=True,
                        metavar="<path>")
    parser.add_argument('-n', '--num-topics', help='Number of topics', dest="number_of_topics", default=20, metavar="<integer>", type=int)
    parser.add_argument('-a', '--alpha', help='Alpha value', default=0.1, metavar="<double>", type=float)
    parser.add_argument('-b', '--beta', help='Beta value', default=0.001, metavar="<double>", type=float)
    parser.add_argument('--output-path', help='Output directory', dest="output_path", default="./output/",
                        metavar="<path>")
    parser.add_argument('--iterations', help='Number of iterations', dest="number_of_iterations", default=2000,
                        metavar="<integer>", type=int)
    parser.add_argument('--num_words', help='Number of most probable topical words', dest="number_of_top_words",
                        default=20, metavar="<integer>", type=int)
    parameters = parser.parse_args(args)
    return parameters


if __name__ == '__main__':
    parsed_parameters = check_arg(sys.argv[1:])
    main(parsed_parameters)
