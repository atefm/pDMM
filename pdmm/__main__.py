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


def parse_args(args=None):
    parser = argparse.ArgumentParser(prog="pdmm", description="Python Dirichlet Mixture Model Implementation")
    
    parser.add_argument("-c", "--corpus-file",
                        dest="corpus_path", metavar="<path>",
                        help="Path to corpus file",
                        required=True)
                        
    parser.add_argument("-n", "--num-topics",
                        dest="number_of_topics", metavar="<integer>",
                        default=20, type=int,
                        help="Number of topics")
                        
    parser.add_argument("-a", "--alpha",
                        metavar="<double>",
                        default=0.1, type=float,
                        help="Alpha value")

    parser.add_argument("-b", "--beta",
                        metavar="<double>",
                        default=0.001, type=float,
                        help="Beta value")

    parser.add_argument("--output-path",
                        dest="output_path", metavar="<path>",
                        help="Output directory")

    parser.add_argument("--iterations",
                        dest="number_of_iterations", metavar="<integer>",
                        default=2000, type=int,
                        help="Number of iterations")

    parser.add_argument("--num_words",
                        dest="number_of_top_words", metavar="<integer>",
                        default=20,  type=int,
                        help="Number of most probable topical words")

    parameters = parser.parse_args(args)
    return parameters


if __name__ == '__main__':
    parsed_parameters = parse_args(sys.argv[1:])
    main(parsed_parameters)
