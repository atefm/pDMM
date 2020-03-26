import argparse
import os
import sys

from .sampling import GibbsSamplingDMM


def check_arg(args=None):
    parser = argparse.ArgumentParser(description='Topic Modelling')
    parser.add_argument('-c', '--corpus-file', help='Path to corpus file', dest="corpus_path", required=True, metavar="<path>")
    parser.add_argument('-o', '--output-path', help='Output directory', dest="output_path", default="./output/", metavar="<path>")
    parser.add_argument('-n', '--num-topics', help='Number of topics', dest="number_of_topics", default=20, metavar="<integer>")
    parser.add_argument('-a', '--alpha', help='Alpha value', default=0.1, metavar="<double>")
    parser.add_argument('-b', '--beta', help='Beta value', default=0.001, metavar="<double>")
    parser.add_argument('-ni', '--iterations', help='Number of iterations', dest="number_of_iterations", default=2000, metavar="<integer>")
    parser.add_argument('-t', '--num_words', help='Number of most probable topical words', dest="number_of_top_words", default=20, metavar="<integer>")
    parser.add_argument('-e', '--name', help='Name of the experiment', default="model", metavar="<string>")
    parameters = parser.parse_args(args)
    return parameters


if __name__ == '__main__':
    args = check_arg(sys.argv[1:])
    model = GibbsSamplingDMM(
        args.corpus_path,
        args.output_path,
        args.number_of_topics,
        args.alpha,
        args.beta,
        args.number_of_iterations,
        args.number_of_top_words,
        args.name
    )
    model.analyse_corpus()
    model.topic_assignment_initialise()
    model.inference()

    print("Writing Results")
    model.save_top_topical_words_to_file(args.output + args.name + ".topWords")
    model.save_topic_assignments_to_file(args.output + args.name + ".topicAssignments")
