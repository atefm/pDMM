import argparse
import os
import sys

from .sampling import GibbsSamplingDMM


def is_valid_file(parser, data_file):
    if not os.path.exists(data_file):
        parser.error("The file \"%s\" does not exist!" % data_file)
    else:
        return data_file


def check_arg(args=None):
    parser = argparse.ArgumentParser(description='Topic Modelling')
    parser.add_argument('-i', '--corpus', help='Path to corpus file', required=True, metavar="<path>",
                        type=lambda corpus_file: is_valid_file(parser, corpus_file))
    parser.add_argument('-o', '--output', help='Output directory', default="./output/", metavar="<path>")
    parser.add_argument('-nt', '--ntopics', help='Number of topics', default=20, metavar="<integer>")
    parser.add_argument('-a', '--alpha', help='Alpha value', default=0.1, metavar="<double>")
    parser.add_argument('-b', '--beta', help='Beta value', default=0.001, metavar="<double>")
    parser.add_argument('-ni', '--niters', help='Number of iterations', default=2000, metavar="<intege>")
    parser.add_argument('-t', '--twords', help='Number of most probable topical words', default=20, metavar="<integer>")
    parser.add_argument('-e', '--name', help='Name of the experiment', default="model", metavar="<string>")
    parameters = parser.parse_args(args)
    return parameters


if __name__ == '__main__':
    args = check_arg(sys.argv[1:])
    model = GibbsSamplingDMM(
        args.corpus,
        args.output,
        args.ntopics,
        args.alpha,
        args.beta,
        args.niters,
        args.twords,
        args.name
    )
    model.analyse_corpus()
    model.topic_assignment_initialise()
    model.inference()

    print("Writing Results")
    model.save_top_topical_words_to_file(args.output + args.name + ".topWords")
    model.save_topic_assignments_to_file(args.output + args.name + ".topicAssignments")
