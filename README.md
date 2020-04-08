# pdmm: Python 3 Implementation for Dirichlet Multinomial Mixture (DMM) Model

This is a Python 3 version of the [original implementation](https://github.com/atefm/pDMM) written by [atefm](https://github.com/atefm). It has a number of improvements, mainly speed and clarity. A full list of changes is available below.

## Description

Applying topic models for short texts (e.g. Tweets) is more challenging because of data sparsity and the limited contexts in such texts. One approach is to combine short texts into long pseudo-documents before training LDA. Another approach is to assume that there is only one topic per document [[3]](#references). pDMM provides implementations of the one-topic-per-document Dirichlet Multinomial Mixture (DMM) model (i.e. mixture of unigrams) [[1]](#references)[[4]](#references). For further reading, see Manning [[6]](#references) and Lu [[7]](#references).

Bug reports, comments and suggestions about pDMM are highly appreciated. As a free open-source package, pDMM is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

## Installation

`pdmm` can be run without installation from within the repository directory, but the package can be installed locally with `pip`:

```shell script
$ pip install setup.py
```

## Usage

From the command line:

```shell script
$ python3 -m pdmm [-h] -c <path> [-n <integer>] [-a <double>] [-b <double>] [--output-path <path>] [--iterations <integer>] [--num-words <integer>]
```

where parameters in [ ] are optional.

`-c, --corpus` Specify the path to the input corpus file.

`-n, --num-topics` Specify the number of topics. The default is 20.

`-a, --alpha` Specify the hyper-parameter alpha. The default value is 0.1.

`-b, --beta` Specify the hyper-parameter beta. The default value is 0.01, which is a common setting in the literature [[5]](#references). Following [[1]](#references), the users may consider a `beta` value of 0.1 for short texts.

`--output-path` Specify the output path for the results, which are saved in a folder at the path containing the files `topWords` and `topicAssignments`. If a path is not given, output will not be saved.

`--iterations` Specify the number of Gibbs sampling iterations. The default value is 2,000.

`--num-words` Specify the number of top words to be presented and/or saved for each topic. The default value is 20.


Consider the following example:

```shell script
$ python3 -m pdmm --corpus-file tests/data/sample_data  --iterations 100
```

From the interpreter:

```python
import pdmm
corpus = pdmm.Corpus.from_document_file("/path/to/corpus/file")
model = pdmm.GibbsSamplingDMM(corpus)
model.randomly_initialise_topic_assignment()
model.inference(number_of_iterations=100)
```

## Tests

Tests can be run from the command line:

```shell script
$ python3 -m tests
```

The tests can be slow as they are insuring that the inference is producing the expected results. Alternatively, a test module can be run individually:

```shell script
$ python3 -m tests corpus
```

This will attempt to run the tests in the file `tests/test_corpus.py`.

## Requirements

Python 3.7 is required. All package requirements can be found in `requirements.txt`, but the main dependencies are `numpy` and `numba`.

## Changes from Original Implementation

All changes can be tracked on Github, but the broad changes are as follows:

* The addition of tests to ensure that the algorithm was not affected during refactoring.
* Creation of a Python module rather than standalone scripts, allowing code to be run properly within the interpreter.
* Cleaner code and PEP8 compliance.
* Code rewritten to use NumPy arrays and run more steps in parallel. This led to a massive subsequent speed-up.
* Code to generate new documents after inference has been completed.

## References
[1] Jianhua Yin and Jianyong Wang, 2014, August. [A Dirichlet Multinomial Mixture Model-based Approach for Short Text Clustering](https://dl.acm.org/doi/10.1145/2623330.2623715). In Proceedings of the 20th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (pp. 233-242). ACM.

[2] David M. Blei. 2012. [Probabilistic Topic Models](https://dl.acm.org/doi/10.1145/2133806.2133826). Communications of the ACM, 55(4):77–84.

[3] Dat Quoc Nguyen, Richard Billingsley, Lan Du and Mark Johnson. 2015. [Improving Topic Models with Latent Feature Word Representations](https://tacl2013.cs.columbia.edu/ojs/index.php/tacl/article/view/582/158). Transactions of the Association for Computational Linguistics, vol. 3, pp. 299-313.

[4] Kamal Nigam, AK McCallum, S Thrun, and T Mitchell. 2000. [Text Classification from Labeled and Unlabeled Documents using EM. Machine Learning](https://link.springer.com/article/10.1023/A:1007692713085), 39:103– 134.

[5] Thomas L. Griffiths and Mark Steyvers. 2004. [Finding Scientific Topics](https://www.pnas.org/content/101/suppl_1/5228). Proceedings of the National Academy of Sciences of the United States of America, 101(Suppl 1):5228–5235.

[6] Christopher D. Manning, Prabhakar Raghavan, and Hinrich Schutze. 2008. [Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/information-retrieval-book.html). Cambridge University Press.

[7] Yue Lu, Qiaozhu Mei, and ChengXiang Zhai. 2011. [Investigating Task Performance of Probabilistic Topic Models: an Empirical Study of PLSA and LDA](https://link.springer.com/article/10.1007%2Fs10791-010-9141-9). Information Retrieval, 14:178–203.
