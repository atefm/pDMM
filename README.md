# pdmm: Python 3 Implementation for Dirichlet Multinomial Mixture (DMM) Model

This is a Python 3 version of the [original implementation](https://github.com/atefm/pDMM) written by [atefm](https://github.com/atefm). It has a number of improvements, mainly speed and clarity. A full list of changes is available below.

## Description

Applying topic models for short texts (e.g. Tweets) is more challenging because of data sparsity and the limited contexts in such texts. One approach is to combine short texts into long pseudo-documents before training LDA. Another approach is to assume that there is only one topic per document [3]. pDMM provides implementations of the one-topic-per-document Dirichlet Multinomial Mixture (DMM) model (i.e. mixture of unigrams) [1][4].

Bug reports, comments and suggestions about pDMM are highly appreciated. As a free open-source package, pDMM is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

## Usage

From the command line:

```shell script
$ python3 -m pdmm [-h] -c <path> [-n <integer>] [-a <double>] [-b <double>] [--output-path <path>] [--iterations <integer>] [--num-words <integer>]
```

where parameters in [ ] are optional.

`-c, --corpus`: Specify the path to the input corpus file.

`-n, --num-topics`: Specify the number of topics. The default is 20.

`-a, --alpha`: Specify the hyper-parameter alpha. The default value is 0.1.

`-b, --beta`: Specify the hyper-parameter beta. The default value is 0.01, which is a common setting in the literature [5]. Following [6], the users may consider a `beta` value of 0.1 for short texts.

`--output-path`: Specify the output path for the results, which are saved in a folder at the path containing the files `topWords` and `topicAssignments`. If a path is not given, output will not be saved.

`--iterations`: Specify the number of Gibbs sampling iterations. The default value is 2,000.

`--num-words`: Specify the number of top words to be presented and/or saved for each topic. The default value is 20.


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

# Requirements

Python 3.7 is required. All package requirements can be found in `requirements.txt`, but the main dependencies are `numpy` and `numba`.

# References
[1]	  Yin, J. and Wang, J., 2014, August. A dirichlet multinomial mixture model-based approach for short text clustering. In Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining (pp. 233-242). ACM.

[2]   David M. Blei. 2012. Probabilistic Topic Models. Communications of the ACM, 55(4):77–84.

[3]   Dat Quoc Nguyen, Richard Billingsley, Lan Du and Mark Johnson. 2015. [Improving Topic Models with Latent Feature Word Representations](https://tacl2013.cs.columbia.edu/ojs/index.php/tacl/article/view/582/158). Transactions of the Association for Computational Linguistics, vol. 3, pp. 299-313.

[4]   Kamal Nigam, AK McCallum, S Thrun, and T Mitchell. 2000. Text Classification from Labeled and Unlabeled Documents Using EM. Machine learning, 39:103– 134.

[5]   Thomas L. Griffiths and Mark Steyvers. 2004. Finding scientific topics. Proceedings of the National Academy of Sciences of the United States of America, 101(Suppl 1):5228–5235.

[6]   Jianhua Yin and Jianyong Wang. 2014. A Dirichlet Multinomial Mixture Model-based Approach for Short Text Clustering. In Proceedings of the 20th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pages 233–242.

[7]   Christopher D. Manning, Prabhakar Raghavan, and Hinrich Sch¨utze. 2008. Introduction to Information Retrieval. Cambridge University Press.

[8]   Yue Lu, Qiaozhu Mei, and ChengXiang Zhai. 2011. Investigating task performance of probabilistic topic models: an empirical study of PLSA and LDA. Information Retrieval, 14:178–203.
