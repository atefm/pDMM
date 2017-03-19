# pDMM: Python implemetation for Dirichlet Multinomial Mixture (DMM) model


Applying topic models for short texts (e.g. Tweets) is more challenging because of data sparsity and the limited contexts in such texts. One approach is to combine short texts into long pseudo-documents before training LDA. Another approach is to assume that there is only one topic per document [3]. pDMM provides implementations of the one-topic-per-document Dirichlet Multinomial Mixture (DMM) model (i.e. mixture of unigrams) [1][4].

Bug reports, comments and suggestions about pDMM are highly appreciated. As a free open-source package, pDMM is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# Usage

	$ python pDMM.py [-h] --corpus <path> [--output <path>] [--ntopics <integer>] [--alpha <double>] [--beta <double>] [--niters <intege>] [--twords <integer>] [--name <string>]


where parameters in [ ] are optional.


`--corpus`: Specify the path to the input corpus file.

`--output`: Specify the path where output must be stored.

`--ntopics <int>`: Specify the number of topics. The default value is 20.

`--alpha <double>`: Specify the hyper-parameter `alpha`. The default value is 0.1.

`--beta <double>`: Specify the hyper-parameter `beta`. The default value is 0.01 which is a common setting in the literature [5]. Following [6], the users may consider to the `beta` value of 0.1 for short texts.

`--niters <int>`: Specify the number of Gibbs sampling iterations. The default value is 2000.

`--twords <int>`: Specify the number of the most probable topical words. The default value is 20.

`--name <String>`: Specify a name to the topic modeling experiment. The default value is `model`.


**Examples:**

	$ python pDMM.py --corpus sample_data -name test

The output files are by default saved in output directory. We have output files of `test.topWords` and `test.topicAssignments`, referring to the top topical words and topic assignments respectively.

# Requirements
Python 2.7


# References
[1]	  Yin, J. and Wang, J., 2014, August. A dirichlet multinomial mixture model-based approach for short text clustering. In Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining (pp. 233-242). ACM.

[2]   David M. Blei. 2012. Probabilistic Topic Models. Communications of the ACM, 55(4):77–84.

[3]   Dat Quoc Nguyen, Richard Billingsley, Lan Du and Mark Johnson. 2015. [Improving Topic Models with Latent Feature Word Representations](https://tacl2013.cs.columbia.edu/ojs/index.php/tacl/article/view/582/158). Transactions of the Association for Computational Linguistics, vol. 3, pp. 299-313.

[4]   Kamal Nigam, AK McCallum, S Thrun, and T Mitchell. 2000. Text Classification from Labeled and Unlabeled Documents Using EM. Machine learning, 39:103– 134.

[5]   Thomas L. Griffiths and Mark Steyvers. 2004. Finding scientific topics. Proceedings of the National Academy of Sciences of the United States of America, 101(Suppl 1):5228–5235.

[6]   Jianhua Yin and Jianyong Wang. 2014. A Dirichlet Multinomial Mixture Model-based Approach for Short Text Clustering. In Proceedings of the 20th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pages 233–242.

[7]   Christopher D. Manning, Prabhakar Raghavan, and Hinrich Sch¨utze. 2008. Introduction to Information Retrieval. Cambridge University Press.

[8]   Yue Lu, Qiaozhu Mei, and ChengXiang Zhai. 2011. Investigating task performance of probabilistic topic models: an empirical study of PLSA and LDA. Information Retrieval, 14:178–203.
