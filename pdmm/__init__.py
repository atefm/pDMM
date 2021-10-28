"""
Python 3 implementation for Dirichlet multinomial mixture model.

Implementation based on the paper "A Dirichlet Multinomial Mixture Model-based
Approach for Short Text Clustering" by Jianhua Yin and Jianyong Wang, which can
be found at https://dl.acm.org/doi/10.1145/2623330.2623715.
"""
from .corpus import Corpus
from .sampling import GibbsSamplingDMM
from .vocabulary import Vocabulary

__version__ = "2.0.1"
