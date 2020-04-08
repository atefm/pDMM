"""
Setup file for pdmm.
"""
import setuptools

import pdmm


with open("README.md", "r") as rf:
    long_description = rf.read()


setuptools.setup(
    name=pdmm.__name__,
    version=pdmm.__version__,
    description="A Python 3 implementation for a Dirichlet multinomial mixture model.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gchqdev227/pDMM",
    packages=[pdmm.__name__],
    install_requires=["coverage", "numba", "numpy"],
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3.7"
    ]
)
