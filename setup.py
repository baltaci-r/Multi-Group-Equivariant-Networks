# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

requires = [
    'pandas',
    'numpy',
    'tqdm',
    'matplotlib',
    'seaborn',
    'nltk',
    'clip',
    'torch',
    'torchvision',
    'pytorch_lightning',
    'tensorboardX',
    'transformers',
    'textblob',
    'vaderSentiment',
    'seqeval',
]

setup(
    name='multi-equi',
    description='implementation of multi-equi paper code',
    author='baltaji-basu',
    url='https://github.com/baltaci-r/multimodal-equivariant-nets',
    packages=find_packages(),
    install_requires=requires,
    tests_require=requires,
)