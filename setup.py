#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages


long_description = """
**pycox** is a python package for survival analysis and time-to-event prediction with [PyTorch](https://pytorch.org/).
It is built on the [torchtuples](https://github.com/havakv/torchtuples) package for training [PyTorch](https://pytorch.org/) models.

Read the documentation at: https://github.com/havakv/pycox

The package contains

- survival models: (Logistic-Hazard, DeepHit, DeepSurv, Cox-Time, MTLR, etc.)
- evaluation criteria (concordance, Brier score, Binomial log-likelihood, etc.)
- event-time datasets (SUPPORT, METABRIC, KKBox, etc)
- simulation studies
- illustrative examples
"""

requirements = [
    'torchtuples>=0.2.0',
    'feather-format>=0.4.0',
    'h5py>=2.9.0',
    'numba>=0.44',
    'scikit-learn>=0.21.2',
    'requests>=2.22.0',
    'py7zr>=0.11.3',
]

setup(
    name='pycox',
    version='0.2.3',
    description="Survival analysis with PyTorch",
    long_description=long_description,
    long_description_content_type='text/markdown',
    author="Haavard Kvamme",
    author_email='haavard.kvamme@gmail.com',
    url='https://github.com/havakv/pycox',
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    license="BSD license",
    zip_safe=False,
    keywords='pycox',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.6',
    ],
    python_requires='>=3.6'
)
