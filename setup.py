#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

requirements = [
    'feather-format>=0.4.0',
    'h5py>=2.9.0',
    'numba>=0.44',
    'scikit-learn>=0.21.2',
    'requests>=2.22.0',
]

setup(
    name='pycox',
    version='0.0.1',
    description="Survival analysis with pytorch",
    author="Haavard Kvamme",
    author_email='haavard.kvamme@gmail.com',
    url='https://github.com/havakv/pycox',
    packages=find_packages(include=['pycox']),
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
        'Programming Language :: Python :: 3.7',
    ],
    python_requires='>=3.6'
)
