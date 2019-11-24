#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages


with open('README.md', 'r') as readme_file:
    readme = readme_file.read()

requirements = [
    'feather-format>=0.4.0',
    'h5py>=2.9.0',
    'numba>=0.44',
    'scikit-learn>=0.21.2',
    'requests>=2.22.0',
    'torchtuples>=0.1.2',
]

setup(
    name='pycox',
    version='0.1.1',
    description="Survival analysis with PyTorch",
    long_description=readme,
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
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.6',
    ],
    python_requires='>=3.6'
)
