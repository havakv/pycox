#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

# with open('README.rst') as readme_file:
#     readme = readme_file.read()

# with open('HISTORY.rst') as history_file:
#     history = history_file.read()

requirements = [
    'scipy<=1.2.1,>=1.0' # needed <=1.2.1 for lifelines
    'lifelines>=0.21.3',
    'feather-format>=0.4.0',
    'h5py>=2.9.0',
    'numba>=0.44',
    'scikit-learn>=0.21.2',
    'requests>=2.22.0',
]

# setup_requirements = [
#     'pytest-runner',
#     # TODO(havakv): Put setup requirements (distutils extensions, etc.) here
# ]

# test_requirements = [
    # 'pytest',
    # TODO: Put package test requirements here
# ]

setup(
    name='pycox',
    version='0.0.1',
    description="Survival analysis with pytorch",
    #long_description=readme + '\n\n' + history,
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
    # test_suite='tests',
    # tests_require=test_requirements,
    # setup_requires=setup_requirements,
)
