# -*- coding: utf-8 -*-

"""Top-level package for pycox."""

__author__ = """Haavard Kvamme"""
__email__ = 'haavard.kvamme@gmail.com'
__version__ = '0.1.0'

import pycox.datasets
import pycox.evaluation
import pycox.preprocessing
import pycox.simulation

_has_torch = False
try:
    import torch
    _has_torch = True
except:
    pass
if _has_torch:
    import pycox.models
