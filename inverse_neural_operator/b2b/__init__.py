"""
B2B (Basis-to-Basis) operator models.

This package contains function encoders, forward models (DeepONet, linear, nonlinear),
and related utilities for basis-to-basis operator learning.
"""

from .function_encoder import create_model as create_function_encoder
from .function_encoder import load as load_function_encoder
from .function_encoder import memory_efficient_inner_product

__all__ = [
    "create_function_encoder",
    "load_function_encoder",
    "memory_efficient_inner_product",
]
