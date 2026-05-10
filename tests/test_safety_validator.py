"""Tests for the AST-based Python safety validator."""
import pytest

from research_agent.tools.python_exec import PythonSafetyError, validate_python_safety


# --- code that should pass ---

VALID_SIMPLE = """
import pandas as pd
import numpy as np

def generate_signals(df):
    close = df["Close"].astype(float)
    fast = close.rolling(10).mean()
    slow = close.rolling(30).mean()
    signal = pd.Series(0.0, index=df.index)
    signal[fast > slow] = 1.0
    signal[fast < slow] = -1.0
    return signal.fillna(0.0)
"""

VALID_NUMPY_ONLY = """
import numpy as np

def generate_signals(df):
    return np.zeros(len(df))
"""


def test_valid_simple_strategy_passes():
    validate_python_safety(VALID_SIMPLE)


def test_valid_numpy_only_passes():
    validate_python_safety(VALID_NUMPY_ONLY)


# --- blocked imports ---

@pytest.mark.parametrize("blocked", [
    "import os",
    "import sys",
    "import subprocess",
    "import socket",
    "import urllib",
    "import requests",
    "import pathlib",
    "import importlib",
    "import ctypes",
    "import shutil",
    "import http",
])
def test_blocked_import_raises(blocked):
    with pytest.raises(PythonSafetyError, match="Blocked import"):
        validate_python_safety(blocked)


def test_blocked_from_import_raises():
    with pytest.raises(PythonSafetyError, match="Blocked import"):
        validate_python_safety("from os import path")


# --- blocked calls ---

@pytest.mark.parametrize("blocked_call", [
    "eval('1+1')",
    "exec('pass')",
    "open('file.txt')",
    "__import__('os')",
    "compile('', '', 'exec')",
    "globals()",
    "locals()",
    "getattr(object, 'attr')",
    "setattr(object, 'x', 1)",
    "input('prompt')",
])
def test_blocked_call_raises(blocked_call):
    with pytest.raises(PythonSafetyError):
        validate_python_safety(blocked_call)


# --- dunder access ---

def test_dunder_attribute_access_raises():
    with pytest.raises(PythonSafetyError):
        validate_python_safety("x = df.__class__")


def test_dunder_name_raises():
    with pytest.raises(PythonSafetyError):
        validate_python_safety("x = __name__")


# --- syntax errors ---

def test_syntax_error_raises():
    with pytest.raises(PythonSafetyError, match="Syntax error"):
        validate_python_safety("def foo(: pass")
