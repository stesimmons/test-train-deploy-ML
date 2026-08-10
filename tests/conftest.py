# Shared pytest configuration, fixtures, and test utilities used across
# the test suite. Centralizes common setup logic to reduce duplication
# and ensure consistent testing behavior.

import sys
import os

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)