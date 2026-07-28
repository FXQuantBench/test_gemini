"""conftest.py — add repo root to sys.path so tests can import top-level modules."""
import pathlib
import sys
from unittest.mock import MagicMock

ROOT = pathlib.Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Stub heavy optional dependencies so unit tests run without installing them.
# vectorbt imports numba / pandas-ta / etc. which are CI-unfriendly; we mock
# the whole module here and patch the object where needed in individual tests.
if "vectorbt" not in sys.modules:
    sys.modules["vectorbt"] = MagicMock()
