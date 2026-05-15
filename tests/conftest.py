"""
tests/conftest.py — pytest fixtures and test-session setup.

Injects a stub for pandas_datareader into sys.modules before any test
module imports it (or imports data.fred which imports it at module level).
This allows tests to run on Python environments where the installed
pandas_datareader is incompatible (e.g. Python 3.14 with the 0.22 release).

The stub exposes a MagicMock for DataReader; individual tests override it
with patch("data.fred.web.DataReader") as needed.
"""
import sys
import types
from unittest.mock import MagicMock

import pandas as pd

# Only inject the stub when the real package fails to import cleanly.
_needs_stub = False
try:
    import pandas_datareader.data as _real_web  # noqa: F401
except Exception:
    _needs_stub = True

if _needs_stub and "pandas_datareader" not in sys.modules:
    _pdr = types.ModuleType("pandas_datareader")
    _pdr_data = types.ModuleType("pandas_datareader.data")
    _pdr_data.DataReader = MagicMock(return_value=pd.DataFrame())
    _pdr.data = _pdr_data
    sys.modules["pandas_datareader"] = _pdr
    sys.modules["pandas_datareader.data"] = _pdr_data
elif _needs_stub:
    # Package already in sys.modules but broken — patch DataReader on it
    import pandas_datareader.data as _pdr_data_mod
    if not hasattr(_pdr_data_mod, "DataReader"):
        _pdr_data_mod.DataReader = MagicMock(return_value=pd.DataFrame())
