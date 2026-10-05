"""Test suite root.

conftest.py provides shared fixtures and pytest plug-in configuration
that will grow as new modules are added day by day.
"""

import pytest


# ---------------------------------------------------------------------------
# Markers registration (mirrors pyproject.toml for IDE discoverability)
# ---------------------------------------------------------------------------

def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers to avoid PytestUnknownMarkWarning."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "scalar: scalar engine tests")
    config.addinivalue_line("markers", "tensor: tensor engine tests")
