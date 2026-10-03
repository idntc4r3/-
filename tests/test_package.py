"""Smoke test: verify the package is importable and exposes __version__."""

import nanotorch


def test_version_exists() -> None:
    """nanotorch.__version__ must be a non-empty string (SemVer)."""
    assert isinstance(nanotorch.__version__, str)
    assert nanotorch.__version__ != ""


def test_version_format() -> None:
    """Version string must follow MAJOR.MINOR.PATCH format."""
    parts = nanotorch.__version__.split(".")
    assert len(parts) == 3, f"Expected 3 version parts, got: {nanotorch.__version__!r}"
    assert all(part.isdigit() for part in parts), (
        f"All version parts must be digits, got: {nanotorch.__version__!r}"
    )
