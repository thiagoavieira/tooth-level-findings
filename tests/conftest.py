from pathlib import Path

import pytest

from toothfindings.config import load_paths

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def paths():
    """Configuration pointing at the repository's results/ (no restricted data needed)."""
    return load_paths()


@pytest.fixture(scope="session")
def sources(paths):
    """Recomputed table inputs, shared by the regression tests."""
    from toothfindings.reporting.sources import Sources

    return Sources(paths)
