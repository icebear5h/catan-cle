import pytest

from .support import SetupPair, setup_pair


@pytest.fixture
def pair() -> SetupPair:
    return setup_pair()
