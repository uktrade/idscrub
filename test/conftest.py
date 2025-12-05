import pytest
from idscrub import IDScrub


@pytest.fixture
def scrub_object():
    return IDScrub(
        [
            "Our names are Hamish McDonald, L. Salah, and Elena Suárez.",
            "My number is +441111111111 and I live at AA11 1AA.",
        ]
    )
