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


@pytest.fixture
def scrub_object_all():
    return IDScrub(
        [
            "We are Hamish McDonald, L. Salah, and Elena Suárez, Professor Patel, @johnsmith, 8.8.8.8, marie-9999@randomemail.co.uk.",
            "My number is +441111111111 and I live at AA11 1AA.",
        ]
    )
