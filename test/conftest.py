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


@pytest.fixture
def idents():
    return [
        IDScrub.IDEnt(
            text_id="A",
            text="The quick brown fox jumps over the lazy dog.",
            start=10,
            end=19,
            label="animal",
            replacement="[ANIMAL]",
            priority=0.92,
            source="custom_regex",
        ),
        IDScrub.IDEnt(
            text_id="A",
            text="My phone number is 123-456-7890.",
            start=19,
            end=31,
            label="phone_number",
            replacement="[PHONE]",
            priority=0.76,
            source="google",
        ),
        IDScrub.IDEnt(
            text_id="B",
            text="Email me at example@example.com.",
            start=12,
            end=31,
            label="email",
            replacement="[EMAIL]",
            priority=0.88,
            source="email",
        ),
    ]
