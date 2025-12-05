import pandas as pd
from idscrub import IDScrub
from pandas.testing import assert_frame_equal


# Note: These tests will fail if the kernel has not been restarted since the SpaCy model was downloaded.
def test_persidio():
    scrub = IDScrub(
        ["Our names are Hamish McDonald, L. Salah, and Elena Suárez.", "My IBAN code is GB91BKEN10000041610008."]
    )
    scrubbed_texts = scrub.presidio(entities_to_scrub=["PERSON", "IBAN_CODE"])

    assert scrubbed_texts == ["Our names are [PERSON], [PERSON], and [PERSON].", "My IBAN code is [IBAN_CODE]."]


def test_persidio_map():
    scrub = IDScrub(
        ["Our names are Hamish McDonald, L. Salah, and Elena Suárez.", "My IBAN code is GB91BKEN10000041610008."]
    )
    scrubbed_texts = scrub.presidio(
        entities_to_scrub=["PERSON", "IBAN_CODE"], replacement_map={"PERSON": "[PHELLO]", "IBAN_CODE": "[IHELLO]"}
    )

    assert scrubbed_texts == ["Our names are [PHELLO], [PHELLO], and [PHELLO].", "My IBAN code is [IHELLO]."]


def test_persidio_get_data():
    scrub = IDScrub(
        ["Our names are Hamish McDonald, L. Salah, and Elena Suárez.", "My IBAN code is GB91BKEN10000041610008."]
    )

    scrub.presidio(entities_to_scrub=["PERSON", "IBAN_CODE"])

    df = scrub.get_scrubbed_data()

    expected_df = pd.DataFrame(
        {
            "text_id": {0: 1, 1: 2},
            "scrubbed_presidio_person": {0: ["Hamish McDonald", "L. Salah", "Elena Suárez"], 1: None},
            "scrubbed_presidio_iban_code": {0: None, 1: ["GB91BKEN10000041610008"]},
        }
    )

    assert_frame_equal(df, expected_df)
