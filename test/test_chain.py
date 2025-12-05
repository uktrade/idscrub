import pandas as pd
from pandas.testing import assert_frame_equal


# Note: These tests will fail if the kernel has not been restarted since the SpaCy model was downloaded.
def test_chain(scrub_object):
    scrub_object.uk_phone_numbers()
    scrub_object.uk_postcodes()
    scrubbed = scrub_object.spacy_persons()

    assert scrubbed == [
        "Our names are [PERSON], [PERSON], and [PERSON].",
        "My number is [PHONENO] and I live at [POSTCODE].",
    ]


def test_chain_order(scrub_object):
    scrubbed = scrub_object.uk_phone_numbers()

    assert scrubbed == [
        "Our names are Hamish McDonald, L. Salah, and Elena Suárez.",
        "My number is [PHONENO] and I live at AA11 1AA.",
    ]

    assert scrub_object.get_scrubbed_data()["scrubbed_uk_phone_numbers"].to_list() == [["+441111111111"]]
    assert "scrubbed_uk_postcodes" not in scrub_object.get_scrubbed_data().columns

    scrubbed = scrub_object.uk_postcodes()

    assert scrubbed == [
        "Our names are Hamish McDonald, L. Salah, and Elena Suárez.",
        "My number is [PHONENO] and I live at [POSTCODE].",
    ]
    assert scrub_object.get_scrubbed_data()["scrubbed_uk_phone_numbers"].to_list() == [["+441111111111"]]
    assert scrub_object.get_scrubbed_data()["scrubbed_uk_postcodes"].to_list() == [["AA11 1AA"]]


def test_get_scrubbed_data_chain(scrub_object):
    scrub_object.uk_phone_numbers()
    scrub_object.uk_postcodes()
    scrub_object.spacy_persons()

    df = scrub_object.get_scrubbed_data()

    expected_df = pd.DataFrame(
        {
            "text_id": {0: 1, 1: 2},
            "scrubbed_uk_phone_numbers": {0: None, 1: ["+441111111111"]},
            "scrubbed_uk_postcodes": {0: None, 1: ["AA11 1AA"]},
            "scrubbed_spacy_person": {0: ["Hamish McDonald", "L. Salah", "Elena Suárez"], 1: None},
        }
    )

    assert_frame_equal(df, expected_df)
