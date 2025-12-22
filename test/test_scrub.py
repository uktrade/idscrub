import pandas as pd
from idscrub import IDScrub
from pandas.testing import assert_frame_equal


# Note: These tests will fail if the kernel has not been restarted since the SpaCy model was downloaded.
def test_scrub(scrub_object):
    scrubbed = scrub_object.scrub(scrub_methods=["spacy_persons", "uk_phone_numbers", "uk_postcodes"])
    assert scrubbed == [
        "Our names are [PERSON], [PERSON], and [PERSON].",
        "My number is [PHONENO] and I live at [POSTCODE].",
    ]


def test_scrub_text_id():
    scrub = IDScrub(["Our names are Hamish McDonald, L. Salah, and Elena Suárez."] * 10)

    scrub.scrub(scrub_methods=["spacy_persons"])

    df = scrub.get_scrubbed_data()

    assert df["text_id"].max() == 10
    assert len(df["text_id"]) == 10


def test_scrub_get_scrubbed_data(scrub_object):
    scrub_object.scrub(scrub_methods=["uk_postcodes"])
    df = scrub_object.get_scrubbed_data()

    expected_df = pd.DataFrame(
        {
            "text_id": {0: 2},
            "uk_postcode": {0: ["AA11 1AA"]},
        }
    )

    assert_frame_equal(df, expected_df)


def test_scrub_order(scrub_object):
    scrub_object.scrub(scrub_methods=["uk_postcodes", "uk_phone_numbers", "spacy_persons"])

    assert scrub_object.get_scrubbed_data().columns.to_list() == [
        "text_id",
        "uk_postcode",
        "uk_phone_number",
        "person",
    ]
