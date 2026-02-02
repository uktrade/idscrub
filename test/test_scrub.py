import pandas as pd
from idscrub import IDScrub
from pandas.testing import assert_frame_equal


def test_scrub(scrub_object):
    scrubbed = scrub_object.scrub(
        pipeline=[{"method": "spacy_entities"}, {"method": "uk_phone_numbers"}, {"method": "uk_postcodes"}]
    )
    assert scrubbed == [
        "Our names are [PERSON], [PERSON], and [PERSON].",
        "My number is [PHONENO] and I live at [POSTCODE].",
    ]


def test_scrub_text_id():
    scrub = IDScrub(["Our names are Hamish McDonald, L. Salah, and Elena Suárez."] * 10)

    scrub.scrub(pipeline=[{"method": "spacy_entities"}])

    df = scrub.get_scrubbed_data()

    assert df["text_id"].max() == 10
    assert len(df["text_id"]) == 10


def test_scrub_get_scrubbed_data(scrub_object):
    scrub_object.scrub(pipeline=[{"method": "uk_postcodes"}])
    df = scrub_object.get_scrubbed_data()

    expected_df = pd.DataFrame(
        {
            "text_id": {0: 2},
            "uk_postcode": {0: ["AA11 1AA"]},
        }
    )

    assert_frame_equal(df, expected_df)


def test_scrub_get_all_identified_data(scrub_object):
    scrub_object.scrub(pipeline=[{"method": "uk_postcodes"}])
    df = scrub_object.get_all_identified_data()

    expected_df = pd.DataFrame(
        {
            "text_id": {0: 2},
            "text": {0: "AA11 1AA"},
            "start": {0: 41},
            "end": {0: 49},
            "label": {0: "uk_postcode"},
            "replacement": {0: "[POSTCODE]"},
            "priority": {0: 0.5},
            "source": {0: "regex"},
        }
    )

    assert_frame_equal(df, expected_df)
