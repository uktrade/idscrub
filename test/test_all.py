import pandas as pd
from idscrub import IDScrub
from pandas.testing import assert_frame_equal


# Note: These tests will fail if the kernel has not been restarted since the SpaCy model was downloaded.
def test_all(scrub_object):
    scrubbed = scrub_object.all()
    assert scrubbed == [
        "Our names are [PERSON], [PERSON], and [PERSON].",
        "My number is [PHONENO] and I live at [POSTCODE].",
    ]


def test_text_id():
    scrub = IDScrub(["Our names are Hamish McDonald, L. Salah, and Elena Suárez."] * 10)

    scrub.all()

    df = scrub.get_scrubbed_data()

    assert df["text_id"].max() == 10
    assert len(df["text_id"]) == 10


def test_get_scrubbed_data(scrub_object):
    scrub_object.all()
    df = scrub_object.get_scrubbed_data()

    expected_df = pd.DataFrame(
        {
            "text_id": {0: 1, 1: 2},
            "scrubbed_presidio_person": {0: ["Hamish McDonald", "L. Salah", "Elena Suárez"], 1: None},
            "scrubbed_uk_phone_numbers": {0: None, 1: ["+441111111111"]},
            "scrubbed_uk_postcodes": {0: None, 1: ["AA11 1AA"]},
        }
    )

    assert_frame_equal(df, expected_df)
