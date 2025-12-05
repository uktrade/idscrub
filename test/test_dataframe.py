import pandas as pd
from idscrub import IDScrub
from pandas.testing import assert_frame_equal


# Note: These tests will fail if the kernel has not been restarted since the SpaCy model was downloaded.
def test_dataframe_outputs():
    df = pd.DataFrame(
        {
            "ID": [1, 2],
            "Pride and Prejudice": [
                "Mr. Darcy walked off; and Elizabeth remained with no very cordial feelings toward him.",
                "Mr. Bennet was so odd a mixture of quick parts, sarcastic humour, reserve, and caprice.",
            ],
            "Fake book": [
                "The letter to freddie-mercury@queen.com was stamped with SW1A 2AA.",
                "She forwarded the memo from Mick Jagger and David Bowie to her chief of staff, noting the postcode SW1A 2WH.",
            ],
        }
    )

    scrubbed_df, scrubbed_data = IDScrub.dataframe(df=df, id_col="ID", scrub_methods=["all"])

    expected_scrubbed_df = pd.DataFrame(
        {
            "ID": [1, 2],
            "Pride and Prejudice": [
                "[TITLE]. [PERSON] walked off; and [PERSON] remained with no very cordial feelings toward him.",
                "[TITLE]. [PERSON] was so odd a mixture of quick parts, sarcastic humour, reserve, and caprice.",
            ],
            "Fake book": [
                "The letter to [EMAIL_ADDRESS] was stamped with [POSTCODE].",
                "She forwarded the memo from [PERSON] and [PERSON] to her chief of staff, noting the postcode [POSTCODE].",
            ],
        }
    )

    expected_scrubbed_data = pd.DataFrame(
        {
            "ID": [1, 2, 1, 2],
            "column": ["Pride and Prejudice", "Pride and Prejudice", "Fake book", "Fake book"],
            "scrubbed_presidio_person": [["Darcy", "Elizabeth"], ["Bennet"], None, ["Mick Jagger", "David Bowie"]],
            "scrubbed_titles": [["Mr"], ["Mr"], None, None],
            "scrubbed_presidio_email_address": [None, None, ["freddie-mercury@queen.com"], None],
            "scrubbed_presidio_url": [None, None, ["queen.com"], None],
            "scrubbed_uk_postcodes": [None, None, ["SW1A 2AA"], ["SW1A 2WH"]],
        }
    )

    assert_frame_equal(scrubbed_df, expected_scrubbed_df)
    assert_frame_equal(scrubbed_data, expected_scrubbed_data)
