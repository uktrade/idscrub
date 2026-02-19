import pandas as pd
import pytest
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

    scrubbed_df, scrubbed_data = IDScrub.dataframe(df=df, id_col="ID")

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
            "person": [["Darcy", "Elizabeth"], ["Bennet"], None, ["Mick Jagger", "David Bowie"]],
            "title": [["Mr"], ["Mr"], None, None],
            "email_address": [None, None, ["freddie-mercury@queen.com"], None],
            "uk_postcode": [None, None, ["SW1A 2AA"], ["SW1A 2WH"]],
        }
    )

    assert_frame_equal(scrubbed_df, expected_scrubbed_df)
    assert_frame_equal(scrubbed_data, expected_scrubbed_data)


def test_dataframe_exclude():
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

    scrubbed_df, scrubbed_data = IDScrub.dataframe(df=df, id_col="ID", exclude_cols=["Fake book"])

    expected_scrubbed_df = pd.DataFrame(
        {
            "ID": [1, 2],
            "Pride and Prejudice": [
                "[TITLE]. [PERSON] walked off; and [PERSON] remained with no very cordial feelings toward him.",
                "[TITLE]. [PERSON] was so odd a mixture of quick parts, sarcastic humour, reserve, and caprice.",
            ],
            "Fake book": [
                "The letter to freddie-mercury@queen.com was stamped with SW1A 2AA.",
                "She forwarded the memo from Mick Jagger and David Bowie to her chief of staff, noting the postcode SW1A 2WH.",
            ],
        }
    )

    expected_scrubbed_data = pd.DataFrame(
        {
            "ID": [1, 2],
            "column": ["Pride and Prejudice", "Pride and Prejudice"],
            "person": [["Darcy", "Elizabeth"], ["Bennet"]],
            "title": [["Mr"], ["Mr"]],
        }
    )

    assert_frame_equal(scrubbed_df, expected_scrubbed_df)
    assert_frame_equal(scrubbed_data, expected_scrubbed_data)


def test_dataframe_scrub_methods():
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

    scrubbed_df, scrubbed_data = IDScrub.dataframe(df=df, id_col="ID", pipeline=[{"method": "titles"}])

    expected_scrubbed_df = pd.DataFrame(
        {
            "ID": [1, 2],
            "Pride and Prejudice": [
                "[TITLE]. Darcy walked off; and Elizabeth remained with no very cordial feelings toward him.",
                "[TITLE]. Bennet was so odd a mixture of quick parts, sarcastic humour, reserve, and caprice.",
            ],
            "Fake book": [
                "The letter to freddie-mercury@queen.com was stamped with SW1A 2AA.",
                "She forwarded the memo from Mick Jagger and David Bowie to her chief of staff, noting the postcode SW1A 2WH.",
            ],
        }
    )

    expected_scrubbed_data = pd.DataFrame(
        {
            "ID": [1, 2],
            "column": ["Pride and Prejudice", "Pride and Prejudice"],
            "title": [["Mr"], ["Mr"]],
        }
    )

    assert_frame_equal(scrubbed_df, expected_scrubbed_df)
    assert_frame_equal(scrubbed_data, expected_scrubbed_data)


def test_dataframe_errors():
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

    with pytest.raises(ValueError):
        IDScrub.dataframe(df=df, id_col="ID_not_present")

    with pytest.raises(TypeError):
        IDScrub.dataframe(df=1, id_col="ID_not_present")


def test_dataframe_defaults_and_kwargs():

    df = pd.DataFrame({"data": ["My postcode is SW1A 2AA", "My postcode is SW1A 2AA"]})

    pipeline = [{"method": "uk_postcodes"}]

    scrubbed_df, scrubbed_data = IDScrub.dataframe(df=df, pipeline=pipeline, replacement="[TEST]")

    assert scrubbed_df["data"].to_list() == ["My postcode is [TEST]", "My postcode is [TEST]"]
    assert scrubbed_df.columns.to_list() == ["data"]
    assert scrubbed_data["uk_postcode"].to_list() == [["SW1A 2AA"], ["SW1A 2AA"]]
    assert scrubbed_data["row_id"].to_list() == [1, 2]
