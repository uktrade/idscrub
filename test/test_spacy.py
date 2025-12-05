import pandas as pd
import pytest
from idscrub import IDScrub
from pandas.testing import assert_frame_equal


# Note: This test will fail if the kernel has not been restarted since the SpaCy model was downloaded.
def test_spacy():
    scrub = IDScrub(texts=["Our names are Hamish McDonald, L. Salah, and Elena Suárez."])
    scrubbed = scrub.spacy_persons(model_name="en_core_web_trf")
    assert scrubbed == ["Our names are [PERSON], [PERSON], and [PERSON]."]


def test_spacy_error():
    scrub = IDScrub(texts=["Our names are Hamish McDonald, L. Salah, and Elena Suárez."])

    with pytest.raises(ValueError):
        scrub.spacy_persons(model_name="not_a_model")


def test_spacy_empty():
    scrub = IDScrub([" ", "John Smith", ""])
    scrubbed = scrub.spacy_persons()

    assert scrubbed == [" ", "[PERSON]", ""]
    assert_frame_equal(
        scrub.get_scrubbed_data(), pd.DataFrame({"text_id": 2, "scrubbed_spacy_person": [["John Smith"]]})
    )
