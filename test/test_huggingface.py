import pandas as pd
import pytest
from idscrub import IDScrub
from pandas.testing import assert_frame_equal


def test_huggingface():
    scrub = IDScrub(texts=["Our names are Hamish McDonald, L. Salah, and Elena Suárez."])
    scrubbed = scrub.huggingface_persons()
    assert scrubbed == ["Our names are [PERSON], [PERSON], and [PERSON]."]


def test_huggingface_error():
    scrub = IDScrub(texts=["Our names are Hamish McDonald, L. Salah, and Elena Suárez."])

    with pytest.raises(OSError):
        scrub.huggingface_persons(hf_model_path="not_a_path")


def test_huggingface_empty():
    scrub = IDScrub([" ", "John Smith", ""])
    scrubbed = scrub.huggingface_persons()

    assert scrubbed == [" ", "[PERSON]", ""]
    assert_frame_equal(scrub.get_scrubbed_data(), pd.DataFrame({"text_id": 2, "scrubbed_hf_person": [["John Smith"]]}))
