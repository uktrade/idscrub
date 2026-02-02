import pandas as pd
import pytest
from idscrub import IDScrub
from pandas.testing import assert_frame_equal


def test_huggingface():
    scrub = IDScrub(texts=["Our names are Hamish McDonald, L. Salah, and Elena Suárez."])
    scrubbed = scrub.scrub(pipeline=[{"method": "huggingface_entities"}])
    assert scrubbed == ["Our names are [PERSON], [PERSON], and [PERSON]."]


def test_huggingface_error():
    scrub = IDScrub(texts=["Our names are Hamish McDonald, L. Salah, and Elena Suárez."])

    with pytest.raises(OSError):
        scrub.scrub(pipeline=[{"method": "huggingface_entities", "hf_model_path": "not_a_model"}])


def test_huggingface_empty():
    scrub = IDScrub([" ", "John Smith", ""])
    scrubbed = scrub.scrub(pipeline=[{"method": "huggingface_entities"}])

    assert scrubbed == [" ", "[PERSON]", ""]
    assert_frame_equal(scrub.get_scrubbed_data(), pd.DataFrame({"text_id": 2, "person": [["John Smith"]]}))
