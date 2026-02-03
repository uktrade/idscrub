import pandas as pd
import pytest
from idscrub import IDScrub
from pandas.testing import assert_frame_equal


def test_spacy():
    scrub = IDScrub(texts=["Our names are Hamish McDonald, L. Salah, and Elena Suárez."])
    scrubbed = scrub.scrub(pipeline=[{"method": "spacy_entities"}])
    assert scrubbed == ["Our names are [PERSON], [PERSON], and [PERSON]."]


def test_spacy_error():
    scrub = IDScrub(texts=["Our names are Hamish McDonald, L. Salah, and Elena Suárez."])

    with pytest.raises(ValueError):
        scrub.scrub(pipeline=[{"method": "spacy_entities", "model_name": "not_a_model"}])


def test_spacy_empty():
    scrub = IDScrub([" ", "  John Smith", ""])
    scrubbed = scrub.scrub(pipeline=[{"method": "spacy_entities"}])

    assert scrubbed == [" ", "  [PERSON]", ""]
    assert_frame_equal(scrub.get_scrubbed_data(), pd.DataFrame({"text_id": 2, "person": [["John Smith"]]}))


def test_spacy_map():
    scrub = IDScrub(["Our names are Hamish McDonald, L. Salah, and Elena Suárez.", "My company code is NASA."])
    scrubbed = scrub.scrub(
        pipeline=[
            {
                "method": "spacy_entities",
                "entity_types": ["PERSON", "ORG"],
                "replacement_map": {"PERSON": "[PHELLO]", "ORG": "[SPACE]"},
            }
        ]
    )
    assert scrubbed == ["Our names are [PHELLO], [PHELLO], and [PHELLO].", "My company code is [SPACE]."]


def test_presidio_get_data():
    scrub = IDScrub(["Our names are Hamish McDonald, L. Salah, and Elena Suárez.", "My company code is NASA."])

    scrub.scrub(pipeline=[{"method": "spacy_entities"}])

    df = scrub.get_scrubbed_data()

    expected_df = pd.DataFrame(
        {
            "text_id": {0: 1, 1: 2},
            "person": {0: ["Hamish McDonald", "L. Salah", "Elena Suárez"], 1: None},
            "org": {0: None, 1: ["NASA"]},
        }
    )

    assert_frame_equal(df, expected_df)
