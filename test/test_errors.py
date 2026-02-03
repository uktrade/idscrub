import pytest
from idscrub import IDScrub


def test_scrub_input():
    with pytest.raises(TypeError):
        IDScrub(texts=[123])
    with pytest.raises(TypeError):
        IDScrub(texts=[1, 2, 3])
    with pytest.raises(TypeError):
        IDScrub(texts=[1.0, 2.0, 3.0])
    with pytest.raises(TypeError):
        IDScrub(texts="not_a_list")


def test_scrub_input_text_ids():
    with pytest.raises(ValueError):
        IDScrub(texts=["Hello"], text_ids=[1, 2])


def test_replacement_error():
    with pytest.raises(TypeError):
        IDScrub(texts=["Hello"], text_ids=[1], replacement=1)
    with pytest.raises(TypeError):
        IDScrub(texts=["Hello"], text_ids=[1], replacement=1.0)
    with pytest.raises(TypeError):
        IDScrub(texts=["Hello"], text_ids=[1], replacement=["ok"])


def test_scrub_pipeline_error(scrub_object):
    with pytest.raises(TypeError):
        scrub_object.scrub(pipeline={"method": "spacy_entities"})
