from idscrub import IDScrub


def test_group_idents(idents):
    scrub = IDScrub(texts=[])
    entities_grouped = scrub.group_idents(idents)

    assert len(entities_grouped) == 2
    assert list(entities_grouped.keys()) == ["A", "B"]
