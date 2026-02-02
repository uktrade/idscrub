from idscrub import IDScrub


def test_overlap():
    scrub = IDScrub(texts=["My email is fakeperson@fakeemail.com"])
    scrubbed = scrub.scrub(
        pipeline=[{"method": "handles", "priority": 0.1}, {"method": "email_addresses", "priority": 1.0}]
    )
    assert max([ident.priority for ident in scrub.idents_all]) == 1.0
    assert scrub.idents_all == [
        IDScrub.IDEnt(
            text_id=1,
            text="@fakeemail.com",
            start=22,
            end=36,
            label="handle",
            replacement="[HANDLE]",
            priority=0.1,
            source="regex",
        ),
        IDScrub.IDEnt(
            text_id=1,
            text="fakeperson@fakeemail.com",
            start=12,
            end=36,
            label="email_address",
            replacement="[EMAIL_ADDRESS]",
            priority=1.0,
            source="regex",
        ),
    ]
    assert scrub.idents == [
        IDScrub.IDEnt(
            text_id=1,
            text="fakeperson@fakeemail.com",
            start=12,
            end=36,
            label="email_address",
            replacement="[EMAIL_ADDRESS]",
            priority=1.0,
            source="regex",
        )
    ]
    assert scrubbed == ["My email is [EMAIL_ADDRESS]"]


def test_overlap_default():
    scrub = IDScrub(texts=["I am @John Smith"])
    scrubbed = scrub.scrub(pipeline=[{"method": "spacy_entities", "entity_types": ["PERSON"]}, {"method": "handles"}])
    assert max([ident.priority for ident in scrub.idents_all]) == 1.0
    assert scrub.idents_all == [
        IDScrub.IDEnt(
            text_id=1,
            text="@John Smith",
            start=5,
            end=16,
            label="person",
            replacement="[PERSON]",
            priority=1.0,
            source="spacy",
        ),
        IDScrub.IDEnt(
            text_id=1,
            text="@John",
            start=5,
            end=10,
            label="handle",
            replacement="[HANDLE]",
            priority=0.4,
            source="regex",
        ),
    ]
    assert scrub.idents == [
        IDScrub.IDEnt(
            text_id=1,
            text="@John Smith",
            start=5,
            end=16,
            label="person",
            replacement="[PERSON]",
            priority=1.0,
            source="spacy",
        )
    ]

    assert scrubbed == ["I am [PERSON]"]
