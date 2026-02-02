from idscrub import IDScrub


def test_exclude():
    scrub = IDScrub(
        [
            "Our names are Hamish McDonald, L. Salah, and Elena Suárez.",
        ],
        exclude=["Hamish McDonald", "L. Salah"],
    )
    scrubbed = scrub.scrub(
        pipeline=[{"method": "spacy_entities"}],
    )

    assert scrubbed == [
        "Our names are Hamish McDonald, L. Salah, and [PERSON].",
    ]

    assert scrub.idents_all[0].text == "Hamish McDonald"
    assert scrub.idents_all[1].text == "L. Salah"

    assert [ident.text for ident in scrub.idents] not in ["Hamish McDonald", "L. Salah"]
