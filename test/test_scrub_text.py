from idscrub import IDScrub


def test_scrub_text(scrub_object):
    scrub_object.scrub(pipeline=[{"method": "uk_postcodes"}])

    assert scrub_object.idents == [
        IDScrub.IDEnt(
            text_id=2,
            text="AA11 1AA",
            start=41,
            end=49,
            label="uk_postcode",
            replacement="[POSTCODE]",
            priority=0.5,
            source="regex",
        )
    ]
    assert scrub_object.scrub_text() == [
        "Our names are Hamish McDonald, L. Salah, and Elena Suárez.",
        "My number is +441111111111 and I live at [POSTCODE].",
    ]
