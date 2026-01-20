def test_label(scrub_object_all):
    for i, scrub_method in enumerate(
        ["uk_postcodes", "email_addresses", "ip_addresses", "uk_phone_numbers", "titles", "handles"]
    ):
        method = getattr(scrub_object_all, scrub_method)
        method(label="test")

    df = scrub_object_all.get_scrubbed_data()

    assert df.columns.to_list() == ["text_id", "test"]


def test_regex_label(scrub_object_all):
    scrub_object_all.custom_regex(custom_regex_patterns=[r"number", r"live"], labels=["regex_number", "regex_live"])
    df = scrub_object_all.get_scrubbed_data()

    assert df.columns.to_list() == ["text_id", "regex_number", "regex_live"]
