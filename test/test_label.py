def test_label(scrub_object_all):
    scrub_object_all.scrub(
        pipeline=[
            {"method": "uk_postcodes", "label": "test"},
            {"method": "email_addresses", "label": "test"},
            {"method": "ip_addresses", "label": "test"},
            {"method": "uk_phone_numbers", "label": "test"},
            {"method": "titles", "label": "test"},
            {"method": "handles", "label": "test"},
        ]
    )

    df = scrub_object_all.get_scrubbed_data()

    assert df.columns.to_list() == ["text_id", "test"]


def test_regex_label(scrub_object_all):
    scrub_object_all.scrub(
        pipeline=[
            {
                "method": "custom_regex",
                "patterns": {
                    "number": {"pattern": r"number", "replacement": "[REDACTED]", "priority": 0.5},
                    "live": {"pattern": r"live", "replacement": "[REDACTED]"},
                },
            }
        ]
    )
    df = scrub_object_all.get_scrubbed_data()

    assert df.columns.to_list() == ["text_id", "number", "live"]
