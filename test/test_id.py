from idscrub import IDScrub


def test_id_ints():
    scrub = IDScrub(texts=["clement_attlee@gmail.com"] * 10, text_ids=range(100, 110), text_id_name="PM")
    scrub.email_addresses()
    assert scrub.get_scrubbed_data()["PM"].min() == 100
    assert scrub.get_scrubbed_data()["PM"].max() == 109
    assert scrub.get_scrubbed_data()["PM"].to_list() == [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]


def test_id_strs():
    scrub = IDScrub(texts=["clement_attlee@gmail.com"] * 2, text_ids=["random", "minister"], text_id_name="PM")
    scrub.email_addresses()
    assert scrub.get_scrubbed_data()["PM"][0] == "random"
    assert scrub.get_scrubbed_data()["PM"][1] == "minister"


def test_multiple():
    scrub = IDScrub(texts=["clement_attlee@gmail.com", "SW1A 2AA"] * 10, text_ids=range(100, 120), text_id_name="PM")
    scrub.email_addresses()
    scrub.uk_postcodes()
    assert scrub.get_scrubbed_data()["PM"].min() == 100
    assert scrub.get_scrubbed_data()["PM"].max() == 119
