from idscrub import IDScrub


def test_google_phone_numbers_gb():
    scrub = IDScrub(texts=["My phone number is +441234567891! My old phone number is 01475 123456."])
    scrubbed = scrub.google_phone_numbers(region="GB")
    assert scrubbed == ["My phone number is [PHONENO]! My old phone number is [PHONENO]."]


def test_google_phone_numbers_us():
    scrub = IDScrub(texts=["My US phone number is +1-718-222-2222! My old phone number is 12124567890."])
    scrubbed = scrub.google_phone_numbers(region="US")
    assert scrubbed == ["My US phone number is [PHONENO]! My old phone number is [PHONENO]."]
