import re

from idscrub import IDScrub


def test_email_addresses():
    scrub = IDScrub(
        texts=["Send me an email at jim@gmail.com or at marie-9999@randomemail.co.uk or at hello_world@john-smith.com."]
    )
    scrubbed = scrub.email_addresses()
    assert scrubbed == ["Send me an email at [EMAIL_ADDRESS] or at [EMAIL_ADDRESS] or at [EMAIL_ADDRESS]."]


def test_ip_addresses():
    scrub = IDScrub(texts=["This has been sent to 8.8.8.8 and requested by 192.0.2.1."])
    scrubbed = scrub.ip_addresses()
    assert scrubbed == ["This has been sent to [IPADDRESS] and requested by [IPADDRESS]."]


def test_uk_postcodes():
    scrub = IDScrub(texts=["I live at A11 1AA. My friend lives at KA308JB. The Prime Minister lives at SW1A  2AA."])
    scrubbed = scrub.uk_postcodes()
    assert scrubbed == ["I live at [POSTCODE]. My friend lives at [POSTCODE]. The Prime Minister lives at [POSTCODE]."]


def test_titles_not_strict():
    scrub = IDScrub(
        texts=[
            "Hello Dr. Smith! I am Mrs Patel",
            "I am here on behalf of Ms Austen, General Eisenhower, and Captain Jack Sparrow.",
        ]
    )
    scrubbed = scrub.titles()
    assert scrubbed == [
        "Hello [TITLE]. Smith! I am [TITLE] Patel",
        "I am here on behalf of [TITLE] Austen, General Eisenhower, and [TITLE] Jack Sparrow.",
    ]


def test_titles_strict():
    scrub = IDScrub(
        texts=[
            "Hello Dr. Smith! I am Mrs Patel",
            "I am here on behalf of Ms Austen, General Eisenhower, and Captain Jack Sparrow.",
        ]
    )
    scrubbed = scrub.titles(strict=True)
    assert scrubbed == [
        "Hello [TITLE]. Smith! I am [TITLE] Patel",
        "I am here on behalf of [TITLE] Austen, [TITLE] Eisenhower, and [TITLE] Jack Sparrow.",
    ]


def test_uk_phone_numbers():
    scrub = IDScrub(texts=["My phone number is +441234567891! My old phone number is 01111 123456."])
    scrubbed = scrub.uk_phone_numbers()
    assert scrubbed == ["My phone number is [PHONENO]! My old phone number is [PHONENO]."]


def test_handles():
    scrub = IDScrub(texts=["Our usernames are @HenrikLarsson, @Jimmy_Johnstone, @Nakamura-67 and @Aidan_McGeady_46."])
    scrubbed = scrub.handles()
    assert scrubbed == ["Our usernames are [HANDLE], [HANDLE], [HANDLE] and [HANDLE]."]


def test_claimants():
    scrub = IDScrub(
        texts=[
            "This is legal text. Claimant: John Smith Respondents: Jill Hill.",
            "Claimant: J Smith Respondents: Jill Hill. J Smith is the respondent.",
        ]
    )
    scrubbed = scrub.claimants()
    assert scrubbed == [
        "This is legal text. Claimant: [CLAIMANT] Respondents: Jill Hill.",
        "Claimant: [CLAIMANT] Respondents: Jill Hill. [CLAIMANT] is the respondent.",
    ]


def test_custom_regex():
    scrub = IDScrub(texts=["It was the best of times, it was the worst of times"])
    scrubbed = scrub.custom_regex(custom_regex_patterns=[r"times"])
    assert scrubbed == ["It was the best of [REDACTED], it was the worst of [REDACTED]"]

    scrub = IDScrub(texts=["It was the best of times, it was the worst of times"])
    scrubbed = scrub.custom_regex(
        custom_regex_patterns=[r"times", "worst"], custom_replacement_texts=["[DICKENS]", "[WORST]"]
    )
    assert scrubbed == ["It was the best of [DICKENS], it was the [WORST] of [DICKENS]"]


def test_scrub_and_collect():
    scrub = IDScrub()
    text = "Hello Muhammad and Jack."
    pattern = r"\bMuhammad|Jack\b"
    replacement = "[NAME]"
    removed_label = "scrubbed_custom_regex"
    i = 1

    def replacer(match):
        return scrub.scrub_and_collect(match, text, replacement, i, removed_label)

    scrubbed = re.sub(pattern, replacer, text)

    assert scrubbed == "Hello [NAME] and [NAME]."
    assert scrub.scrubbed_data == [
        {"text_id": 1, "scrubbed_custom_regex": "Muhammad"},
        {"text_id": 1, "scrubbed_custom_regex": "Jack"},
    ]


def test_remove_regex():
    scrub = IDScrub(texts=["Hi! My name is Clement Atlee!", "I am Harold Wilson."])
    removed_label = "scrubbed_regex_names"
    pattern = r"Clement Atlee|Harold Wilson"
    replacement_text = "[PM]"
    scrubbed = scrub.scrub_regex(pattern, replacement_text, removed_label)

    assert scrubbed == ["Hi! My name is [PM]!", "I am [PM]."]
    assert scrub.scrubbed_data == [
        {"text_id": 1, "scrubbed_regex_names": "Clement Atlee"},
        {"text_id": 2, "scrubbed_regex_names": "Harold Wilson"},
    ]
