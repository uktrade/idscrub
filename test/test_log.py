from idscrub import IDScrub


def test_log_message():
    scrub = IDScrub(texts=["My name is Dr Strangelove. Dr. Strangelove is my name", "My name is Professor Oppenheimer"])
    scrub.titles()
    count = scrub.log_message("scrubbed_titles")
    assert count == 3


def test_log_message_custom_regex():
    scrub = IDScrub(texts=["My name is Dr Strangelove. Dr. Strangelove is my name", "My name is Professor Oppenheimer"])
    scrub.custom_regex([r"Strangelove", r"Oppenheimer"], ["[DR]", "[PROFESSOR]"])
    count_1 = scrub.log_message("scrubbed_custom_regex_1")
    count_2 = scrub.log_message("scrubbed_custom_regex_2")
    assert count_1 == 2
    assert count_2 == 1
