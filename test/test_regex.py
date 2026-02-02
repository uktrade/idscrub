from idscrub import IDScrub


def test_email_addresses():
    scrub = IDScrub(
        texts=[
            "Send me an email at jim@testemail.com or at marie-9999@randomemail.co.uk or at hello_world@john-smith.com."
        ]
    )
    scrubbed = scrub.scrub(pipeline=[{"method": "email_addresses"}])
    assert scrubbed == ["Send me an email at [EMAIL_ADDRESS] or at [EMAIL_ADDRESS] or at [EMAIL_ADDRESS]."]


def test_ip_addresses():
    scrub = IDScrub(texts=["This has been sent to 8.8.8.8 and requested by 192.0.2.1."])
    scrubbed = scrub.scrub(pipeline=[{"method": "ip_addresses"}])
    assert scrubbed == ["This has been sent to [IPADDRESS] and requested by [IPADDRESS]."]


def test_uk_postcodes():
    scrub = IDScrub(texts=["I live at A11 1AA. My friend lives at KA308JB. The Prime Minister lives at SW1A  2AA."])
    scrubbed = scrub.scrub(pipeline=[{"method": "uk_postcodes"}])
    assert scrubbed == ["I live at [POSTCODE]. My friend lives at [POSTCODE]. The Prime Minister lives at [POSTCODE]."]


def test_titles_not_strict():
    scrub = IDScrub(
        texts=[
            "Hello Dr. Smith! I am Mrs Patel",
            "I am here on behalf of Ms Austen, General Eisenhower, and Captain Jack Sparrow.",
        ]
    )
    scrubbed = scrub.scrub(pipeline=[{"method": "titles"}])
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
    scrubbed = scrub.scrub(pipeline=[{"method": "titles", "strict": True}])
    assert scrubbed == [
        "Hello [TITLE]. Smith! I am [TITLE] Patel",
        "I am here on behalf of [TITLE] Austen, [TITLE] Eisenhower, and [TITLE] Jack Sparrow.",
    ]


def test_uk_phone_numbers():
    scrub = IDScrub(texts=["My phone number is +441234567891! My old phone number is 01111 123456."])
    scrubbed = scrub.scrub(pipeline=[{"method": "uk_phone_numbers"}])
    assert scrubbed == ["My phone number is [PHONENO]! My old phone number is [PHONENO]."]


def test_handles():
    scrub = IDScrub(texts=["Our usernames are @HenrikLarsson, @Jimmy_Johnstone, @Nakamura-67 and @Aidan_McGeady_46."])
    scrubbed = scrub.scrub(pipeline=[{"method": "handles"}])
    assert scrubbed == ["Our usernames are [HANDLE], [HANDLE], [HANDLE] and [HANDLE]."]


def test_urls():
    scrub = IDScrub(
        [
            "www.example.co.uk",
            "https://example.com",
            "http://sub.domain.co.uk/path?query=1&x=2",
            "www.example.org/page/index.html",
            "https://example.com:8080/path/to/resource#anchor",
            "www.test-site123.net/some/path?with=paramsexample.comexample.co.uk/home",
        ]
    )

    scrubbed = scrub.scrub(pipeline=[{"method": "urls"}])

    assert scrubbed == ["[URL]", "[URL]", "[URL]", "[URL]", "[URL]", "[URL]"]


def test_uk_addresses():
    scrub = IDScrub(
        [
            "221B Baker Street",
            "12 high road",
            "Flat 3B, 47 King's Court",
            "12-14 High Street",
            "5a-7a Church Lane",
            "1/2 Main Street",
            "10 St John’s Rd",
            "33 Queen-Anne Walk",
            "8 Deansgate Ct",
        ],
    )

    scrubbed = scrub.scrub(pipeline=[{"method": "uk_addresses"}])

    assert scrubbed == [
        "[ADDRESS]",
        "[ADDRESS]",
        "[ADDRESS]",
        "[ADDRESS]",
        "[ADDRESS]",
        "[ADDRESS]",
        "[ADDRESS]",
        "[ADDRESS]",
        "[ADDRESS]",
    ]

    negative_tests = [
        "12 High",
        "Baker Street",
        "High Road 12",
        "Go to the high road now",
        "500 the big building near river",
        "I walked the long road home",
        "12b misspelledstreet",
        "London SW1A 1AA",
        "12,,, High?",
    ]

    scrub = IDScrub(negative_tests)

    scrubbed = scrub.scrub(pipeline=[{"method": "uk_addresses"}])
    assert scrubbed == negative_tests


def test_custom_regex():
    scrub = IDScrub(texts=[])

    scrubbed_idents = scrub.custom_regex(
        texts=["It was the best of times, it was the worst of times"],
        text_ids=["A"],
        patterns={
            "times": {"pattern": r"times", "replacement": "[DICKENS]", "priority": 0.5},
            "worst": {"pattern": r"worst", "replacement": "[WORST]", "priority": 0.8},
        },
    )

    assert scrubbed_idents == [
        IDScrub.IDEnt(
            text_id="A",
            text="times",
            start=19,
            end=24,
            label="times",
            replacement="[DICKENS]",
            priority=0.5,
            source="custom_regex",
        ),
        IDScrub.IDEnt(
            text_id="A",
            text="times",
            start=46,
            end=51,
            label="times",
            replacement="[DICKENS]",
            priority=0.5,
            source="custom_regex",
        ),
        IDScrub.IDEnt(
            text_id="A",
            text="worst",
            start=37,
            end=42,
            label="worst",
            replacement="[WORST]",
            priority=0.8,
            source="custom_regex",
        ),
    ]

    scrub = IDScrub(
        texts=[
            "It was the best of times, it was the worst of times",
        ]
    )

    scrubbed_text = scrub.scrub(
        pipeline=[
            {
                "method": "custom_regex",
                "patterns": {
                    "times": {"pattern": r"times", "replacement": "[DICKENS]", "priority": 0.5},
                    "worst": {"pattern": r"worst", "replacement": "[WORST]", "priority": 0.5},
                },
            }
        ]
    )

    assert scrubbed_text == ["It was the best of [DICKENS], it was the [WORST] of [DICKENS]"]


def test_remove_regex():
    texts = ["Hi! My name is Clement Atlee!"]
    text_ids = ["UK"]
    scrub = IDScrub([])
    label = "regex_names"
    pattern = r"Clement Atlee|Harold Wilson"
    replacement = "[PM]"
    priority = 0.5
    idents = scrub.find_regex(
        texts=texts, text_ids=text_ids, pattern=pattern, replacement=replacement, label=label, priority=priority
    )

    assert len(idents) == 1
    assert idents[0].text_id == "UK"
    assert idents[0].text == "Clement Atlee"
    assert idents[0].start == 15
    assert idents[0].end == 28
    assert idents[0].label == "regex_names"
    assert idents[0].replacement == "[PM]"
    assert idents[0].priority == 0.5
    assert idents[0].source == "regex"
