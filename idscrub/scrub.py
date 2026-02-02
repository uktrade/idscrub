import logging
import os
import re
import warnings
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass

import pandas as pd
import phonenumbers
import spacy
from huggingface_hub.utils import HFValidationError
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import SpacyNlpEngine
from spacy.cli import download
from spacy.language import Language
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
from transformers.utils import logging as trf_logging

from idscrub.locations import DOWNLOAD_DIR

# Suppress Torch FutureWarning
# TODO: Find better way
warnings.simplefilter(action="ignore", category=FutureWarning)

trf_logging.set_verbosity_error()


class IDScrub:
    """
    Class for identifying and scrubbing entities in text.
    """

    @dataclass
    class IDEnt:
        """
        Structured representation of an identified entity (ident) within text.

        Attributes:
            text_id (str | int | float): A unique identifier for the original text.
            text (str): The exact substring extracted from the original text.
            start (int): The starting character offset of the ident within the original text.
            end (int): The ending character offset of the ident within the original text.
            label (str): The ident type (e.g. 'person').
            replacement (str): The text that should replace this ident during scrubbing.
            priority (float): A priority score for overlapping entitities.
            source (str): The source model or method that identified the ident.
        """

        text_id: str | int | float
        text: str
        start: int
        end: int
        label: str
        replacement: str
        priority: float
        source: str

    def __init__(
        self,
        texts: list[str] = None,
        text_ids: list | Iterable = None,
        text_id_name: str = "text_id",
        replacement: str = None,
        exclude: list[str] = [],
        verbose: bool = True,
    ):
        """
        Instantiate IDScrub object with list of texts and optional IDs, ID name, and global replacement text.

        Attributes:
            texts (list[str]): A list of strings to scrub.
            text_ids (list): A list of identifiers that correspond to each string in `texts`,
            such as the ID column in a DataFrame. If None, an integer index starting at 1 is applied.
            This is used to identify texts in get_scrubbed_data().
            text_id_name (str): Name of the ID column in get_scrubbed_data(). Default is `text_id`.
            replacement (str): A global string to replace every scrubbed string with.
            exclude (list[str]): A list of strings that will not be scrubbed if identified.
            verbose (bool): Whether to show all log messages or only warnings.
        """

        if not isinstance(texts, list):
            raise TypeError("`texts` must be a list.")
        if not all(isinstance(text, str) for text in texts):
            raise TypeError("`texts` must be a list of strings.")

        if replacement is not None and not isinstance(replacement, str):
            raise TypeError("`replacement` must be a string or None.")

        self.texts = texts

        if text_ids is None:
            text_ids = range(1, len(self.texts) + 1)

        if not len(self.texts) == len(text_ids):
            raise ValueError("Length of texts is different to the length of text IDs.")

        self.text_ids = text_ids

        self.replacement = replacement
        self.text_id_name = text_id_name
        self.exclude = exclude
        self.scrubbed_texts = []
        self.idents: list[IDScrub.IDEnt] = []

        self.hf_ner = None
        self.spacy_docs = None

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG if verbose else logging.WARNING)

        if not self.logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(levelname)s: %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.logger.info("Texts loaded.")

    def find_regex(
        self,
        texts: list[str],
        text_ids: list,
        pattern: str,
        replacement: str,
        label: str,
        priority: float,
    ) -> list[IDEnt]:
        """
        General method to clean text using a regex pattern.

        Args:
            texts (list[str]): Strings to scrub.
            text_ids (list): A list of identifiers that correspond to each string in `texts`.
            If None, current cleaned state of `texts` passed at Class initiation used.
            pattern (str): Regex pattern to apply.
            replacement (str): The replacement text for the removed text.
            label (str): Label for the personal data removed.
            priority (float): Priority score for personal data match (range 0 - 1).
            Higher scored matches are scrubbed when overlapping personal data found.

        Returns:
            list[IDEnt]: A list of IDEnt objects.
        """

        if self.replacement:
            replacement = self.replacement

        compiled = re.compile(pattern, re.IGNORECASE)
        idents = []

        for text_id, text in zip(text_ids, texts):
            for match in compiled.finditer(text):
                idents.append(
                    self.IDEnt(
                        text_id=text_id,
                        text=match.group(),
                        start=match.start(),
                        end=match.end(),
                        label=label,
                        replacement=replacement,
                        priority=priority,
                        source="regex",
                    )
                )

        return idents

    def custom_regex(
        self, texts: list[str] = None, text_ids: list = None, patterns: dict = None, source: str = "custom_regex"
    ) -> list[IDEnt]:
        """
        Remove text matching a custom regex pattern.

        Args:
            texts (list[str]): Strings to scrub.
            text_ids (list): A list of identifiers that correspond to each string in `texts`.
            patterns (dict): {"name": {"pattern": r"John", "replacement": "[NAME]", "priority": 0.5}}
            source (str): The methodological source of the scrubbed ident.
        Returns:
            list[IDEnt]: A list of IDEnt objects.

        """

        idents = []

        for text, text_id in zip(texts, text_ids):
            for label, params in patterns.items():
                pattern = params["pattern"]
                replacement = params.get("replacement", "[REDACTED]")
                priority = params.get("priority", 0.5)

                compiled = re.compile(pattern, flags=re.IGNORECASE)

                for match in compiled.finditer(text):
                    idents.append(
                        self.IDEnt(
                            text_id=text_id,
                            text=match.group(),
                            start=match.start(),
                            end=match.end(),
                            label=label,
                            replacement=replacement,
                            priority=priority,
                            source=source,
                        )
                    )

        return idents

    def email_addresses(
        self,
        texts: list[str] = None,
        text_ids: list = None,
        replacement: str = "[EMAIL_ADDRESS]",
        label: str = "email_address",
        priority: float = 0.7,
    ) -> list[IDEnt]:
        """
        Remove email addresses using regex.
        e.g. `johnsmith@mail.com` scrubbed

        Args:
            texts (list[str]): Strings to scrub.
            text_ids (list): A list of identifiers that correspond to each string in `texts`.
            If None, current cleaned state of `texts` passed at Class initiation used.
            replacement (str): The replacement text for the removed text.
            label (str): Label for the personal data removed.
            priority (float): Priority score for personal data match (range 0 - 1).
            Higher scored matches are scrubbed when overlapping personal data found.

        Returns:
            list[IDEnt]: A list of IDEnt objects.
        """

        pattern = r"\b\S+@\S+\.\S+\b"
        return self.find_regex(
            texts=texts, text_ids=text_ids, pattern=pattern, label=label, replacement=replacement, priority=priority
        )

    def urls(
        self,
        texts: list[str] = None,
        text_ids: list = None,
        replacement: str = "[URL]",
        label: str = "url",
        priority: float = 0.3,
    ) -> list[IDEnt]:
        """
        Remove `http`, `https` and `www` URLs using regex
        e.g. `www.google.com` scrubbed.

        `example.com` will not be scrubbed by this method.

        Args:
            texts (list[str]): Strings to scrub.
            text_ids (list): A list of identifiers that correspond to each string in `texts`.
            If None, current cleaned state of `texts` passed at Class initiation used.
            replacement (str): The replacement text for the removed text.
            label (str): Label for the personal data removed.
            priority (float): Priority score for personal data match (range 0 - 1).
            Higher scored matches are scrubbed when overlapping personal data found.

        Returns:
            list[IDEnt]: A list of IDEnt objects.
        """

        pattern = r"\b(?:https?://|www\.)[^\s<>()\"']+"
        return self.find_regex(
            texts=texts, text_ids=text_ids, pattern=pattern, label=label, replacement=replacement, priority=priority
        )

    def handles(
        self,
        texts: list[str] = None,
        text_ids: list = None,
        replacement: str = "[HANDLE]",
        label: str = "handle",
        priority: float = 0.4,
    ) -> list[IDEnt]:
        """
        Remove `@` user handles using regex
        e.g. `@username` scrubbed

        Args:
            texts (list[str]): Strings to scrub.
            text_ids (list): A list of identifiers that correspond to each string in `texts`.
            If None, current cleaned state of `texts` passed at Class initiation used.
            replacement (str): The replacement text for the removed text.
            label (str): Label for the personal data removed.
            priority (float): Priority score for personal data match (range 0 - 1).
            Higher scored matches are scrubbed when overlapping personal data found.

        Returns:
            list[IDEnt]: A list of IDEnt objects.
        """

        pattern = r"@[\w.-]+(?=\b)"
        return self.find_regex(
            texts=texts, text_ids=text_ids, pattern=pattern, label=label, replacement=replacement, priority=priority
        )

    def google_phone_numbers(
        self,
        texts: list[str] = None,
        text_ids: list = None,
        region: str = "GB",
        replacement: str = "[PHONENO]",
        label: str = "phone_number",
        priority: float = 0.8,
    ) -> list[IDEnt]:
        """
        Remove phone numbers using Google's `phonenumbers`.
        e.g. `+441234567891` scrubbed

        Args:
            texts (list[str]): Strings to scrub.
            text_ids (list): A list of identifiers that correspond to each string in `texts`.
            If None, current cleaned state of `texts` passed at Class initiation used.
            region (str): The region to find phone numbers for. See `phonenumbers` regions.
            replacement (str): The replacement text for the removed text.
            label (str): Label for the personal data removed.
            priority (float): Priority score for personal data match (range 0 - 1).
            Higher scored matches are scrubbed when overlapping personal data found.

        Returns:
            list[IDEnt]: A list of IDEnt objects.
        """

        if self.replacement:
            replacement = self.replacement

        idents = []

        for text, text_id in zip(texts, text_ids):
            matches = list(phonenumbers.PhoneNumberMatcher(text, region))
            for match in matches:
                idents.append(
                    self.IDEnt(
                        text_id=text_id,
                        text=match.raw_string,
                        start=match.start,
                        end=match.end,
                        priority=priority,
                        replacement=replacement,
                        label="phone_no",
                        source="google_phone_numbers",
                    )
                )

        return idents

    def uk_phone_numbers(
        self,
        texts: list[str] = None,
        text_ids: list = None,
        replacement: str = "[PHONENO]",
        label: str = "uk_phone_number",
        priority: float = 0.8,
    ) -> list[IDEnt]:
        """
        Remove phone numbers using regex.
        e.g. `+441234567891` scrubbed

        Args:
            texts (list[str]): Strings to scrub.
            If None, current cleaned state of `text` passed at Class initiation used.
            replacement (str): The replacement text for the removed text.
            label (str): Label for the personal data removed.
            priority (float): Priority score for personal data match (range 0 - 1).
            Higher scored matches are scrubbed when overlapping personal data found.

        Returns:
            list[IDEnt]: A list of IDEnt objects.
        """

        pattern = r"(\+?\d[\d\s]{7,}\d)"
        return self.find_regex(
            texts=texts, text_ids=text_ids, pattern=pattern, label=label, replacement=replacement, priority=priority
        )

    def titles(
        self,
        texts: list[str] = None,
        text_ids: list = None,
        strict: bool = False,
        replacement: str = "[TITLE]",
        label: str = "title",
        priority: float = 0.4,
    ) -> list[IDEnt]:
        """
        Remove titles using regex.

        Args:
            texts (list[str]): Strings to scrub.
            text_ids (list): A list of identifiers that correspond to each string in `texts`.
            If None, current cleaned state of `text` passed at Class initiation used.
            strict (bool): Whether to use all of the titles or only essential titles.
            If strict, you may find scrubbing of common words, such as general.
            replacement (str): The replacement text for the removed text.
            label (str): Label for the personal data removed.
            priority (float): Priority score for personal data match (range 0 - 1).
            Higher scored matches are scrubbed when overlapping personal data found.

        Returns:
            list[IDEnt]: A list of IDEnt objects.
        """

        titles = [
            "Mr",
            "Mrs",
            "Miss",
            "Ms",
            "Mx",
            "Dr",
            "Doc",
            "Doctor",
            "Prof",
            "Professor",
            "Rev",
            "Revd",
            "Reverend",
            "Sir",
            "Dame",
            "Lord",
            "Lady",
            "Captain",
            "Major",
            "Colonel",
            "General",
            "Father",
            "Rabbi",
            "Imam",
            "Master",
            "Hon",
            "Senator",
            "Duke",
            "Duchess",
            "Prince",
            "Princess",
            "MP",
            "Rt Hon",
            "Rt. Hon",
            "Right Honourable",
            "Judge",
            "Madame",
            "Sister",
        ]

        if not strict:
            titles_to_remove = ["General", "Major", "Judge", "Master", "Father", "Sister", "Miss"]
            titles = [title for title in titles if title not in titles_to_remove]

        # Add dotted versions
        titles += [title + "." for title in titles]
        titles += [title + ":" for title in titles]

        pattern = r"\b(?:{})\b".format("|".join(re.escape(t) for t in titles))
        return self.find_regex(
            texts=texts, text_ids=text_ids, pattern=pattern, label=label, replacement=replacement, priority=priority
        )

    def ip_addresses(
        self,
        texts: list[str] = None,
        text_ids: list = None,
        replacement: str = "[IPADDRESS]",
        label: str = "ip_address",
        priority: float = 0.5,
    ) -> list[IDEnt]:
        """
        Removes IP addresses.
        e.g. `192.168.1.1` scrubbed

        Args:
            texts (list[str]): Strings to scrub.
            text_ids (list): A list of identifiers that correspond to each string in `texts`.
            If None, current cleaned state of `texts` passed at Class initiation used.
            replacement (str): The replacement text for the removed text.
            label (str): Label for the personal data removed.
            priority (float): Priority score for personal data match (range 0 - 1).
            Higher scored matches are scrubbed when overlapping personal data found.

        Returns:
            list[IDEnt]: A list of IDEnt objects.
        """

        pattern = r"(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"
        return self.find_regex(
            texts=texts, text_ids=text_ids, pattern=pattern, label=label, replacement=replacement, priority=priority
        )

    def uk_postcodes(
        self,
        texts: list[str] = None,
        text_ids: list = None,
        replacement: str = "[POSTCODE]",
        label: str = "uk_postcode",
        priority: float = 0.5,
    ) -> list[IDEnt]:
        """
        Removes postcodes.
        e.g. `A11 1AA` scrubbed

        Args:
            texts (list[str]): Strings to scrub.
            text_ids (list): A list of identifiers that correspond to each string in `texts`.
            If None, current cleaned state of `texts` passed at Class initiation used.
            replacement (str): The replacement text for the removed text.
            label (str): Label for the personal data removed.
            priority (float): Priority score for personal data match (range 0 - 1).
            Higher scored matches are scrubbed when overlapping personal data found.

        Returns:
            list[IDEnt]: A list of IDEnt objects.
        """

        pattern = r"\b(?:(?:[A-Z][A-HJ-Y]?[0-9][0-9A-Z]?)[ \t]*[0-9][A-Z]{2}|GIR[ \t]*0A{2}|SAN[ \t]*TA1|ASCN[ \t]*1ZZ|STHL[ \t]*1ZZ|TDCU[ \t]*1ZZ|BBND[ \t]*1ZZ|[BFS]IQ{2}[ \t]*1ZZ|GX11[ \t]*1AA|PCRN[ \t]*1ZZ|TKCA[ \t]*1ZZ|AI-?[0-9]{4}|BFPO[ \t-]?[0-9]{2,4}|MSR[ \t-]?1(?:1[12]|[23][135])0|VG[ \t-]?11[1-6]0|KY[1-3][ \t-]?[0-2][0-9]{3})\b"
        return self.find_regex(
            texts=texts, text_ids=text_ids, pattern=pattern, label=label, replacement=replacement, priority=priority
        )

    def uk_addresses(
        self,
        texts: list[str] = None,
        text_ids: list = None,
        replacement: str = "[ADDRESS]",
        label: str = "uk_address",
        priority: float = 0.8,
    ) -> list[IDEnt]:
        """
        Removes addresses.
        e.g. `10 Downing Street` scrubbed

        Args:
            texts (list[str]): Strings to scrub.
            text_ids (list): A list of identifiers that correspond to each string in `texts`.
            If None, current cleaned state of `texts` passed at Class initiation used.
            replacement (str): The replacement text for the removed text.
            label (str): Label for the personal data removed.
            priority (float): Priority score for personal data match (range 0 - 1).
            Higher scored matches are scrubbed when overlapping personal data found.


        Returns:
            list[IDEnt]: A list of IDEnt objects.
        """

        if self.texts and self.text_ids:
            texts = self.texts
            text_ids = self.text_ids
        else:
            texts = texts
            text_ids = text_ids

        pattern = r"(?i)\b(?:flat\s+\w+,\s*)?\d+[a-z]?(?:[-–/]\d+[a-z]?)?\s+[a-z][a-z'’\- ]+\s+(street|st|road|rd|avenue|ave|lane|ln|close|cl|drive|dr|way|walk|gardens|gdns|place|pl|mews|court|ct|crescent|cres|terrace|ter)\b"
        return self.find_regex(
            texts=texts, text_ids=text_ids, pattern=pattern, label=label, replacement=replacement, priority=priority
        )

    def get_spacy_model(self, model_name: str = "en_core_web_trf") -> Language:
        """
        Loads a SpaCy language model if it has been downloaded.
        If the model has not been downloaded, it is downloaded.
        Note:  Only `en_core_web_trf` has been evaluated.

        Args:
            model_name (str): Name of SpaCy model. Only `en_core_web_trf` has been evaluated.

        Returns:
            Language: A SpaCy Language object.

        """
        if model_name in ["en_core_web_trf", "en_core_web_lg", "en_core_web_md", "en_core_web_sm"]:
            try:
                model = spacy.load(model_name, disable=["parser", "tagger", "lemmatizer"])
            except (OSError, ValueError):
                self.logger.warning(
                    f"SpaCy model `{model_name}` not downloaded. Downloading...\n"
                    "⚠️ You may need to restart the kernel and rerun after download is complete ⚠️"
                )
                download(model_name)
                try:
                    model = spacy.load(model_name, disable=["parser", "tagger", "lemmatizer"])
                except (OSError, ValueError):
                    raise RuntimeError(
                        f"The model `{model_name}` has been downloaded but not found in the environment. To resolve you must restart the kernel and/or run again."
                    )
        else:
            raise ValueError(
                f"The model `{model_name}` is not appropriate for this task. Please use `en_core_web_trf` as it is been benchmarked for performance."
            )

        return model

    def spacy_entities(
        self,
        texts: list[str] = None,
        text_ids: list = None,
        model_name: str = "en_core_web_trf",
        entity_types: list[str] = ["PERSON", "ORG", "NORP"],
        replacement_map: dict = {"PERSON": "[PERSON]", "ORG": "[ORG]", "NORP": "[NORP]"},
        priority: float = 1.0,
        n_process: int = 1,
        batch_size: int = 1000,
    ) -> list[IDEnt]:
        """
        Remove SpaCy idents using a given SpaCy model.
        Documentation for entity labels: https://spacy.io/models/en#en_core_web_trf
        Note: only "en_core_web_trf" has been evaluated.

        Args:
            texts (list[str]): Strings to scrub.
            text_ids (list): A list of identifiers that correspond to each string in `texts`.
            If None, current cleaned state of `texts` passed at Class initiation used.
            model_name (str): Name of Spacy model. Only `en_core_web_trf` has been evaluated.
            entity_types (list[str]): Which SpaCy idents to scrub (based on SpaCy entity keys).
            replacement_map (str): The replacement texts for the removed text. Key is entity type, value is replacement.
            label_prefix (str): Prefix for the Spacy entity removed, e.g. `{label}_person`.
            n_process (int): Number of parallel processes.
            batch_size (int): The number of texts in each batch.
            priority (float): Priority score for personal data match (range 0 - 1).
            Higher scored matches are scrubbed when overlapping personal data found.

        Returns:
            list[IDEnt]: A list of IDEnt objects.
        """

        nlp = self.get_spacy_model(model_name)
        stripped_texts = [s.strip() if s.isspace() else s for s in texts]
        docs = nlp.pipe(stripped_texts, n_process=n_process, batch_size=batch_size)

        idents = []

        for doc, text_id in zip(docs, text_ids):
            for ent in doc.ents:
                if ent.label_ not in entity_types:
                    continue
                if self.replacement:
                    replacement = self.replacement
                elif replacement_map:
                    replacement = replacement_map.get(ent.label_, "[REDACTED]")
                else:
                    replacement = f"[{ent.label_}]"

                idents.append(
                    self.IDEnt(
                        text_id=text_id,
                        text=ent.text,
                        start=ent.start_char,
                        end=ent.end_char,
                        priority=priority,
                        replacement=replacement,
                        label=ent.label_.lower(),
                        source="spacy",
                    )
                )

        return idents

    def get_hf_model(
        self,
        hf_model_path: str = "dbmdz/bert-large-cased-finetuned-conll03-english",
        download_directory: str = f"{DOWNLOAD_DIR}/huggingface/",
    ) -> str:
        """
        Loads a Hugging Face model from the chosen directory if it has been downloaded.
        If the model has not been downloaded, it is downloaded to the chosen directory.
        Note: No Hugging Face models have been evaluated for performance.

        Args:
            hf_model_path (str): Path to the Hugging Face model.
            Only `dbmdz/bert-large-cased-finetuned-conll03-english` has been evaluated.
            download_directory (str): Directory in which to save the model.
            Default is current working directory.

        Returns:
            str: The path to the downloaded model.

        """

        if os.path.exists(download_directory):
            try:
                tokenizer = AutoTokenizer.from_pretrained(hf_model_path, cache_dir=download_directory)
            except HFValidationError as e:
                raise ValueError(
                    f"Hugging Face model `{hf_model_path}` has not been downloaded. Please check model path then retry.\n"
                    f"Full error message: {e}"
                )
        else:
            os.makedirs(download_directory, exist_ok=True)
            tokenizer = AutoTokenizer.from_pretrained(hf_model_path, cache_dir=download_directory)

        return tokenizer

    def huggingface_entities(
        self,
        texts: list[str] = None,
        text_ids: list = None,
        entity_type="PER",
        replacement: str = "[PERSON]",
        label: str = "person",
        priority: float = 1.0,
        hf_model_path: str = "dbmdz/bert-large-cased-finetuned-conll03-english",
        download_directory: str = f"{DOWNLOAD_DIR}/huggingface/",
    ) -> list[IDEnt]:
        """
        Remove idents using a Hugging Face model. Default is a PERSON entity identifier.
        Note: No Hugging Face models have been evaluated for performance.

        Args:
            texts (list[str]): Strings to scrub.
            text_ids (list): A list of identifiers that correspond to each string in `texts`.
            entity_type (str): Which entity to scrub (based on particular model keys).
            If None, current cleaned state of `texts` passed at Class initiation used.
            hf_model_path (str): Path to the Hugging Face model.
            Only `dbmdz/bert-large-cased-finetuned-conll03-english` has been tested.
            download_directory (str): Directory in which to save the model.
            Default is current working directory.
            replacement (str): The replacement text for the removed text.
            label (str): Label for the personal data removed.
            priority (float): Priority score for personal data match (range 0 - 1).
            Higher scored matches are scrubbed when overlapping personal data found.
            batch_size (int): Number of texts passed to the model in each batch.
            Memory (instance size) dependent.

        Returns:
            list[str]: The input list of text with PERSON idents replaced.

        """

        if self.replacement:
            replacement = self.replacement

        tokenizer = self.get_hf_model(hf_model_path=hf_model_path, download_directory=download_directory)

        try:
            names_model = AutoModelForTokenClassification.from_pretrained(hf_model_path)
        except OSError:
            raise RuntimeError(
                f"Hugging Face model `{hf_model_path}` does has not been downloaded correctly. Please delete `huggingface/` and retry."
            )

        ner = pipeline(task="ner", model=names_model, tokenizer=tokenizer, aggregation_strategy="simple")

        idents = []

        results = ner(texts)

        for ents, text_id in zip(results, text_ids):
            for ent in ents:
                if ent["entity_group"] != entity_type:
                    continue
                idents.append(
                    self.IDEnt(
                        text_id=text_id,
                        text=ent["word"],
                        start=ent["start"],
                        end=ent["end"],
                        priority=priority,
                        replacement=replacement,
                        label=label,
                        source="huggingface",
                    )
                )

        return idents

    def presidio_entities(
        self,
        texts: list[str] = None,
        text_ids: list = None,
        model_name: str = "en_core_web_trf",
        entity_types: list[str] = [
            "PERSON",
            "EMAIL_ADDRESS",
            "UK_NINO",
            "UK_NHS",
            "CREDIT_CARD",
            "CRYPTO",
            "MEDICAL_LICENSE",
            "SWIFT_CODE",
            "IBAN_CODE",
            "LOCATION",
            "NRP",
        ],
        replacement_map: dict = {},
        priority: float = 1.0,
    ) -> list[IDEnt]:
        """
        Scrub specified idents from texts using Presidio.

        See https://microsoft.github.io/presidio/supported_entities/ for further detail.

        Args:
            texts (list[str]): Strings to scrub.
            text_ids (list): A list of identifiers that correspond to each string in `texts`.
            If None, current cleaned state of `texts` passed at Class initiation used.
            model_name (str): spaCy model to use
            entity_types (list[str]): entity types to scrub (e.g. ["PERSON", "IP_ADDRESS"])
            replacement_map (str): The replacement texts for the removed text. Key is entity type, value is replacement.
            priority (float): Priority score for personal data match (range 0 - 1).
            Higher scored matches are scrubbed when overlapping personal data found.

        Returns:
            list[str]: The input list of text with idents replaced.
        """

        class LoadedSpacyNlpEngine(SpacyNlpEngine):
            def __init__(self, loaded_spacy_model):
                super().__init__()
                self.nlp = {"en": loaded_spacy_model}

        nlp = self.get_spacy_model(model_name)
        loaded_nlp_engine = LoadedSpacyNlpEngine(loaded_spacy_model=nlp)

        analyzer = AnalyzerEngine(nlp_engine=loaded_nlp_engine)

        idents = []

        for text, text_id in zip(texts, text_ids):
            results = analyzer.analyze(text=text, language="en", entities=entity_types)
            for res in results:
                if res.entity_type not in entity_types:
                    continue

                if self.replacement:
                    replacement = self.replacement
                elif replacement_map:
                    replacement = replacement_map.get(res.entity_type, "[REDACTED]")
                else:
                    replacement = f"[{res.entity_type}]"

                idents.append(
                    self.IDEnt(
                        text_id=text_id,
                        text=text[res.start : res.end],
                        start=res.start,
                        end=res.end,
                        priority=priority,
                        replacement=replacement,
                        label=res.entity_type.lower(),
                        source="presidio",
                    )
                )

        return idents

    def group_idents(self, idents: list[IDEnt]) -> dict[int | str | float, list[IDEnt]]:
        """
        Group a list of IDEnt objects by `text_id`.

        Each unique `text_id` becomes a dictionary key,
        and its value is a list of all IDEnt objects associated with that ID.

        Args:
            idents (list[IDEnt]) A list of IDEnt objects.

        Returns:
            dict[int | str | float, list[IDEnt]]: A dictionary mapping each text_id to a list of IDEnt objects.
        """

        idents_grouped = defaultdict(list)

        for ident in idents:
            idents_grouped[ident.text_id].append(ident)

        return idents_grouped

    def resolve_overlaps(self, idents: list[IDEnt]) -> list[IDEnt]:
        """
        Select the highest-scoring non-overlapping idents.

        Resolves conflicts between idents that overlap in their
        character ranges. Entities are first sorted by descending priority and then by
        start position to ensure a priority order.

        Each IDEnt is accepted only if it does not overlap with any IDEnt
        already selected. The resulting set of idents is returned in ascending
        document order.

        A IDEnt is considered overlapping if:
            IDEnt.start <= other.end  and  IDEnt.end >= other.start

        Args:
            idents (list[IDEnt]) A list of IDEnt objects.

        Returns:
            list[IDEnt]: A list of non-overlapping idents, sorted by their start position.
        """

        idents_grouped = self.group_idents(idents)

        resolved = []

        for text_id, idents in idents_grouped.items():
            if not idents:
                return []

            idents_by_score = sorted(idents, key=lambda ident: (-ident.priority, ident.start))

            kept_idents = []

            for current_ident in idents_by_score:
                has_overlap = any(
                    current_ident.start <= existing_ident.end and current_ident.end >= existing_ident.start
                    for existing_ident in kept_idents
                )

                if not has_overlap:
                    kept_idents.append(current_ident)

            resolved.extend(kept_idents)

        return resolved

    def scrub_text(self, texts: str = None, text_ids: list = None, idents: list[IDEnt] = None):
        """
        Apply a set of non-overlapping replacement idents to a text.

        Each IDEnt specifies a character range to replace (`IDEnt.start` to `IDEnt.end`)
        and a `replacement` string that will be inserted in place of that range.

        Args:
            texts list[str]: The original input text with overlaps resolved.
            text_ids (list): A list of identifiers that correspond to each string in `texts`.
            idents list[IDEnt]: a list of IDEnt objects. Must be non-overlapping.
            See `resolve_conflicts`.

        Return:
            str: A scrubbed string with all replacements applied.
        """

        if texts is None:
            texts = getattr(self, "texts", None)
        if text_ids is None:
            text_ids = getattr(self, "text_ids", None)
        if idents is None:
            idents = getattr(self, "idents", None)

        if texts is None or text_ids is None or idents is None:
            raise ValueError("texts, text_ids, and idents must be provided or set on self.")

        if len(texts) != len(text_ids):
            raise ValueError("texts and text_ids must be the same length.")

        scrubbed_texts = list(texts)
        idents_grouped = self.group_idents(idents)

        for i, text_id in enumerate(text_ids):
            text = texts[i]

            group = idents_grouped.get(text_id, [])
            sorted_group = sorted(group, key=lambda ident: ident.start, reverse=True)

            for ident in sorted_group:
                text = text[: ident.start] + ident.replacement + text[ident.end :]

            scrubbed_texts[i] = text

        return scrubbed_texts

    def scrub(
        self,
        pipeline: list[dict] = [
            {"method": "presidio_entities"},
            {"method": "spacy_entities"},
            {"method": "email_addresses"},
            {"method": "handles"},
            {"method": "ip_addresses"},
            {"method": "uk_addresses"},
            {"method": "uk_phone_numbers"},
            {"method": "google_phone_numbers"},
            {"method": "uk_postcodes"},
            {"method": "urls"},
            {"method": "titles"},
        ],
    ):
        """
        Scrubs text using given methods.
        Uses default values for the given scrub method.

        Args:
            pipeline (list[dict]): Scrub methods and their method parameters to apply.
            Methods are specified with "method" key.
            Parameters are specified with argument name as "key" and argument value as value.

            Example: IDScrub.scrub(pipeline=[{"method": "google_phone_numbers", "region": "GB"}])

            Methods available (see associated method docstring for further parameters):
            "spacy_entities", "huggingface_entities", "email_addresses", "handles",
            "ip_addresses", "uk_addresses", "uk_phone_numbers", "google_phone_numbers", "uk_postcodes"
            "titles", "presidio_entities"

        Returns:
            list[str]: The input texts scrubbed of personal data.

        """

        if not isinstance(pipeline, list):
            raise TypeError("Argument `pipeline` must be a list of dicts.")

        self.idents_all = []
        self.idents = []

        for step in pipeline:
            scrub_method = step["method"]
            args = {k: v for k, v in step.items() if k != "method"}

            if args:
                self.logger.info(f"Scrubbing using {scrub_method} with parameters {args}...")
            else:
                self.logger.info(f"Scrubbing using {scrub_method} with default parameters...")

            try:
                method = getattr(self, scrub_method)
            except AttributeError:
                self.logger.warning("Not a scrub method.")

            self.idents_all.extend(method(texts=self.texts, text_ids=self.text_ids, **args))

        idents_exclude = [ident for ident in self.idents_all if ident.text not in self.exclude]
        idents_resolved = self.resolve_overlaps(idents=idents_exclude)
        self.idents.extend(idents_resolved)
        self.scrubbed_texts = self.scrub_text(texts=self.texts, text_ids=self.text_ids, idents=self.idents)

        return self.scrubbed_texts

    def get_all_identified_data(self) -> pd.DataFrame:
        """
        Get all of the identified data before overlaps have been resolved.

        Each row is a identified entity. Columns are the IDEnt attributes.

        Args:
            None
        Return:
            pd.DataFrame: All identified data and their attributes.
        """
        all_idents = pd.DataFrame([asdict(ident) for ident in self.idents_all])
        return all_idents

    def get_scrubbed_data(self) -> pd.DataFrame:
        """
        Create a DataFrame summarising scrubbed text idents grouped by text ID and label.

        Each row corresponds to a unique `text_id`, and each column represents a IDEnt label.
        The cell values are lists of the IDEnt text values associated with that label for the given text ID.
        Args:
            None
        Return:
            pd.DataFrame: All data scrubbed from text.
        """
        data = defaultdict(lambda: defaultdict(list))

        for ident in self.idents:
            data[ident.text_id][ident.label].append(ident.text)

        df = pd.DataFrame.from_dict(data, orient="index")
        df = df.reset_index().rename(columns={"index": self.text_id_name})
        df = df.where(pd.notna(df), None)

        return df

    @staticmethod
    def dataframe(
        df: pd.DataFrame = None,
        id_col: str = None,
        exclude_cols: list = None,
        pipeline: list[dict] = [
            {"method": "presidio_entities"},
            {"method": "spacy_entities"},
            {"method": "email_addresses"},
            {"method": "handles"},
            {"method": "ip_addresses"},
            {"method": "uk_addresses"},
            {"method": "uk_phone_numbers"},
            {"method": "google_phone_numbers"},
            {"method": "uk_postcodes"},
            {"method": "urls"},
            {"method": "titles"},
        ],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Scrubs all personal data from a Pandas Dataframe.

        Args:
            df (pd.DataFrame): A Pandas dataframe to scrub.
            id_col (str): Name of the ID column in `df`. If None, an integer index starting at 1  with the name `text_id` is applied.
            exclude_cols (list): Columns to exclude from scrubbing. if None all columns are scrubbed.
            pipeline (list[dict]): Scrub methods and their method parameters to apply.
            Methods are specified with "method" key.
            Parameters are specified with argument name as "key" and argument value as value.

            Example: IDScrub.scrub(pipeline=[{"method": "google_phone_numbers", "region": "GB"}])

            Methods available (see associated method docstring for further parameters):
            "spacy_entities", "huggingface_entities", "email_addresses", "handles",
            "ip_addresses", "uk_addresses", "uk_phone_numbers", "google_phone_numbers", "uk_postcodes"
            "titles", "presidio_entities"

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: The input dataframe with all personal data removed and a dataframe with the personal data that has been removed.

        """

        if not isinstance(df, pd.DataFrame):
            raise TypeError("`df` must be a Pandas DataFrame.")

        if id_col is None:
            ids = range(1, len(df) + 1)
            id_col = "id"
        else:
            if id_col not in df.columns:
                raise ValueError(f"`id_col` '{id_col}' is not a column in df.")

            ids = df[id_col].tolist()

        if not len(df) == len(ids):
            raise ValueError("Length of dataframe is different to the length of IDs.")

        if exclude_cols is None:
            cols_to_scrub = df.columns.to_list()
        else:
            cols_to_scrub = [col for col in df.columns if col not in exclude_cols]

        cols_to_scrub.remove(id_col)

        scrubbed_df = df.copy()

        all_scrubbed_data = []

        for col in tqdm(cols_to_scrub):
            original_dtype = scrubbed_df[col].dtype
            scrubbed_df[col] = scrubbed_df[col].astype(str)

            scrub = IDScrub(texts=scrubbed_df[col].to_list(), text_ids=ids)
            scrub.logger.info(f"Scrubbing column `{col}`...")

            scrubbed_texts = scrub.scrub(pipeline=pipeline)
            scrubbed_df[col] = scrubbed_texts

            scrubbed_data = scrub.get_scrubbed_data()

            if scrubbed_data is not None:
                scrubbed_data.insert(1, "column", col)
                scrubbed_data.rename(columns={"text_id": id_col}, inplace=True)
                all_scrubbed_data.append(scrubbed_data)

            try:
                scrubbed_df[col] = scrubbed_df[col].astype(original_dtype)
            except ValueError:
                # If dtype is not revertable because it has been scrubbed, then pass
                pass

        all_scrubbed_data = pd.concat(all_scrubbed_data).reset_index(drop=True)
        all_scrubbed_data["column"] = pd.Categorical(
            all_scrubbed_data["column"], categories=cols_to_scrub, ordered=True
        )
        all_scrubbed_data = all_scrubbed_data.sort_values(by=["column", id_col]).reset_index(drop=True)
        all_scrubbed_data["column"] = all_scrubbed_data["column"].astype(str)
        all_scrubbed_data = all_scrubbed_data.where(pd.notna(all_scrubbed_data), None)

        if not df.shape == scrubbed_df.shape:
            raise ValueError("Original and scrubbed dataframe not the same shape. Check input DataFrame.")

        return scrubbed_df, all_scrubbed_data
