import logging
import os
import re
import warnings
from collections.abc import Iterable
from functools import partial

import pandas as pd
import phonenumbers
import spacy
from huggingface_hub.utils import HFValidationError
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import SpacyNlpEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from spacy.cli import download
from spacy.language import Language
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
from transformers.utils import logging as trf_logging

from idscrub.locations import DOWNLOAD_DIR, PROJECT_DIR

# Suppress Torch FutureWarning
# TODO: Find better way
warnings.simplefilter(action="ignore", category=FutureWarning)

trf_logging.set_verbosity_error()


class IDScrub:
    def __init__(
        self,
        texts: list[str] = [],
        text_ids: list | Iterable = None,
        text_id_name: str = "text_id",
        replacement_text: str = None,
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
            replacement_text (str): A global string to replace every scrubbed
            string with.
            verbose (bool): Whether to show all log messages or only warnings.
        """

        assert isinstance(texts, list) and all(isinstance(text, str) for text in texts), (
            "`texts` can only be a list of strings or a single string in a list."
        )

        assert isinstance(replacement_text, str) or isinstance(replacement_text, type(None)), (
            "`replacement_text` can only be string."
        )

        self.texts = texts

        if text_ids:
            self.text_ids = text_ids
        else:
            self.text_ids = range(1, len(self.texts) + 1)

        assert len(self.texts) == len(self.text_ids), "Length of texts is different to the length of text IDs."

        self.text_id_name = text_id_name
        self.cleaned_texts = []
        self.scrubbed_data = []
        self.replacement_text = replacement_text

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG if verbose else logging.WARNING)

        if not self.logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(levelname)s: %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.logger.info("Texts loaded.")

    def get_texts(self) -> list[str]:
        """
        Get the text that will be processed.
        If no cleaning has occured, then use the raw input
        texts. If cleaning has occured then update the cleaned texts.

        Args:
            None

        Returns:
            A Pandas DataFrame with text_id
            and scrubbed in a list format.
        """
        if self.cleaned_texts:
            texts = self.cleaned_texts
        else:
            texts = self.texts

        return texts

    def get_scrubbed_data(self) -> pd.DataFrame:
        """
        Turn text ids and scrubbed text into a DataFrame.

        Args:
            None

        Returns:
            A Pandas DataFrame with text_id
            and scrubbed in a list format.
        """
        df = pd.DataFrame(self.scrubbed_data)

        if self.text_id_name not in df.columns:
            return None

        # Group by the id and aggregate non-null values into lists
        if df[self.text_id_name].dtype == object or df[self.text_id_name].dtype == str:
            grouped = (
                df.groupby(self.text_id_name, sort=False)
                .agg(lambda x: [i for i in x if pd.notna(i)])
                .reset_index()
                .map(lambda x: None if isinstance(x, list) and len(x) == 0 else x)
            )
        else:
            grouped = (
                df.groupby(self.text_id_name)
                .agg(lambda x: [i for i in x if pd.notna(i)])
                .reset_index()
                .map(lambda x: None if isinstance(x, list) and len(x) == 0 else x)
            )

        return grouped

    def log_message(self, removed_label) -> None:
        """
        Log message with count of PII-type scrubbed.

        Args:
            removed_label (str): Label for the PII-type removed. Must be prefixed with `scrubbed_`.
        Returns:
            int: The count of PII-type scrubbed.
        """

        if any(removed_label in key for key in self.scrubbed_data):
            scrubbed_data = self.get_scrubbed_data()
            count = scrubbed_data[removed_label].dropna().apply(len).sum()
        else:
            count = 0

        label_name = removed_label.removeprefix("scrubbed_").replace("_", " ")
        self.logger.info(f"{count} {label_name} scrubbed.")

        return count

    def scrub_and_collect(self, match, text, replacement_text, i, removed_label) -> str:
        """
        Scrub pattern match and collect scrubbed name.

        Args:
            match (str): The regex match passed from `re.sub()`.
            i (int): the enumerate id of the string.
            removed_label (str): Label for the PII-type removed. Must be prefixed with `scrubbed_`.

        Returns:
            str: The replacement text.
        """

        self.scrubbed_data.append({self.text_id_name: i, removed_label: match.group()})

        return replacement_text

    def scrub_regex(self, pattern, replacement_text, removed_label) -> list[str]:
        """
        General method to clean text using a regex pattern.

        Args:
            pattern (str): Regex pattern to apply.
            replacement_text (str): The replacement text for the removed text.
            removed_label (str): Label for the PII-type removed. Must be prefixed with `scrubbed_`.

        Returns:
            list[str]: Cleaned texts.
        """

        texts = self.get_texts()

        compiled_pattern = re.compile(pattern, flags=re.IGNORECASE)

        if self.replacement_text:
            replacement_text = self.replacement_text

        cleaned_texts = [
            compiled_pattern.sub(
                partial(
                    self.scrub_and_collect,
                    text=text,
                    replacement_text=replacement_text,
                    i=i,
                    removed_label=removed_label,
                ),
                text,
            )
            for i, text in zip(self.text_ids, texts)
        ]

        self.cleaned_texts = cleaned_texts

        self.log_message(removed_label)

        return cleaned_texts

    def custom_regex(
        self,
        custom_regex_patterns: list[str] = None,
        custom_replacement_texts: list[str] = None,
    ) -> list[str]:
        """
        Remove text matching a custom regex pattern.

        Args:
            custom_regex_patterns list[str]: Regex(s) pattern to apply.
            custom_replacement_texts list[str]: The replacement texts for the removed text.
            Defaults to '[REDACTED]' for all.

        Returns:
            list[str]: Cleaned texts.

        """
        self.logger.info("Scrubbing custom regex...")

        if custom_replacement_texts:
            assert len(custom_regex_patterns) == len(custom_replacement_texts), (
                "There must be a replacement text for each pattern."
            )
        else:
            custom_replacement_texts = ["[REDACTED]"] * len(custom_regex_patterns)

        for i, (pattern, replacement_text) in enumerate(zip(custom_regex_patterns, custom_replacement_texts)):
            self.scrub_regex(pattern, replacement_text, removed_label=f"scrubbed_custom_regex_{i + 1}")

        return self.cleaned_texts

    def email_addresses(self, replacement_text="[EMAIL_ADDRESS]") -> list[str]:
        """
        Remove email addresses using regex.
        e.g. `johnsmith@gmail.com` scrubbed

        Args:
            replacement_text (str): The replacement text for the removed text.

        Returns:
            list[str]: The input list of text with email addresses replaced.
        """

        self.logger.info("Scrubbing email addresses using regex...")
        pattern = r"\b\S+@\S+\.\S+\b"

        return self.scrub_regex(pattern, replacement_text, removed_label="scrubbed_email_addresses")

    def handles(self, replacement_text: str = "[HANDLE]") -> list[str]:
        """
        Remove `@` user handles using regex
        e.g. `@username` scrubbed

        Args:
            replacement_text (str): The replacement text for the removed text.

        Returns:
            list[str]: The input list of text with handles replaced.
        """

        self.logger.info("Scrubbing @user handles using regex...")
        pattern = r"@[\w.-]+(?=\b)"

        return self.scrub_regex(pattern, replacement_text, removed_label="scrubbed_handles")

    def google_phone_numbers(self, region: str = "GB", replacement_text: str = "[PHONENO]") -> list[str]:
        """
        Remove phone numbers using Google's `phonenumbers`.
        e.g. `+441234567891` scrubbed

        Args:
            region (str): The region to find phone numbers for. See `phonenumbers` regions.
            replacement_text (str): The replacement text for the removed text.

        Returns:
            list[str]: The input list of text with phone numbers replaced.
        """

        self.logger.info(f"Scrubbing {region} phone numbers using Google's `phonenumbers`...")

        texts = self.get_texts()

        if self.replacement_text:
            replacement_text = self.replacement_text

        cleaned_texts = []

        for i, text in zip(self.text_ids, texts):
            matches = list(phonenumbers.PhoneNumberMatcher(text, region))
            phone_nos = [match.raw_string for match in matches]

            for phone_no in phone_nos:
                self.scrubbed_data.append({self.text_id_name: i, "scrubbed_phone_numbers": phone_no})

            cleaned = text
            for match in reversed(matches):
                cleaned = cleaned[: match.start] + replacement_text + cleaned[match.end :]

            cleaned_texts.append(cleaned)

        self.cleaned_texts = cleaned_texts

        self.log_message(f"scrubbed_{region.lower()}_phone_numbers")

        return cleaned_texts

    def uk_phone_numbers(self, replacement_text: str = "[PHONENO]") -> list[str]:
        """
        Remove phone numbers using regex.
        e.g. `+441234567891` scrubbed

        Args:
            replacement_text (str): The replacement text for the removed text.

        Returns:
            list[str]: The input list of text with phone numbers replaced.
        """

        self.logger.info("Scrubbing phone numbers using regex...")
        pattern = r"(\+?\d[\d\s]{7,}\d)"

        return self.scrub_regex(pattern, replacement_text, removed_label="scrubbed_uk_phone_numbers")

    def titles(self, strict: bool = False, replacement_text: str = "[TITLE]") -> list[str]:
        """
        Remove titles using regex.

        Args:
            strict (bool): Whether to use all of the titles or only essential titles.
            If strict, you may find scrubbing of common words, such as general.
            replacement_text (str): The replacement text for the removed text.

        Returns:
            list[str]: The input list of text with names after titles replaced.
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
            titles_to_remove = ["General", "Major", "Judge", "Master", "Father", "Sister"]
            titles = [title for title in titles if title not in titles_to_remove]

        # Add dotted versions
        titles += [title + "." for title in titles]
        titles += [title + ":" for title in titles]

        self.logger.info("Scrubbing titles using regex...")
        pattern = r"\b(?:{})\b".format("|".join(re.escape(t) for t in titles))

        return self.scrub_regex(pattern, replacement_text, removed_label="scrubbed_titles")

    def ip_addresses(self, replacement_text: str = "[IPADDRESS]") -> list[str]:
        """
        Removes IP addresses.
        e.g. `192.168.1.1` scrubbed

        Args:
            replacement_text (str): The replacement text for the removed text.

        Returns:
            list[str]: The input list of text with IP addresses replaced.
        """

        self.logger.info("Scrubbing IP addresses using regex...")
        pattern = r"(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"

        return self.scrub_regex(pattern, replacement_text, removed_label="scrubbed_ip_addresses")

    def uk_postcodes(self, replacement_text: str = "[POSTCODE]") -> list[str]:
        """
        Removes UK postcodes.
        e.g. `A11 1AA` scrubbed

        Args:
            replacement_text (str): The replacement text for the removed text.

        Returns:
            list[str]: The input list of text with postcodes replaced.
        """

        self.logger.info("Scrubbing UK postcodes using regex...")
        pattern = r"\b(?:(?:[A-Z][A-HJ-Y]?[0-9][0-9A-Z]?)[ \t]*[0-9][A-Z]{2}|GIR[ \t]*0A{2}|SAN[ \t]*TA1|ASCN[ \t]*1ZZ|STHL[ \t]*1ZZ|TDCU[ \t]*1ZZ|BBND[ \t]*1ZZ|[BFS]IQ{2}[ \t]*1ZZ|GX11[ \t]*1AA|PCRN[ \t]*1ZZ|TKCA[ \t]*1ZZ|AI-?[0-9]{4}|BFPO[ \t-]?[0-9]{2,4}|MSR[ \t-]?1(?:1[12]|[23][135])0|VG[ \t-]?11[1-6]0|KY[1-3][ \t-]?[0-2][0-9]{3})\b"

        return self.scrub_regex(pattern, replacement_text, removed_label="scrubbed_uk_postcodes")

    def claimants(self, replacement_text="[CLAIMANT]") -> list[str]:
        """
        Removes claimant names from employment tribunal texts.
        e.g. `Claimant: Jim Smith` scrubbed

        Args:
            None
        Returns:
            list[str]: The input list of text with claimants replaced.
        """

        self.logger.info("Scrubbing claimants using regex...")

        texts = self.get_texts()

        claimant_name = None

        cleaned_texts = []

        for i, text in zip(self.text_ids, texts):

            def replace_claimant(match):
                nonlocal claimant_name
                claimant_name = match.group(2).strip()
                return f"{match.group(1)}[CLAIMANT] "

            cleaned = re.sub(r"[\r\n]", " ", text)

            cleaned = re.sub(r"(Claimant\s*:\s*)(.*?)(?=\bRespondents?\s*:)", replace_claimant, cleaned)

            if claimant_name:
                cleaned = re.sub(re.escape(claimant_name), replacement_text, cleaned)
                self.scrubbed_data.append({self.text_id_name: i, "scrubbed_claimants": claimant_name})
                # self.scrubbed_data[self.text_id_name].append(i)
                # self.scrubbed_data['scrubbed_claimant'].append(claimant_name)

            cleaned_texts.append(cleaned)

        self.cleaned_texts = cleaned_texts

        return cleaned_texts

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

    def spacy_persons(
        self,
        model_name: str = "en_core_web_trf",
        n_process: int = 1,
        batch_size: int = 1000,
        replacement_text: str = "[PERSON]",
    ) -> list[str]:
        """
        Remove PERSON entities using a Spacy model.
        Note: only "en_core_web_trf" has been evaluated.

        Args:
            model_name (str): Name of Spacy model. Only `en_core_web_trf` has been evaluated.
            n_process (int): Number of parallel processes.
            batch_size (int): The number of texts in each batch.
            replacement_text (str): The replacement text for the removed text.

        Returns:
            list[str]: The input list of text with PERSON entities scrubbed.
        """
        self.logger.info(f"Scrubbing names using SpaCy model `{model_name}`...")

        texts = self.get_texts()

        if self.replacement_text:
            replacement_text = self.replacement_text

        cleaned_texts = []

        nlp = self.get_spacy_model(model_name)
        stripped_texts = [s.strip() if s.isspace() else s for s in texts]
        documents = nlp.pipe(stripped_texts, n_process=n_process, batch_size=batch_size)

        for i, (ids, doc, stripped_text) in tqdm(
            enumerate((zip(self.text_ids, documents, stripped_texts))), total=len(texts)
        ):
            if stripped_text == "":
                cleaned_texts.append(texts[i])
                continue

            # Collect person entities
            person_entities = [
                ent for ent in doc.ents if ent.label_ == "PERSON" and ent.text not in {"PERSON", "HANDLE"}
            ]
            self.scrubbed_data.extend(
                {self.text_id_name: ids, "scrubbed_spacy_person": ent.text} for ent in person_entities
            )

            # Remove person entities
            cleaned = stripped_text
            for ent in sorted(person_entities, key=lambda x: [x.start_char], reverse=True):
                cleaned = cleaned[: ent.start_char] + replacement_text + cleaned[ent.end_char :]

            cleaned_texts.append(cleaned)

        self.cleaned_texts = cleaned_texts

        self.log_message("scrubbed_spacy_person")

        return cleaned_texts

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
            hf_model_path (str): Path to the Hugging Face model on the DBT mirror.
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

    def huggingface_persons(
        self,
        hf_model_path: str = "dbmdz/bert-large-cased-finetuned-conll03-english",
        download_directory: str = f"{DOWNLOAD_DIR}/huggingface/",
        replacement_text: str = "[PERSON]",
        batch_size: int = 8,
    ) -> list[str]:
        """
        Remove PERSON entities using a Hugging Face model.
        Note: No Hugging Face models have been evaluated for performance.

        Args:
            hf_model_path (str): Path to the Hugging Face model on the DBT mirror.
            Only `dbmdz/bert-large-cased-finetuned-conll03-english` has been tested.
            download_directory (str): Directory in which to save the model.
            Default is current working directory.
            replacement_text (str): The replacement text for the removed text.
            batch_size (int): Number of texts passed to the model in each batch.
            Memory (instance size) dependent.

        Returns:
            list[str]: The input list of text with PERSON entities replaced.

        """

        self.logger.info(f"Scrubbing names using Hugging Face model ({hf_model_path})...")

        tokenizer = self.get_hf_model(hf_model_path=hf_model_path, download_directory=download_directory)

        texts = self.get_texts()

        if self.replacement_text:
            replacement_text = self.replacement_text

        try:
            names_model = AutoModelForTokenClassification.from_pretrained(hf_model_path)
        except OSError:
            raise RuntimeError(
                f"Hugging Face model `{hf_model_path}` does has not been downloaded correctly. Please delete `huggingface/` and retry."
            )

        ner_pipeline = pipeline("ner", model=names_model, tokenizer=tokenizer, aggregation_strategy="simple")
        stripped_texts = [s.strip() if s.isspace() else s for s in texts]
        batched_entities = ner_pipeline(stripped_texts, batch_size=batch_size)

        cleaned_texts = []

        for i, (ids, stripped_text, entities) in enumerate(zip(self.text_ids, stripped_texts, batched_entities)):
            if stripped_text == "":
                cleaned_texts.append(texts[i])
                continue

            person_entities = [
                ent for ent in entities if ent["entity_group"] == "PER" and ent["word"] not in {"HANDLE", "PERSON"}
            ]
            self.scrubbed_data.extend(
                {self.text_id_name: ids, "scrubbed_hf_person": ent["word"]} for ent in person_entities
            )

            cleaned = stripped_text
            for ent in sorted(person_entities, key=lambda x: x["start"], reverse=True):
                cleaned = cleaned[: ent["start"]] + replacement_text + cleaned[ent["end"] :]

            cleaned_texts.append(cleaned)

        self.cleaned_texts = cleaned_texts

        self.log_message("scrubbed_hf_person")

        return cleaned_texts

    def presidio(
        self,
        model_name="en_core_web_trf",
        entities_to_scrub=[
            "PERSON",
            "UK_NINO",
            "UK_NHS",
            "CREDIT_CARD",
            "CRYPTO",
            "MEDICAL_LICENSE",
            "URL",
            "IBAN_CODE",
        ],
        replacement_map=None,
    ) -> list[str]:
        """
        Scrub specified entities from texts using Presidio.

        See https://microsoft.github.io/presidio/supported_entities/ for further detail.

        Args:
            model_name (str): spaCy model to use
            entities_to_scrub (list[str]): Entity types to scrub (e.g. ["PERSON", "IP_ADDRESS"])
            replacement_map (dict): Mapping of entity_type to replacement string (e.g. {'PERSON': '[PERSON]'})

        Returns:
            list[str]: The input list of text with entities replaced.
        """

        self.logger.info("Scrubbing using Presidio...")

        texts = self.get_texts()

        cleaned_texts = []

        class LoadedSpacyNlpEngine(SpacyNlpEngine):
            def __init__(self, loaded_spacy_model):
                super().__init__()
                self.nlp = {"en": loaded_spacy_model}

        nlp = self.get_spacy_model(model_name)
        loaded_nlp_engine = LoadedSpacyNlpEngine(loaded_spacy_model=nlp)

        analyzer = AnalyzerEngine(nlp_engine=loaded_nlp_engine)
        anonymizer = AnonymizerEngine()

        cleaned_texts = []
        unique_labels = []

        stripped_texts = [s.strip() if s.isspace() else s for s in texts]

        for i, (ids, stripped_text) in tqdm(enumerate(zip(self.text_ids, stripped_texts)), total=len(texts)):
            if stripped_text == "":
                cleaned_texts.append(texts[i])
                continue

            results = analyzer.analyze(text=stripped_text, language="en")
            results = [r for r in results if r.entity_type in entities_to_scrub]

            labels = [f"scrubbed_presidio_{res.entity_type.lower()}" for res in results]
            unique_labels.append(list(set(labels)))

            self.scrubbed_data.extend(
                {self.text_id_name: ids, label: stripped_text[res.start : res.end]}
                for res, label in zip(results, labels)
            )

            if self.replacement_text:
                operators = {
                    res.entity_type: OperatorConfig("replace", {"new_value": self.replacement_text}) for res in results
                }
            elif replacement_map:
                operators = {
                    res.entity_type: OperatorConfig("replace", {"new_value": replacement_map.get(res.entity_type)})
                    for res in results
                }
            else:
                operators = {
                    res.entity_type: OperatorConfig("replace", {"new_value": f"[{res.entity_type}]"}) for res in results
                }

            anonymized = anonymizer.anonymize(text=stripped_text, analyzer_results=results, operators=operators)

            cleaned_texts.append(anonymized.text)

        self.cleaned_texts = cleaned_texts

        for label in unique_labels:
            if label:
                self.log_message(label[0])

        return cleaned_texts

    def all_regex(self) -> list[str]:
        """
        Use all regex methods to remove personal information from text.

        Args:
            None

        Returns:
            list[str]: The input list of text with various personal information replaced.

        """

        self.email_addresses()
        self.handles()
        self.ip_addresses()
        self.uk_phone_numbers()
        self.uk_postcodes()
        self.titles()

        return self.cleaned_texts

    def all(
        self,
        custom_regex_patterns: list = None,
        custom_replacement_texts: list[str] = None,
        model_name: str = "en_core_web_trf",
        presidio_entities_to_scrub: list[str] = [
            "PERSON",
            "EMAIL_ADDRESS",
            "UK_NINO",
            "UK_NHS",
            "CREDIT_CARD",
            "CRYPTO",
            "MEDICAL_LICENSE",
            "URL",
            "SWIFT_CODE",
            "IBAN_CODE",
            "LOCATION",
            "NRP",
        ],
        n_process: int = 1,
        batch_size: int = 1000,
    ) -> list[str]:
        """
        Use all regex and NER (Spacy) methods to remove personal information from text.

        Args:
            custom_regex_patterns list[str]: Regex(s) pattern to apply.
            custom_replacement_texts list[str]: The replacement texts for the removed text. Defaults to '[REDACTED]' for all.
            model_name (str): Name of Spacy model. Only `en_core_web_trf` has been evaluated.
            n_process (str): Number of parallel processes.
            batch_size (int): The number of texts in each batch.

        Returns:
            list[str]: The input list of text with various personal information replaced.
        """

        if custom_regex_patterns:
            self.custom_regex(
                custom_regex_patterns=custom_regex_patterns,
                custom_replacement_texts=custom_replacement_texts,
            )

        self.presidio(model_name=model_name, entities_to_scrub=presidio_entities_to_scrub)
        self.spacy_persons(model_name=model_name, n_process=n_process, batch_size=batch_size)
        self.google_phone_numbers()
        self.all_regex()

        return self.cleaned_texts

    def call_scrub_method(self, scrub_method: str) -> list[str]:
        """
        Calls a given scrub method based on its matching string name.
        Uses default values for the given scrub method.

        Example:
        "all" == scrub.all() and "email_addresses" == scrub.email_addresses().

        Args:
            scrub_method (str): string name of scrub method.

        Returns:
             list[str]: The input list of text with personal information replaced.

        """

        scrub_methods = {
            "all": self.all,
            "spacy_persons": self.spacy_persons,
            "huggingface_persons": self.huggingface_persons,
            "email_addresses": self.email_addresses,
            "handles": self.handles,
            "ip_addresses": self.ip_addresses,
            "uk_phone_numbers": self.uk_phone_numbers,
            "google_phone_numbers": self.google_phone_numbers,
            "uk_postcodes": self.uk_postcodes,
            "titles": self.titles,
            "presidio": self.presidio,
        }

        return scrub_methods.get(scrub_method, lambda: "Unknown method.")()

    @staticmethod
    def dataframe(
        df: pd.DataFrame = None,
        id_col: str = None,
        scrub_methods: list[str] = ["all"],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Scrubs all personal data from a Pandas Dataframe.

        Args:
            df (pd.DataFrame): A Pandas dataframe to scrub.
            id_col (str): Name of the ID column in `df`. If None, an integer index starting at 1  with the name `id` is applied.
            scrub_methods (list[str]): Which scrub methods to apply to the DataFrame (in order).
            These are string versions of the existing methods e.g. "all" == scrub.all() and "email_addresses" == scrub.email_addresses().

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: The input dataframe with all personal data removed and a dataframe with the personal data that has been removed.

        """

        if id_col:
            ids = df[id_col].to_list()
        if not id_col:
            id_col = "id"
            ids = range(1, len(df) + 1)

        assert isinstance(df, pd.DataFrame), "`df` must be a Pandas DataFrame."
        assert len(df) == len(ids), "Length of dataframe is different to the length of IDs."

        scrubbed_df = df.copy()

        all_scrubbed_data = []

        for col in tqdm(scrubbed_df.columns):
            if col == id_col:
                continue

            original_dtype = scrubbed_df[col].dtype
            scrubbed_df[col] = scrubbed_df[col].astype(str)

            scrub = IDScrub(texts=scrubbed_df[col].to_list(), text_id_name=id_col, text_ids=ids)

            for i, scrub_method in enumerate(scrub_methods):
                if i == len(scrub_methods) - 1:
                    scrubbed_texts = scrub.call_scrub_method(scrub_method)
                else:
                    scrub.call_scrub_method(scrub_method)

            scrubbed_df[col] = scrubbed_texts

            scrubbed_data = scrub.get_scrubbed_data()

            if scrubbed_data is not None:
                scrubbed_data.insert(1, "column", col)
                all_scrubbed_data.append(scrubbed_data)

            try:
                scrubbed_df[col] = scrubbed_df[col].astype(original_dtype)
            except ValueError:
                # If dtype is not revertable because it has been scrubbed, then pass
                pass

        all_scrubbed_data = pd.concat(all_scrubbed_data).reset_index(drop=True)
        all_scrubbed_data = all_scrubbed_data.where(pd.notna(all_scrubbed_data), None)

        return scrubbed_df, all_scrubbed_data
