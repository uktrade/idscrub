# idscrub 🧽✨

* Names and other personally identifying information are often present in text, even if they are not clearly visible or requested.
* This information may need to be removed prior to further analysis in many cases.
* `idscrub` identifies and removes (*✨scrubs✨*) personal data from text using [regular expressions](https://en.wikipedia.org/wiki/Regular_expression) and [named-entity recognition](https://en.wikipedia.org/wiki/Named-entity_recognition).

## Installation

`idscrub` can be installed using `pip` into a Python **>=3.12** environment. Example:

```console
pip install idscrub
```
or with the spaCy transformer model (`en_core_web_trf`) already installed:

```console
pip install idscrub[trf]
```
## How to use the code

Basic usage example (see [basic_usage.ipynb](https://github.com/uktrade/idscrub/blob/main/notebooks/basic_usage.ipynb) for further examples):

```python
from idscrub import IDScrub

scrub = IDScrub(['Our names are Hamish McDonald, L. Salah, and Elena Suárez.', 'My number is +441111111111 and I live at AA11 1AA.'])x
scrubbed_texts = scrub.scrub(scrub_methods=['spacy_persons', 'uk_phone_numbers', 'uk_postcodes'])

print(scrubbed_texts)

# Output: ['Our names are [PERSON], [PERSON], and [PERSON].', 'My number is [PHONENO] and I live at [POSTCODE].']
```

## Considerations before use

- You must follow [GDPR guidance](https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/the-research-provisions/principles-and-grounds-for-processing/) when processing personal data using this package.
- This package has been designed as a *first pass* for standardised personal data removal. 
- Users are encouraged to check and confirm outputs and conduct manual reviews where necessary, e.g. when cleaning high risk datasets.
- It is up to the user to assess whether this removal process needs to be supplemented by other methods for their given dataset and security requirements.

### Input data

- This package is designed for text-based documents structured as a list of strings.
- It performs best when contextual meaning can be inferred from the text.
- For best results, input text should therefore resemble natural language. 
- **Highly fragmented, informal, technical, or syntactically broken text may reduce detection accuracy and lead to incomplete or incorrect name detection.**

### Biases and evaluation

- `idscrub` supports integration with SpaCy and Hugging Face models for name cleaning.
- These models are state-of-the-art, capable of identifying approximately 90% of named entities, but **may not remove all names**.
- **Biases present in these models due to their training data may affect performance**. For example:
    - English names may be more reliably identified than names common in other languages.
    - Uncommon or non-Western naming conventions may be missed or misclassified.

> [!IMPORTANT]
> * See [our wiki](https://github.com/uktrade/idscrub/wiki/Evaluation) for further details and notes on our evaluation of `idscrub`.

### Models

* Only Spacy's `en_core_web_trf` and no Hugging Face models have been formally evaluated.
* We therefore recommend that the current default `en_core_web_trf` is used for name scrubbing. **Other models need to be evaluated by the user.**

## Similar Python packages

* Similar packages exist for undertaking this task, such as [Presidio](https://microsoft.github.io/presidio/), [Scrubadub](https://github.com/LeapBeyond/scrubadub) and [Sanityze](https://github.com/UBC-MDS/sanityze). 
* Development of `idscrub` was undertaken to: 

    * Bring together different scrubbing methods across the Department for Business and Trade.
    * Adhere to infrastructure requirements.
    * Guarantee future stability and maintainability.
    * Encourage future scrubbing methods to be added collaboratively and transparently.
    * Allow for full flexibility depending on the use case and required outputs.
    
* To leverage the power of other packages, we have added methods that allow you to interact with them. These include: `IDScrub.presidio()` and `IDScrub.google_phone_numbers()`. See the [usage example notebook](https://github.com/uktrade/idscrub/blob/main/notebooks/basic_usage.ipynb) and method docstrings for further information.

## AI declaration

AI has been used in the development of `idscrub`, primarily to develop regular expressions, suggest code refinements and draft documentation.

## Development setup

This project is managed by [uv](https://docs.astral.sh/uv/).

To install all dependencies for this project, run:

```console
uv sync --all-extras
```

If you do not have Python 3.12, run:

```console
uv python install 3.12
```

To run tests:

```console
uv run pytest
```

or

```console
make test
```

## Author 

Analytical Data Science, Department for Business and Trade
