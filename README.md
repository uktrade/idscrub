# idscrub 🧽✨

## Project Info

* This package removes (*✨scrubs✨*) identifying personal data from text using [regular expressions](https://en.wikipedia.org/wiki/Regular_expression) and [named-entity recognition](https://en.wikipedia.org/wiki/Named-entity_recognition).

> [!WARNING]
> You must follow [GDPR guidance](https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/the-research-provisions/principles-and-grounds-for-processing/) when processing personal data using this package.
>
> Specifically, you must:
>
> - **Update privacy notices**: Clearly state this processing activity in new or existing privacy notices before using the package.
> - **Ensure secure deletion**: Remove any temporary or intermediary files and outputs in a secure manner.
> - **Ensure data subject rights upheld**: Ensure individuals can access, correct, or erase their data as required.
> - **Maintain processing records**: Document how personal data is handled and for what purpose.

### Description

* Names and other personally identifying information are often present in text.
* This information may need to be removed prior to further analysis in many cases.
* `idscrub` provides a standardised way to do this in the Department for Business and Trade. 

### Expected Outputs

* A list of text with names and other identifying information removed.

> [!WARNING]
> * This package has been designed as a *first pass* for standardised personal data removal. 
> * Users are encouraged to check and confirm outputs and conduct manual reviews where necessary, e.g. when cleaning high risk datasets.
> * It is up to the user to assess whether this removal process needs to be supplemented by other methods for their given dataset and security requirements.

### Data

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

### Models and Memory

* Only Spacy's `en_core_web_trf` and no Hugging Face models have been formally evaluated.
* We therefore recommend that the current default `en_core_web_trf` is used for name scrubbing. **Other models need to be evaluated by the user.**

> [!IMPORTANT]
> Spacy and Hugging Face models have high memory requirements. To avoid memory-related errors. Clear the auto-generated `huggingface` folder if not in use. Do not push the `huggingface` folder (or user-defined equivalent) to GitHub.

## Similar Python packages

* Similar packages exist for undertaking this task, such as [presidio](https://microsoft.github.io/presidio/), [scrubadub](https://github.com/LeapBeyond/scrubadub) and [sanityze](https://github.com/UBC-MDS/sanityze). 
* Development of `idscrub` was undertaken to: bring together different scrubbing methods across the department, adhere to infrastructure requirements, guarantee future stability and maintainability, and encourage future scrubbing methods to be added collaboratively and transparently. 
* To leverage the power of other packages, we have added methods that allow you to interact with them. These include: `IDScrub.presidio()` and `IDScrub.google_phone_numbers()`. See the [usage example notebook](https://github.com/uktrade/idscrub/blob/main/notebooks/basic_usage.ipynb) and method docstrings for further information.


## Installation

`idscrub` can be installed using `pip` into a Python **>=3.12** environment. Example (with spaCy model installed):

```console
pip install 'git+ssh://git@github.com/uktrade/idscrub.git#egg=idscrub[trf]'
```
or without spaCy installed (it will be installed automatically if name cleaning methods are called):

```console
pip install 'git+ssh://git@github.com/uktrade/idscrub.git'
```

## How to use the code

Basic usage example (see `notebooks/basic_usage.ipynb` for further examples):

```python
from idscrub import IDScrub

scrub = IDScrub(['Our names are Hamish McDonald, L. Salah, and Elena Suárez.', 'My number is +441111111111 and I live at AA11 1AA, Lapland.'])
scrubbed_texts = scrub.all()

print(scrubbed_texts)

# Output: ['Our names are [PERSON], [PERSON], and [PERSON].', 'My number is [PHONENO] and I live at [POSTCODE], [LOCATION].']
```

## AI Declaration

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
