.PHONY: clean precommit test 

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Reformat, lint
format:
	uv run ruff format .
	uv run ruff check . --fix

## Run Python tests
test:
	uv run pytest