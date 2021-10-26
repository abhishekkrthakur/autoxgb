.PHONY: quality style

quality:
	black --check --line-length 119 --target-version py38 .
	isort --check-only .
	flake8 --max-line-length 119

style:
	black --line-length 119 --target-version py38 .
	isort .