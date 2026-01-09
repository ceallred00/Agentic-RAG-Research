# Makefile for AI Agents project

.PHONY: help install install-dev test test-cov lint format clean run-tests benchmark

help:
	@echo "Available commands:"
	@echo "  make install       - Install project dependencies"
	@echo "  make install-dev   - Install with dev dependencies"
	@echo "  make test          - Run tests"
	@echo "  make test-cov      - Run tests with coverage report"
	@echo "  make lint          - Run code linting"
	@echo "  make format        - Format code with black"
	@echo "  make clean         - Clean up generated files"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev,ml,notebooks]"

test:
	pytest tests/

test-cov:
	pytest tests/ --cov=src --cov-report=html

lint:
	pylint src/ tests/

format:
	black src/ tests/ configs/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ *.egg-info .coverage htmlcov/

benchmark:
	python -m pytest benchmarks/ -v
