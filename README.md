# AI Agents Project
Building and testing different AI agents and agentic system architectures.

## Overview 
This repository provides a lightweight framework for defining, configuring, testing, and benchmarking different agent types and multi-agent system architectures. The code emphasizes clear separation between agent implementations (`think()`/`act()`), configurations, architecture coordination, and experiment-focused notebooks and benchmarks.

Key concepts:
- **Agent**: An individual unit — implemented by subclasses of `BaseAgent` in `src/agents/`.
- **Architecture**: Orchestrates one or more agents, defines communication & coordination, and may include central or distributed planning logic — implemented in `src/architectures/`.
- **Configuration**: YAMLfiles that set model parameters, tools, memory, and behavior policies for agents and architectures (in `configs/`).


## Project Structure 

* **`configs/`**:
    * `agents/`: YAML definitions for agents.
    * `architectures/`: YAML definitions for architectures.
    * `schemas/`: Pydantic models enforcing validation rules.
* **`src/`**: Source code.
    * `agents/`: 
    * `architectures/`: 
    * `core`: Core infrastructure files
        * `logging_setup.py`: Sets the logging configuration for the application.
        * `execution_service.py`: Creates LLM client based upon specified configuration file.
        * `architecture_manager.py`:
        * `state.py`:
    * `utils`: Core utility files.
        * `config_loader.py`: Parses YAML files and validates against Pydantic models. Options available for both agent YAML files and architecture YAML files.
    * `constants.py`: Contains constants used throughout SRC folder. Includes directory file paths.
    * `main.py`:
* **`tests/`**: Test code.
    * `integration`: Contains integration tests.
    * `unit`: Contains unit tests.
        * `agents`:
        * `architecture`:
        * `core`:
        * `utils`:
    * `conftest.py`:

See ARCHITECTURE.md for more information on project architecture.

## Getting Started 

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install the package and dev extras (recommended):
```bash
pip install -e ".[dev]"
```

3. Run a quick test to verify the environment:
```bash
# Run mypy static checks
python -m mypy src

# Run the tests
pytest tests/ -v

# Run the tests with coverage. Report shows missing lines.
pytest --cov=src --cov-report=term-missing
```

## Running Tests & CI 

Use pytest for local testing:

```bash
pytest tests/ -v
```

For `mypy` / type-checking, run:

```bash
python -m mypy src
```

CI should run these checks and may use a pinned `requirements-dev.txt` to ensure reproducible test environments.

## Helpful Commands

To view the entire project structure from the terminal:

```bash
tree -I "__pycache__|.git|.venv|.pytest_cache"
```


 
