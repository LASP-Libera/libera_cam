# Libera Wide Field-of-View Camera L1b Algorithm

This is the alpha version of the L1b algorithm for the Libera Wide Field-of-View Camera.

# Project Setup

## Software Installation Requirements

Python Version Management

- Conda (we recommend using Miniconda)
  Python Dependency Management
- Poetry (we recommend installing from the official script not with pipx)

_Note: This setup assumes that both conda and poetry are installed and available in your PATH._

## One Time Setup steps

### Python Version Management with Conda

To begin we want to set the correct version of python to build our environment from.

```bash
conda create -n conda-python3.11 python=3.11
```

This creates a conda environment named `conda-python3.11` with Python 3.11 installed with
no additional packages. This environment will serve as the base interpreter for all
subsequent virtual environments.

### Poetry Configuration

Poetry is a dependency management tool for Python that simplifies package management
and virtual environment handling. Our team prefers using Poetry for managing Python projects
and configures our poetry system to create virtual environments in the project directory.

```bash
poetry config virtualenvs.in-project true
```

## Virtual Environment Setup

To set up a virtual environment for your project, follow these steps: 1. Save the path to the base conda environment's Python interpreter 2. Create a new poetry virtual environment using the base conda environment 3. Install the project dependencies using Poetry 4. Activate the virtual environment

### Linux and MacOS

Steps 1 + 2

```bash
export PATH_TO_PYTHON=$(conda env list | grep "conda-python3.11" | awk '{print $2}')/bin/python
poetry env use $PATH_TO_PYTHON
```

Here you can ensure that the virtual environment was created successfully by running:

```bash
poetry env info
```

This should result in output similar to:

```
Virtualenv
   Python:         3.11.7
   Implementation: CPython
   Path:           /Users/myuser/path/to/libera_cam/.venv
   Valid:          True
```

Steps 3 + 4

```bash
poetry install
source .venv/bin/activate
```

This will install the project dependencies and activate the virtual environment in the current shell session.

### Windows (Git Bash or WSL)

Steps 1 + 2

```bash
export PATH_TO_PYTHON=$(conda env list | grep "conda-python3.11" | awk '{print $2}')\\python.exe
poetry env use $PATH_TO_PYTHON
```

Here you can ensure that the virtual environment was created successfully by running:

```bash
poetry env info
```

This should result in output similar to:

```
Virtualenv
   Python:         3.11.7
   Implementation: CPython
   Path:           /Users/myuser/path/to/libera_cam/.venv
   Valid:          True
```

Steps 3 + 4

```bash
poetry install
source .venv/Scripts/activate
```

This will install the project dependencies and activate the virtual environment in the current shell session.
