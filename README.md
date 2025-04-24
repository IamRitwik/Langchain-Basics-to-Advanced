# First Time Setup

## Using Venv [Optional]

These instructions are included if you wish to use venv to manage your evironment and dependencies instead of Pipenv.

```
# Create the venv virtual environment
python -m venv .venv

# On MacOS, WSL, Linux
source .venv/bin/activate

# On Windows
.\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

pip install python-magic python-magic-bin unstructured[md]
