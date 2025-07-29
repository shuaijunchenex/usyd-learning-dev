# ml-utils Package

The 'ml-utils' package 



## Package build & deploy

- Step 1: Change deploy VERSION in pyproject.toml

```
###
# Package version
###
version = "0.0.3"			# here
```

- Step 2: Build package command

```cmd
python -m build
```
- Step 3: Upload package to Nexus command

```cmd
twine upload -r nexus dist/*
```

- Step 4: Pip install or upgrade

```
pip install ml_utils
pip install ml_utils --upgrade
```

  Ruff check & format code
```cmd
ruff check
ruff format
ruff check --fix
```