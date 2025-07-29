import os
import json
import yaml

from .string import String

"""
Load config file(json, yaml or yml)
"""

class ConfigLoader:
    """
    Load config file(json, yaml or yml)
    """

    @staticmethod
    def load(file_name: str, encoding: str = "utf-8"):
        """
        Load a JSON/YAML file and return the data as a Python object.

        Args:
            file_name (str): Path to the JSON/YAML file.

        Returns:
            dict or list: Parsed JSON/YAML data.
        """

        if String.is_none_or_empty(file_name):
            raise ValueError(f"Config file is none or empty.")

        if(not os.path.exists(file_name)):
            raise Exception(f"Config file '{file_name}' not exists")

        file_name = file_name.lower()
        if file_name.endswith(".json"):
            with open(file_name, "r", encoding = encoding) as f:
                data = json.load(f)
        elif file_name.endswith(".yaml") or file_name.endswith(".yml"):
            with open(file_name, "r", encoding = encoding) as f:
                data = yaml.safe_load(f)

        return data
