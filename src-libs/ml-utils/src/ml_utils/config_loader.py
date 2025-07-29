import os
import json
import yaml

"""
Load config file(json, yaml or yml)
"""

class ConfigLoader:
    @staticmethod
    def load(file_name: str, encoding: str = "utf-8"):
        """
        Load a JSON/YAML file and return the data as a Python object.

        Args:
            file_name (str): Path to the JSON/YAML file.

        Returns:
            dict or list: Parsed JSON/YAML data.
        """

        if(not os.path.exists(file_name)):
            raise Exception("Config file not exists")

        _, extension = os.path.splitext(file_name)
        if extension == ".json":
            with open(file_name, "r", encoding = encoding) as f:
                data = json.load(f)
        elif extension == ".yaml" or ".yml":
            with open(file_name, "r", encoding = encoding) as f:
                data = yaml.safe_load(f)

        return data
