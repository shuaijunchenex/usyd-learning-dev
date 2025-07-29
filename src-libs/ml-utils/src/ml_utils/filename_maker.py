from __future__ import annotations

from collections import namedtuple
import os
import datetime
import hashlib


"""
Declare FileNameMaker make method return named tuple
"""
FileNameMakerNames = namedtuple("FileNameMakerNames", ["name", "path", "filename", "fullname"])

"""
Make file name by config(dict)
"""
class FileNameMaker:
    _file_path: str = "./results/"
    _custom_prefix: str = ""
    _include_timestamp: bool = True
    _use_hash: bool = True
    _name: str = ""


    @staticmethod
    def with_args(file_path: str = "./results/", custom_prefix: str = "", include_timestamp: bool = True, use_hash: bool = True):
        """
        Set file name generate args

        Args:
            config (dict): Dictionary containing parameters.
            file_path (str): Directory where the file will be saved.
            custom_prefix (str): Custom prefix for the filename (default: "experiment").
            include_timestamp (bool): Whether to include a timestamp in the filename.
            use_base64 (bool): Whether to generate a base64 encoded string from config to shorten the filename.
        """

        FileNameMaker._file_path = file_path
        FileNameMaker._custom_prefix = custom_prefix
        FileNameMaker._include_timestamp = include_timestamp
        FileNameMaker._use_hash = use_hash
        return FileNameMaker


    @staticmethod
    def make(config: dict, name: str = "", file_extension: str = ".csv"):
        """
        Generate a unique filename based on the configuration.

        Args:
            file_extension (str): The file extension (default: ".csv").

        Returns:
            tuple: Generated filename and full file path.
            (file_path, filename, fullname)
        """
        
        FileNameMaker._name = name
        os.makedirs(FileNameMaker._file_path, exist_ok=True)

        # Step 1: Convert config to consistent string
        important_keys = sorted(config.keys())
        config_str = "_".join(f"{key}={config[key]}" for key in important_keys)

        # Step 2: Optional timestamp
        filename = FileNameMaker._custom_prefix

        if name:
            filename += f"{name}"

        if FileNameMaker._include_timestamp:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename += f"_{timestamp}"

        # Step 3: Encode config
        if FileNameMaker._use_hash:
            config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
            filename += f"_{config_hash}"
        else:
            safe_str = (
                config_str.replace(" ", "")
                .replace(":", "")
                .replace(",", "")
                .replace("/", "")
                .replace("\\", "")
            )
            filename += f"_{safe_str}"

        # Step 4: Add extension and path
        filename += file_extension
        fullname = os.path.join(FileNameMaker._file_path, filename)

        return FileNameMakerNames(FileNameMaker._name, FileNameMaker._file_path, filename, fullname)