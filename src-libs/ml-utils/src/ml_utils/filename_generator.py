from __future__ import annotations

import os
import datetime
import hashlib


class FileNameGenerator:
    def __init__(
        self,
        config,
        file_path: str = "./results/",
        custom_prefix: str = "",
        include_timestamp: bool = True,
        use_hash: bool = True
    ):
        """
        Initialize the FileNameGenerator.

        Args:
            config (dict): Dictionary containing parameters.
            file_path (str): Directory where the file will be saved.
            custom_prefix (str): Custom prefix for the filename (default: "experiment").
            include_timestamp (bool): Whether to include a timestamp in the filename.
            use_base64 (bool): Whether to generate a base64 encoded string from config to shorten the filename.
        """

        self.config = config
        self.file_path = file_path
        self.custom_prefix = custom_prefix
        self.include_timestamp = include_timestamp
        self.use_hash = use_hash

        # Ensure the directory exists
        os.makedirs(self.file_path, exist_ok=True)


    def generate(self, name: str = "", file_extension: str = ".csv"):
        """
        Generate a unique filename based on the configuration.

        Args:
            file_extension (str): The file extension (default: ".csv").

        Returns:
            str: Generated filename and full file path.
        """

        # Step 1: Convert config to consistent string
        important_keys = sorted(self.config.keys())
        config_str = "_".join(f"{key}={self.config[key]}" for key in important_keys)

        # Step 2: Optional timestamp
        filename = self.custom_prefix

        if name:
            filename += f"{name}"

        if self.include_timestamp:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename += f"_{timestamp}"

        # Step 3: Encode config
        if self.use_hash:
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
        full_path = os.path.join(self.file_path, filename)

        return filename, full_path