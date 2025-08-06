from __future__ import annotations
import os

from .csv_data_recorder import CsvDataRecorder
from .filename_maker import FileNameMaker


class TrainingLogger:
    """
    log train result to file
    """

    def __init__(self, config_dict: dict|None = None):
        """
        Init training logger
        Args:
            config_dict: dictionary, include "name", "path", "prefix"
        """
        d: dict = {}
        if config_dict is None:
            d = {}
        elif "training_logger" in config_dict:
            d = config_dict["training_logger"]

        self.name = d.get("name", "train")
        self.path: str = d.get("path", "./.training_results/")
        self.prefix: str = d.get("prefix", "")

        self.__file_names = None
        self.__logger = None
        return

    def begin(self, header_config_dict: dict|None = None):
        """
        Write log begin
        """
        self.__file_names = FileNameMaker.with_path(self.path).with_prefix(self.prefix).make(self.name)

        (path, _) = os.path.split(self.__file_names.fullname)
        os.makedirs(path, exist_ok=True)

        self.__logger = CsvDataRecorder(self.__file_names.fullname)
        self.__logger.begin(header_config_dict)
        return

    def end(self):
        """
        Write log end
        """
        if hasattr(self, "_TrainingLogger__logger") and  self.__logger is not None:
            self.__logger.end()
        return

    def record(self, result_dict: dict):
        """
        Write record to CSV
        """
        self.__logger.record(result_dict)
        return self

    def __del__(self):
        self.end()
