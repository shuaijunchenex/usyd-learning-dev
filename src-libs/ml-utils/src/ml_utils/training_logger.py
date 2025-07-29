from __future__ import annotations

from ml_utils.config_dict_data_recorder import ConfigDictDataRecorder
from ml_utils.filename_maker import FileNameMaker

"""
log train result to file
"""


class TrainingLogger:
    def __init__(self, name: str, config: dict = None):
        self.__config = config

        """
        logger file names
        """
        self.__filenames = FileNameMaker.make(config, name)

        """
        logger
        """
        self.__logger = ConfigDictDataRecorder(self.__filenames.fullname)
        return

    def with_args(
        self,
        file_path: str = "./results/",
        custom_prefix: str = "",
        include_timestamp: bool = True,
        use_hash: bool = True,
    ):
        FileNameMaker.with_args(file_path, custom_prefix, include_timestamp, use_hash)
        self.__file_names = FileNameMaker.make(self.__config, self.__filenames.name)
        self.__logger = ConfigDictDataRecorder(self.__file_names.fullname)

        return self

    def begin(self):
        """
        Write log begin
        """

        self.__logger.begin(self.__config)
        return

    def end(self):
        """
        Write log end
        """

        self.__logger.end()
        return

    def record(self, result_dict: dict):
        """
        Write record to CSV
        """

        self.__logger.record(result_dict)
        return self

    def __del__(self):
        self.__logger.__del__()
