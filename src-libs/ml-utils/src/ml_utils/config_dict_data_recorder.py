from __future__ import annotations

import json
import csv

"""
Config(dict) and its data(dict) recorder
"""

class ConfigDictDataRecorder:
    def __init__(self, filename: str):
        """
        file name
        """
        self.__filename: str = filename
        
        """
        Indicate whether header is writed
        """
        self.__is_header_write: bool = False
        return


    def begin(self, config: dict = None):
        """
        Write log begin
        """

        self.__csv_stream = open(self.__filename, "a", newline = "", encoding = "utf-8")
        if(config is not None):
            self.__csv_stream.write(f"Config,{json.dumps(config, ensure_ascii=False)}\n\n")
        return


    def end(self):
        """
        Write log end
        """

        if self.__csv_stream is None or self.__csv_stream.closed:
            return

        self.__csv_stream.close()
        self.__csv_stream = None
        return


    def record(self, result_dict: dict) -> ConfigDictDataRecorder:
        """
        Write record to CSV
        """

        if self.__csv_stream is None:
            raise Exception("Must call begin() first")

        if(not self.__is_header_write):
            self._csv_writer = csv.DictWriter(self.__csv_stream, fieldnames = result_dict.keys())
            self._csv_writer.writeheader()
            self.__is_header_write = True
            
        self._csv_writer.writerow(result_dict)
        return self


    def __del__(self):
        self.end()
