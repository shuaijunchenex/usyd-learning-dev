from __future__ import annotations
from abc import ABC

from .console import console
from .filename_helper import FileNameHelper
from .string import String
from .config_loader import ConfigLoader
from .object_map import ObjectMap


class AppEntry(ABC, ObjectMap):
    """
    App entry class
    """
    # Static class variables
    __app_objects = ObjectMap()  # App objects map
    __app_config_define: dict = {}  # App config define dict

    def __init__(self):
        ObjectMap.__init__(self)

    @property
    def app_config(self):
        return AppEntry.__app_objects

    @property
    def app_config_define(self):
        return AppEntry.__app_config_define

    def run(self, training_rounds=50):
        """
        Run app
        """
        pass

    def load_app_config(self, app_config_define_file: str):
        """
        Load app config define file and parse it
        """
        if not FileNameHelper.exists(app_config_define_file):
            console.raise_exception(ValueError(f"App config define file '{app_config_define_file}' not exists."))

        AppEntry.__app_config_define = ConfigLoader.load(app_config_define_file)

        file_part = FileNameHelper.split(app_config_define_file)
        self.__parse_app_config_define(file_part.folder, AppEntry.__app_config_define)
        return

    def __parse_app_config_define(self, folder: str, config_define: dict) -> None:
        yaml_map = {}
        yaml_path = config_define.get("yaml_section_path", "")

        # Section: "yaml_section_files"
        for yaml_section_files in config_define["yaml_section_files"]:
            file_name = next(iter(yaml_section_files))
            name = yaml_section_files[file_name]

            # Check file exists
            fullname = FileNameHelper.combine(folder, yaml_path, file_name)
            if not FileNameHelper.exists(fullname):
                raise ValueError(f"yaml file not found '{fullname}'")

            # Empty name
            if String.is_none_or_empty(name):
                file_part = FileNameHelper.split(fullname)
                name = file_part.name

            config_dict = ConfigLoader.load(fullname)
            AppEntry.__app_objects.add_object(name, config_dict)
            yaml_map[name] = fullname

        # Section: "yaml_combination"
        yaml_combine = config_define["yaml_combination"]
        for cfg_name, combination in yaml_combine.items():
            if cfg_name == "yaml_section_path" or cfg_name == "yaml_section_files":
                continue

            # Existence check -- repeat not allowed
            if AppEntry.__app_objects.exists_object(cfg_name):
                raise ValueError(f"Combined yaml '{cfg_name}' already exists in app configs")

            combine_dict = {}
            if isinstance(combination, list):
                for name in combination:
                    config_dict = ConfigLoader.load(yaml_map[name])
                    combine_dict.update(config_dict)

            AppEntry.__app_objects.add_object(cfg_name, combine_dict)
        return
