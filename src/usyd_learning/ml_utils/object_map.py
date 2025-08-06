from __future__ import annotations
from typing import Any


class ObjectMap(object):
    def __init__(self):
        self.__object_map: dict = {}
        return

    @property
    def count(self):
        """
        Number of object added
        """
        return len(self.__object_map)

    def add_object(self, any_key: Any, object_instance):
        """
        Register an object instance, if object exists, replace the old one
        """
        if object_instance is None:
            raise ValueError("Add object instance is not None.")

        self.__object_map[any_key] = object_instance
        return self

    def remove_object(self, any_key: Any):
        """
        Remove object by key
        """
        if any_key in self.__object_map:
            del self.__object_map[any_key]
        return self

    def exists_object(self, any_key):
        """
        Determine fn key is registered
        """
        return any_key in self.__object_map

    def set_object(self, any_key: Any, object_instance: Any):
        """
        Set or replace object instance
        """
        self.__object_map[any_key] = object_instance

    def get_object(self, any_key: Any, cast_type=None):
        """
        get object by key
        """
        if any_key is None:
            raise ValueError(f"Object key is None")
        if not self.exists_object(any_key):
            raise ValueError(f"Object not found by key '{any_key}'")

        if cast_type is not None:
            try:
                return cast_type(self.__object_map[any_key])
            except Exception as e:
                raise ValueError(f"Failed to cast instance '{self.__object_map[any_key]}' to {cast_type}: {e}")

        return self.__object_map[any_key]
