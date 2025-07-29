from __future__ import annotations

class ObjectMap(object):

    def __init__(self):
        self.__object_map: dict = {}
        return

    @property
    def count(self):
        """
        Number of fn registered
        """
        return len(self.__object_map)


    def add_object(self, any_key: any, object_instance):
        """
        Register an object instance, if object exists, replace the old one
        """

        if object_instance is None:
            raise ValueError("Register object instance is None.")

        self.__object_map[any_key] = object_instance
        return self


    def remove_object(self, any_key: any):
        """
        Unregister object
        """

        if any_key in self.__object_map:
            del self.__object_map[any_key]
        return self


    def exists_object(self, any_key):
        """
        Determine fn key is registered
        """

        return any_key in self.__object_map


    def set_object(self, any_key: any, object_instance: any = None):
        """
        Set or replace object instance
        """
        self.__object_map[any_key] = object_instance


    def get_object(self, any_key: any, cast_type = None):
        """
        get object
        """

        if not self.exists_object(any_key):
            return None

        if cast_type is not None:
            try:
                return cast_type(self.__object_map[any_key])
            except Exception as e:
                raise ValueError(f"Failed to cast instance '{self.__object_map[any_key]}' to {cast_type}: {e}")

        return self.__object_map[any_key]
