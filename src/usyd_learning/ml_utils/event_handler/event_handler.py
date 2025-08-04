from __future__ import annotations

from .event_args import EventArgs

###
# Event Handler(delegate) class
###


class EventHandler:
    def __init__(self):
        """
        " Event function handler(callback) dict
        """

        self.__event_handler_dict = {}

    def exists_event(self, event_name: str, handler=None) -> bool:
        """
        " event exists
        """

        if event_name not in self.__event_handler_dict:
            return False

        if handler is not None:
            return handler in self.__event_handler_dict[event_name]

        return True

    def attach_event(self, event_name: str, handler):
        """
        " Attach event handler to event id
        """

        if not self.exists_event(event_name):
            self.__event_handler_dict[event_name] = [handler]
            return self

        if self.exists_event(event_name, handler):
            return self

        self.__event_handler_dict[event_name].append(handler)
        return self

    def detach_event(self, event_name: str, handler=None):
        """
        " Detach event handler
        """

        if not self.exists_event(event_name, handler):
            return self

        if handler is None:
            del self.__event_handler_dict[event_name]  # Remove all handler
        else:
            self.__event_handler_dict[event_name].remove(handler)
            if len(self.__event_handler_dict[event_name]) <= 0:
                del self.__event_handler_dict[event_name]
        return self

    def raise_event(self, event_name: str, args: EventArgs):
        """
        " Raise event
        """

        if not self.exists_event(event_name):
            return

        for handler in self.__event_handler_dict[event_name]:
            if handler is not None:
                handler(args)

    def clear_event(self):
        """
        " Clear all handler
        """

        self.__event_handler_dict.clear()

    def __str__(self):
        return " ".join(self.__event_handler_dict)
