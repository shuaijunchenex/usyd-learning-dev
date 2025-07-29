from __future__ import annotations

from ml_simu_switcher.event_handler.event_args import EventArgs

###
# Event Handler(delegate) class
###


class EventHandler:
    def __init__(self):
        """
        " Event function handler(callback) dict
        """

        self._event_handler_dict = {}

    def exists_event(self, event_name: str, handler=None) -> bool:
        """
        " event exists
        """

        if event_name not in self._event_handler_dict:
            return False

        if handler is not None:
            return handler in self._event_handler_dict[event_name]

        return True

    def attach_event(self, event_name: str, handler):
        """
        " Attach event handler to event id
        """

        if not self.exists_event(event_name):
            self._event_handler_dict[event_name] = [handler]
            return

        if self.exists_event(event_name, handler):
            return

        self._event_handler_dict[event_name].append(handler)

        return

    def detach_event(self, event_name: str, handler = None):
        """
        " Detach event handler
        """

        if not self.exists_event(event_name, handler):
            return

        if handler is None:
            del self._event_handler_dict[event_name]  # Remove all handler
        else:
            self._event_handler_dict[event_name].remove(handler)
            if len(self._event_handler_dict[event_name]) <= 0:
                del self._event_handler_dict[event_name]

    def raise_event(self, event_name: str, args: EventArgs):
        """
        " Raise event
        """

        if not self.exists_event(event_name):
            return

        for handler in self._event_handler_dict[event_name]:
            if handler is not None:
                handler(args)

    def clear_event(self):
        """
        " Clear all handler
        """

        self._event_handler_dict.clear()

    def __str__(self):
        return " ".join(self._event_handler_dict.keys)
