from __future__ import annotations

###
# Event Args class
###


class EventArgs:
    def __init__(self, sender: any=None, kind: str= "", data: any = None):
        """
        Event sender object
        Members:
            sender: who call the event
            data: event data
            kind: event kind string
        """
        self._sender: any = sender
        self.kind: str = kind
        self.data: any = data


    @property
    def sender(self) -> any:
        """
        " Readonly sender object
        """
        return self._sender


    def with_kind(self, kind: str) -> EventArgs:
        """
        " set event kind
        """
        self.kind = kind
        return self

    def with_data(self, data: any) -> EventArgs:
        """
        " set event kind
        """
        self.data = data
        return self

    def with_sender(self, sender: any) -> EventArgs:
        """
        " set event arg with sender object
        """
        self._sender = sender
        return self
