from __future__ import annotations

###
# Event Args class
###


class EventArgs:
    def __init__(self, sender: any = None):
        """
        " Event sender object
        """

        self._sender: any = sender

    @property
    def sender(self) -> any:
        """
        " Readonly sender object
        """

        return self._sender

    def with_sender(self, sender: any) -> EventArgs:
        """
        " set event arg with sender object
        """

        self._sender = sender
