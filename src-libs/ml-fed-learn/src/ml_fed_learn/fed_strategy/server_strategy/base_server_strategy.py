from abc import ABC, abstractmethod

class ServerStrategy(ABC):
    def __init__(self, server):
        self.server = server

    @abstractmethod
    def run(self) -> dict:
        pass