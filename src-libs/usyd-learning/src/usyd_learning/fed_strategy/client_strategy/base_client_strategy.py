from abc import ABC, abstractmethod

class ClientStrategy(ABC):
    def __init__(self, client):
        self.client = client

    @abstractmethod
    def run_local_training(self) -> dict:
        pass