from abc import ABC, abstractmethod

class ServerStrategy(ABC):
    def __init__(self, server):
        self.server = server

    @abstractmethod
    def aggregation(self, client_weights: list) -> dict:
        """
        Aggregates the weights from clients.
        :param client_weights: List of weights from clients.
        :return: Aggregated weights.
        """
        pass

    @abstractmethod
    def broadcast(self, aggregated_weights: dict) -> None:
        """
        Broadcasts the aggregated weights to clients.
        :param aggregated_weights: The aggregated weights to be broadcasted.
        """
        pass

    @abstractmethod
    def run(self) -> dict:
        pass

    @abstractmethod
    def evaluate(self) -> None:
        """
        Evaluates the server's performance.
        This method can be overridden by subclasses to implement specific evaluation logic.
        """
        pass