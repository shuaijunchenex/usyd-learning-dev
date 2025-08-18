from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Sequence, Dict, Any, List, Optional
from ...fed_runner.fed_runner import FedRunner

class RunnerStrategy(ABC):
    
    @abstractmethod
    def run(self, runner: FedRunner) -> None:
        self.runner = runner
        return
    
    @abstractmethod
    def simulate_local_train():
        """
        Local training simulation method.
        This method should be overridden by subclasses to implement local training logic.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def simulate_server_broadcast():
        """
        Server broadcast simulation method.
        This method should be overridden by subclasses to implement server broadcast logic.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def simulate_server_update():
        """
        Server update simulation method.
        This method should be overridden by subclasses to implement server update logic.
        """
        raise NotImplementedError("Subclasses must implement this method.")
