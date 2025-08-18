from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Sequence, Dict, Any, List, Optional
from ...fed_runner.fed_runner import FedRunner

class RunnerStrategy(ABC):
    @abstractmethod
    def run(self, runner: FedRunner) -> None:
        return
