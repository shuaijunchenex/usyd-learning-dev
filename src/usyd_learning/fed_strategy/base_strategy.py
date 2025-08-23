from abc import ABC, abstractmethod
from .strategy_args import StrategyArgs

class BaseStrategy(ABC):

    def __init__(self):
        self._strategy_type = None
        self._obj = None
        self._is_created = False
        self._args = None
        self._after_create_fn = None

        return
    
    # --------------------------------------------------
    def create(self, args: StrategyArgs):
        """
        Create Dataset Loader
        """
        self._args = args
        self._create_inner(args)  # create dataset loader

        return self
    
    @abstractmethod
    def _create_inner(self, args: StrategyArgs) -> None:
        """
        Real create loader
        """
        pass