from __future__ import annotations

from abc import ABC, abstractmethod


class AbstractFedRunner(ABC):
    def __init__(self, runer_config_file, cfg_file: str):
        self.config_file = cfg_file
        self.runer_config = runer_config_file

        self.switcher = SimuSwitcher()          #Simulate net switcher

        return

    @abstractmethod
    def load_config(self):
        pass

    @abstractmethod
    def prepare_data(self):
        pass

    @abstractmethod
    def prepare_data_distribution(self):
        pass

    @abstractmethod
    def prepare_model(self):
        pass

    @abstractmethod
    def prepare_base_model(self):
        pass

    @abstractmethod
    def prepare_server(self):
        pass
        
    @abstractmethod
    def prepare_clients(self):
        pass

    @abstractmethod
    def run_loop(self):
        pass

    def run(self):
        self.load_config()
        self.prepare_data()
        self.prepare_data_distribution()
        self.prepare_base_model()
        self.prepare_server()
        self.prepare_clients()
        self.run_loop()