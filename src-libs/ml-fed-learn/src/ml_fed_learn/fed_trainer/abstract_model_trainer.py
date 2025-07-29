from abc import ABC, abstractmethod

class BaseModelTrainer(ABC):
    def __init__(self, model, optimizer, criterion, train_loader, val_loader=None, device="cpu", save_path=None):
        self.model = model.to(device) if str(next(model.parameters()).device) != device else model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_path = save_path
        self.best_val_acc = 0.0

    @abstractmethod
    def train_step(self):
        """Performs a single training step."""
        pass

    @abstractmethod
    def train(self, epochs):
        """Trains the model for a number of epochs."""
        pass

    def observe(self, epochs=5):
        """Performs observation without updating the global state."""
        pass

    def extract_WbAB(self):
        """Extracts structured model components (e.g., LoRA components)."""
        pass
