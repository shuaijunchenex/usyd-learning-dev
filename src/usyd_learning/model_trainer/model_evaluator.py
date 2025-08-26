from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import torch.nn as nn

from ..ml_utils import console

class ModelEvaluator:
    """
    A stateful evaluator for PyTorch models.
    Initialized with model, validation dataloader, and device.
    """

    def __init__(self, model, val_loader, criterion = None, device = "cpu"):
        """
        :param model: PyTorch model to evaluate
        :param val_loader: DataLoader with validation or test data
        :param criterion: Loss function (e.g., CrossEntropyLoss). If None, uses CrossEntropyLoss
        :param device: Computation device ('cpu' or 'cuda')
        """

        self.model = model.to(device)
        self.val_loader = val_loader
        self.device = device
        
        # Default to CrossEntropyLoss if no criterion provided
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        self.latest_metrics = {}

    def evaluate(self, average = "macro"):
        """
        Evaluate the model using multiple metrics including loss.

        :param average: averaging method for multi-class metrics
        :return: dictionary of evaluation results
        """

        self.model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                
                # Calculate loss
                loss = self.criterion(outputs, labels)
                total_loss += loss.item() * inputs.size(0)  # Multiply by batch size
                total_samples += inputs.size(0)
                
                # Get predictions
                _, predicted = torch.max(outputs.data, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate average loss
        avg_loss = total_loss / total_samples

        self.latest_metrics = {
            "accuracy": accuracy_score(all_labels, all_preds),
            "average_loss": avg_loss,
            "precision": precision_score(all_labels, all_preds, average=average, zero_division=0),
            "recall": recall_score(all_labels, all_preds, average=average, zero_division=0),
            "f1_score": f1_score(all_labels, all_preds, average=average, zero_division=0),
            "total_test_samples": total_samples
        }

        return self.latest_metrics
    
    def print_results(self):
        """
        Pretty-print the latest evaluation metrics.
        Should be called after evaluate().
        """

        if not self.latest_metrics:
            console.error("No evaluation metrics available. run .evaluate() first.")
            return

        console.info("Evaluation Summary:")
        console.info(f"  - Loss     : {self.latest_metrics['average_loss']:.4f}")
        console.info(f"  - Accuracy : {self.latest_metrics['accuracy'] * 100:.2f}%")
        console.info(f"  - Precision: {self.latest_metrics['precision']:.4f}")
        console.info(f"  - Recall   : {self.latest_metrics['recall']:.4f}")
        console.info(f"  - F1-Score : {self.latest_metrics['f1_score']:.4f}")
        console.info(f"  - Samples  : {self.latest_metrics['total_test_samples']}")
        return

    def get_accuracy(self):
        """
        Quick access to accuracy metric.
        :return: accuracy value or None if not evaluated yet
        """
        return self.latest_metrics.get('accuracy', None)


    def get_loss(self):
        """
        Quick access to loss metric.
        :return: loss value or None if not evaluated yet
        """
        return self.latest_metrics.get('loss', None)

