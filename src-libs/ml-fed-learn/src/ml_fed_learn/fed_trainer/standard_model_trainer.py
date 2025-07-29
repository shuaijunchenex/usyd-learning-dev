import torch
import torch
import math

from trainer.abstract_model_trainer import BaseModelTrainer
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from model_extractor.advanced_model_extractor import AdvancedModelExtractor
from tools.optimizer_builder import OptimizerBuilder

class StandardModelTrainer(BaseModelTrainer):
    def __init__(self, model, optimizer, criterion, train_loader, val_loader=None, device="cpu", save_path=None):

        super().__init__(model, optimizer, criterion, train_loader, val_loader, device, save_path)

        if str(next(model.parameters()).device) != device:
            self.model = model.to(device)
        else:
            self.model = model

    def train_step(self):
        self.model.train()
        running_loss = 0.0
        loop = tqdm(self.train_loader, desc="Training", leave=True, ncols=100, mininterval=0.1, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [elapsed: {elapsed} remaining: {remaining}]")
        total_batch = 0

        for inputs, labels in loop:
            total_batch += 1
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            #loop.set_postfix(loss=loss.item())

        return running_loss / total_batch

    def train(self, epochs, is_return_wbab = False):
        train_stats = {"train_loss_sum": 0, "epoch_loss": [], "train_loss_power_two_sum":0}

        print(f"\nTraining Start ({epochs} epochs)")
        for epoch in range(epochs):
            train_loss = self.train_step()
            train_stats["train_loss_sum"] += train_loss
            train_stats["train_loss_power_two_sum"] += train_loss ** 2
            train_stats["epoch_loss"].append(train_loss)
            # print(f"Epoch {epoch + 1:02d}/{epochs} - Loss: {train_loss:.4f}")

        train_stats["avg_loss"] = train_stats["train_loss_sum"] / epochs
        train_stats["sqrt_train_loss_power_two_sum"] = math.sqrt(train_stats["train_loss_power_two_sum"])
        # print(f"\n[Summary] Total Loss: {train_stats['train_loss_sum']:.4f} | Avg Loss: {train_stats["avg_loss"]:.4f}")

        # # Optional: comment out or reduce the debug prints for model params
        # print("\n[Debug] Model param means:")
        # for name, param in self.model.named_parameters():
        #     if param.requires_grad:
        #         print(f"  {name}: {param.data.mean():.6f}")

        if is_return_wbab == False:
            return self.model.state_dict(), train_stats
        else:
            return self.model.state_dict(), train_stats, self.extract_wbab()
    
    def observe(self, epochs=5):
        train_stats = {"train_loss_sum": 0, "epoch_loss": [], "train_loss_power_two_sum":0}

        print(f"\nObservation start ({epochs} epochs)")
        for epoch in range(epochs):
            train_loss = self.train_step()
            train_stats["train_loss_sum"] += train_loss
            train_stats["train_loss_power_two_sum"] += train_loss ** 2
            train_stats["epoch_loss"].append(train_loss)
            # print(f"Epoch {epoch + 1:02d}/{epochs} - Loss: {train_loss:.4f}")

        train_stats["avg_loss"] = train_stats["train_loss_sum"] / epochs
        train_stats["sqrt_train_loss_power_two_sum"] = math.sqrt(train_stats["train_loss_power_two_sum"])
        # print(f"\n[Summary] Total Loss: {train_stats['train_loss_sum']:.4f} | Avg Loss: {train_stats["avg_loss"]:.4f}")

        return self.model.state_dict(), train_stats

    def extract_wbab(self):
        """
        Extracts the model parameters using the AdvancedModelExtractor.
        """
        
        return AdvancedModelExtractor(self.model).get_layer_dict()
    
if __name__ == "__main__":

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from lora.lora_implementation.lora_linear import LoRALinear

    # 定义 LoRAMLP 模型
    class LoRAMLP(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim, rank=4):
            super().__init__()
            self.fc1 = LoRALinear(input_dim, hidden_dim, rank=rank)
            self.relu = nn.ReLU()
            self.fc2 = LoRALinear(hidden_dim, output_dim, rank=rank)

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    # 使用 ModelTrainer 训练 LoRAMLP 模型
    def train_with_trainer(epochs=5, batch_size=64, learning_rate=1e-3, rank=4, device='cuda' if torch.cuda.is_available() else 'cpu'):
        # MNIST 数据集加载
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(torch.flatten)
        ])
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        val_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # 初始化模型
        model = LoRAMLP(input_dim=28*28, hidden_dim=256, output_dim=10, rank=rank).to(device)

        # 损失函数 & 优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # 使用 ModelTrainer 训练
        trainer = ModelTrainer(model, optimizer, criterion, train_loader, val_loader, device=device)
        state_dict, stats = trainer.train(epochs)

        return state_dict, stats
    
    train_with_trainer(epochs=5, rank=60)
