import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int = 4, scaling: float = 0.5, use_bias: bool = True, pretrained_weight: torch.Tensor = None, lora_mode: str = "standard", device: str = "cpu"):
        """
        Linear layer with Low-Rank Adaptation (LoRA).

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            rank (int): Rank of the low-rank decomposition.
            alpha (float): Scaling factor for the low-rank update.
            use_bias (bool): If True, includes a bias parameter.
            pretrained_weight (Optional[torch.Tensor]): If provided, initializes self.weight with this tensor.
            lora_mode (str): Determines the LoRA inference mode. Options: "standard", "alternate", etc.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = scaling
        self.lora_mode = lora_mode

        # Initialize weight matrix
        if pretrained_weight != None:
            # If pretrained weight is provided, use it directly (set requires_grad=False to freeze)
            self.weight = nn.Parameter(pretrained_weight, requires_grad=False)
            self.weight.requires_grad = False
        else:
            # Xavier Uniform initialization for weight matrix
            self.weight = nn.Parameter(torch.empty(out_features, in_features))
            nn.init.zeros_(self.weight)  # Xavier Uniform initialization
            self.weight.requires_grad = False  # Ensure weight is not trainable

        # Initialize bias
        self.bias = nn.Parameter(torch.zeros(out_features)) if use_bias else None
        self.bias.requires_grad = False  # Ensure bias is not trainable

        # LoRA trainable parameters with Xavier initialization
        self.lora_A = nn.Parameter(torch.empty(out_features, rank).to(device))
        self.lora_B = nn.Parameter(torch.empty(rank, in_features).to(device))
        nn.init.xavier_uniform_(self.lora_A)

    def set_lora_mode(self, mode: str):
        if mode not in ["standard", "lora_only", "lora_disabled", "scaling"]:
            raise ValueError(f"Unsupported lora_mode: {mode}")
        self.lora_mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the LoRALinear layer."""
        
        # Compute low-rank update           
        delta_weight = torch.matmul(self.lora_A, self.lora_B)
        
        if self.lora_mode == "standard":
            effective_weight = self.weight + delta_weight
        elif self.lora_mode == "lora_only":
            effective_weight = delta_weight
        elif self.lora_mode == "lora_disabled":
            effective_weight = self.weight
        elif self.lora_mode == "scaling":
            effective_weight = (1 - self.scaling) * self.weight + self.scaling * delta_weight
        else:
            raise ValueError(f"Unsupported lora_mode: {self.lora_mode}")
        
        return F.linear(x, effective_weight, self.bias)


if __name__ == "__main__":
    
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    # 假设你已实现 LoRALinear
    # from your_module import LoRALinear

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

    def train_lora_mlp_mnist(epochs=5, batch_size=64, learning_rate=1e-3, rank=4, device='cuda' if torch.cuda.is_available() else 'cpu'):
        # 数据预处理
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

        # 模型实例化
        model = LoRAMLP(28*28, 256, 10, rank=rank).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            model.train()
            total_loss = 0
            correct = 0
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                data = data.view(data.size(0), -1)  # 展平 28x28 图像

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()

            avg_loss = total_loss / len(train_loader)
            accuracy = correct / len(train_loader.dataset)
            print(f"Epoch {epoch+1}: Train Loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}")

        # 测试集评估
        model.eval()
        test_correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                data = data.view(data.size(0), -1)
                output = model(data)
                pred = output.argmax(dim=1)
                test_correct += pred.eq(target).sum().item()

        test_acc = test_correct / len(test_loader.dataset)
        print(f"Test Accuracy: {test_acc:.4f}")

    train_lora_mlp_mnist(epochs=5, rank=60)