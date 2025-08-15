import os
from startup_init import startup_init_path
startup_init_path(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F

from usyd_learning.fl_algorithms import FedAggregatorFactory, FedAggregatorArgs
from usyd_learning.ml_algorithms import MSLoRAConv2d, MSLoRALinear

from copy import deepcopy

class LoRACNN(nn.Module):
    def __init__(self, num_classes=10, r=4, lora_alpha=16, lora_dropout=0.1):
        super(LoRACNN, self).__init__()
        self.conv1 = MSLoRAConv2d(1, 32, kernel_size=3, padding=1, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        self.conv2 = MSLoRAConv2d(32, 64, kernel_size=3, padding=1, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = MSLoRALinear(64 * 7 * 7, 128, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        self.fc2 = MSLoRALinear(128, num_classes, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [B, 32, 14, 14]
        x = self.pool(F.relu(self.conv2(x)))  # [B, 64, 7, 7]
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def test_lora_cnn_aggregation_and_distribution():
    # 创建 r=1,2,3 的模型
    models = [LoRACNN(r=r) for r in [1, 2, 3]]

    # 提取 state_dict 并加入随机扰动
    state_dicts = []
    for model in models:
        sd = deepcopy(model.state_dict())
        for k in sd:
            if sd[k].dtype == torch.float32:
                sd[k] += torch.randn_like(sd[k]) * 0.01
        state_dicts.append(sd)

    agg_cfg = {
    "method": "rbla",
    "device": "cpu",
    }

    agg_args = FedAggregatorArgs(agg_cfg)

    aggregator = FedAggregatorFactory.create_aggregator(agg_args)

    data_dict = {"state_dicts":state_dicts, "weights":[0.2, 0.3, 0.5]}

    global_sd = aggregator.aggregate(data_dict)

    # 打印聚合结果中的 LoRA 参数维度（示例）
    print("✅ Aggregated LoRA parameter shapes:")
    for k, v in global_sd.items():
        if "lora" in k:
            print(f"{k}: {v.shape}")

    # 分发回每个 client，并验证维度匹配
    for i, model in enumerate(models):
        local_sd = model.state_dict()
        new_sd = aggregator.broadcast_lora_state_dict(global_sd, local_sd)
        model.load_state_dict(new_sd, strict=False)

        # 检查某个代表性的 LoRA A/B 的形状
        for k in new_sd:
            if "lora_A" in k:
                expected_r = local_sd[k].shape[0]
                actual_r = new_sd[k].shape[0]
                assert expected_r == actual_r, f"[Client {i}] A rank mismatch!"
            if "lora_B" in k:
                expected_r = local_sd[k].shape[1]
                actual_r = new_sd[k].shape[1]
                assert expected_r == actual_r, f"[Client {i}] B rank mismatch!"

        print(f"✅ Client {i+1} (rank={i+1}) received LoRA weights successfully.")

# ==== 执行测试 ====
test_lora_cnn_aggregation_and_distribution()
