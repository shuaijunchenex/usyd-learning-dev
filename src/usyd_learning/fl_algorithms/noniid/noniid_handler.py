import sys
sys.path.insert(0, '')

import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from data_process.dataloader import DataLoaderFactory
from data_process.custom_dataset import CustomDataset

class NoniidDataHandler:
    def __init__(self, dataloader):
        """
        Args:
            dataloader (DataLoader): PyTorch DataLoader for dataset
        """
        self.dataloader = dataloader
        self.data_pool = None
        self.x_train = []
        self.y_train = []

        # Load data into memory
        self._load_data()
        self.create_data_pool()

    def _load_data(self):
        """Load data from DataLoader and store in x_train, y_train"""
        images_list, labels_list = [], []
        for images, labels in self.dataloader:
            images_list.append(images)
            labels_list.append(labels)
        
        self.x_train = torch.cat(images_list, dim=0)
        self.y_train = torch.cat(labels_list, dim=0)

    def create_data_pool(self):
        """
        Organizes dataset into a dictionary where keys are class labels (0-9),
        and values are lists of corresponding images.

        Returns:
            dict: {label: tensor(images)}
        """
        self.data_pool = {i: [] for i in range(10)}
        for i in range(10):
            self.data_pool[i] = self.x_train[self.y_train.flatten() == i]

        return self.data_pool

    @staticmethod
    def distribution_generator(distribution='mnist_lt', data_volum_list=None):
        """
        Generates the distribution pattern for data allocation.

        Args:
            distribution (str): Type of distribution ('mnist_lt' for long-tail, 'custom' for user-defined).
            data_volum_list (list): Custom data volume distribution, required if distribution='custom'.

        Returns:
            list: A nested list where each sublist represents the data volume per class for a client.
        """
        mnist_data_volum_list_lt = [
            [592, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [592, 749, 0, 0, 0, 0, 0, 0, 0, 0],
            [592, 749, 744, 0, 0, 0, 0, 0, 0, 0],
            [592, 749, 744, 875, 0, 0, 0, 0, 0, 0],
            [592, 749, 745, 876, 973, 0, 0, 0, 0, 0],
            [592, 749, 745, 876, 973, 1084, 0, 0, 0, 0],
            [592, 749, 745, 876, 974, 1084, 1479, 0, 0, 0],
            [593, 749, 745, 876, 974, 1084, 1479, 2088, 0, 0],
            [593, 749, 745, 876, 974, 1084, 1480, 2088, 2925, 0],
            [593, 750, 745, 876, 974, 1085, 1480, 2089, 2926, 5949]
        ]

        mnist_data_volum_list_one_label = [[5920, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                            [0, 6742, 0, 0, 0, 0, 0, 0, 0, 0],
                                            [0, 0, 5958, 0, 0, 0, 0, 0, 0, 0],
                                            [0, 0, 0, 6131, 0, 0, 0, 0, 0, 0],
                                            [0, 0, 0, 0, 5842, 0, 0, 0, 0, 0],
                                            [0, 0, 0, 0, 0, 5421, 0, 0, 0, 0],
                                            [0, 0, 0, 0, 0, 0, 5918, 0, 0, 0],
                                            [0, 0, 0, 0, 0, 0, 0, 6265, 0, 0],
                                            [0, 0, 0, 0, 0, 0, 0, 0, 5851, 0],
                                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 5949]]

        mnist_data_volum_balance =  [[592, 674, 595, 613, 584, 542, 591, 626, 585, 594],
                                      [592, 674, 595, 613, 584, 542, 591, 626, 585, 594],
                                      [592, 674, 595, 613, 584, 542, 591, 626, 585, 594],
                                      [592, 674, 595, 613, 584, 542, 591, 626, 585, 594],
                                      [592, 674, 595, 613, 584, 542, 591, 626, 585, 594],
                                      [592, 674, 595, 613, 584, 542, 591, 626, 585, 594],
                                      [592, 674, 595, 613, 584, 542, 591, 626, 585, 594],
                                      [593, 674, 595, 613, 584, 542, 591, 626, 585, 594],
                                      [593, 674, 595, 613, 584, 542, 591, 626, 585, 594],
                                      [593, 674, 595, 613, 584, 542, 591, 626, 585, 594]]

        if distribution == "mnist_lt":
            return mnist_data_volum_list_lt
        if distribution == 'mnist_data_volum_balance':
            return mnist_data_volum_balance
        if distribution == 'mnist_lt_one_label':
            return mnist_data_volum_list_one_label
        elif distribution == "custom":
            if data_volum_list is None:
                raise ValueError("Custom distribution requires 'data_volum_list'.")
            return data_volum_list
        else:
            raise ValueError("Invalid distribution type. Choose 'mnist_lt' or 'custom'.")

    def generate_noniid_data(self, data_volum_list=None, verify_allocate=True, distribution="mnist_lt", batch_size=64, shuffle=False, num_workers=0):
        """
        Distributes imbalanced data to different clients based on predefined patterns and returns a list of DataLoader for each client.

        Args:
            data_volum_list (list): A list containing data volume for different classes (used only if distribution="custom").
            verify_allocate (bool): Whether to print allocation results.
            distribution (str): Default is "mnist_lt", supports different distributions.
            batch_size (int): Number of samples per batch for the DataLoader.
            shuffle (bool): Whether to shuffle the data in the DataLoader.
            num_workers (int): Number of worker threads for the DataLoader.

        Returns:
            list: A list of DataLoader objects, each corresponding to one client's data.
        """
        # Ensure data_pool is initialized
        if self.data_pool is None:
            raise ValueError("Data pool is not created. Call create_data_pool() first.")

        # Get the distribution pattern
        distribution_pattern = self.distribution_generator(distribution, data_volum_list)

        # Allocate data for each client
        allocated_data = []
        for client_idx, client_data in enumerate(distribution_pattern):
            client_images = []
            client_labels = []
            
            # Track client's distribution for verification
            client_distribution = {}
            
            # Collect data for this client from each class
            for label_idx, num_samples in enumerate(client_data):
                if num_samples > 0:
                    if num_samples > len(self.data_pool[label_idx]):
                        raise ValueError(f"Not enough samples for class {label_idx}: requested {num_samples}, available {len(self.data_pool[label_idx])}")
                    
                    # Select and remove data from pool
                    selected_data = self.data_pool[label_idx][:num_samples]
                    client_images.extend(selected_data)
                    client_labels.extend([label_idx] * num_samples)
                    self.data_pool[label_idx] = self.data_pool[label_idx][num_samples:]
                    
                    # Update distribution tracking
                    client_distribution[label_idx] = num_samples
            
            # Skip clients with no data
            if len(client_images) == 0:
                continue
                
            # Store client data
            allocated_data.append({
                'images': client_images,
                'labels': client_labels,
                'distribution': client_distribution
            })
            
            # Verify allocation results
            if verify_allocate:
                print(f"Client {client_idx + 1} distribution:")
                for label, count in client_distribution.items():
                    print(f"  Label {label}: {count} samples")
                print(f"  Total samples: {len(client_images)}")

        # Create DataLoader for each client
        train_loaders = []
        
        for client_idx, client_data in enumerate(allocated_data):
            # Skip if client has no data
            if len(client_data['images']) == 0:
                continue
            
            # Create dataset (without transform)
            train_dataset = CustomDataset(
                client_data['images'], 
                client_data['labels'], 
                transform=None  # No transform applied
            )
            
            # Create DataLoader for this client
            train_loader = DataLoaderFactory.create_loader(
                train_dataset, 
                batch_size=batch_size,  # Ensure batch_size doesn't exceed dataset size
                shuffle=shuffle, 
                num_workers=num_workers
            )
            
            train_loaders.append(train_loader)

        return train_loaders


# import torch
# from torch.utils.data import DataLoader, Subset
# from typing import List, Dict, Optional
# import numpy as np

# class NoniidDataHandler:
#     def __init__(self, dataloader: DataLoader, seed: int = 42):
#         """
#         初始化非独立同分布数据处理器

#         Args:
#             dataloader (DataLoader): PyTorch数据加载器
#             seed (int): 随机数种子，默认为42
#         """
#         torch.manual_seed(seed)
#         np.random.seed(seed)

#         self.dataloader = dataloader
#         self.data_pool = None
#         self.x_train = None
#         self.y_train = None

#         # 加载数据到内存
#         self._load_data()
#         self.create_data_pool()

#     def _load_data(self):
#         """使用Subset加载数据"""
#         images_list, labels_list = [], []
#         for images, labels in self.dataloader:
#             images_list.append(images)
#             labels_list.append(labels)
        
#         self.x_train = torch.cat(images_list, dim=0)
#         self.y_train = torch.cat(labels_list, dim=0)

#     def create_data_pool(self):
#         """
#         将数据集组织为以类标签为键的字典

#         Returns:
#             dict: {label: Subset(images)}
#         """
#         self.data_pool = {i: [] for i in range(10)}
#         for i in range(10):
#             class_indices = torch.nonzero(self.y_train.flatten() == i).squeeze()
#             self.data_pool[i] = Subset(self.x_train, class_indices)

#         return self.data_pool

#     @staticmethod
#     def distribution_generator(distribution='mnist_lt', data_volum_list=None):
#         """
#         生成数据分配分布模式

#         Args:
#             distribution (str): 分布类型
#             data_volum_list (list): 自定义数据量分布

#         Returns:
#             list: 嵌套列表，表示每个客户端的类别数据量
#         """
#         mnist_data_volum_list_lt = [
#             [592, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#             [592, 749, 0, 0, 0, 0, 0, 0, 0, 0],
#             [592, 749, 744, 0, 0, 0, 0, 0, 0, 0],
#             [592, 749, 744, 875, 0, 0, 0, 0, 0, 0],
#             [592, 749, 745, 876, 973, 0, 0, 0, 0, 0],
#             [592, 749, 745, 876, 973, 1084, 0, 0, 0, 0],
#             [592, 749, 745, 876, 974, 1084, 1479, 0, 0, 0],
#             [593, 749, 745, 876, 974, 1084, 1479, 2088, 0, 0],
#             [593, 749, 745, 876, 974, 1084, 1480, 2088, 2925, 0],
#             [593, 750, 745, 876, 974, 1085, 1480, 2089, 2926, 5949]
#         ]

#         if distribution == "mnist_lt":
#             return mnist_data_volum_list_lt
#         elif distribution == "custom":
#             if data_volum_list is None:
#                 raise ValueError("Custom distribution requires 'data_volum_list'.")
#             return data_volum_list
#         else:
#             raise ValueError("Invalid distribution type. Choose 'mnist_lt' or 'custom'.")

#     def generate_noniid_data(self, 
#                               data_volum_list: Optional[List[int]] = None, 
#                               distribution: str = "mnist_lt"):
#         """
#         根据预定义模式分配不平衡数据给不同客户端

#         Args:
#             data_volum_list (list): 不同类别的数据量列表
#             distribution (str): 分布类型，默认为"mnist_lt"

#         Returns:
#             list of dict: 每个字典包含特定客户端的数据和标签
#         """
#         if self.data_pool is None:
#             raise ValueError("Data pool is not created. Call create_data_pool() first.")

#         distribution_pattern = self.distribution_generator(distribution, data_volum_list)
#         allocated_data = [{'data': [], 'labels': [], 'distribution': []} for _ in range(len(distribution_pattern))]

#         for client_idx, client_data in enumerate(distribution_pattern):
#             for label_idx, num_samples in enumerate(client_data):
#                 if num_samples > len(self.data_pool[label_idx]):
#                     raise ValueError(f"Not enough samples for class {label_idx}")

#                 # 使用torch.split替代手动切分
#                 split_indices = torch.randperm(len(self.data_pool[label_idx]))[:num_samples]
#                 selected_data = torch.stack([self.data_pool[label_idx][i] for i in split_indices])
                
#                 allocated_data[client_idx]['data'].append(selected_data)
#                 allocated_data[client_idx]['labels'].append(torch.full((num_samples,), label_idx, dtype=torch.long))

#             allocated_data[client_idx]['distribution'].append(client_data)

#         return self.validate_data_distribution(allocated_data)

#     def validate_data_distribution(self, allocated_data):
#         """
#         验证和打印数据分配的详细信息

#         Args:
#             allocated_data (list): 分配的数据列表

#         Returns:
#             list: 验证后的数据分配
#         """
#         print("数据分配验证报告：")
#         total_samples_global = 0
#         client_class_summary = []

#         for idx, client_data in enumerate(allocated_data):
#             print(f"\n客户端 {idx + 1} 数据分布:")
#             client_total_samples = 0
#             client_class_counts = [0] * 10

#             for label_idx, data_tensor in enumerate(client_data['data']):
#                 samples_count = len(data_tensor)
#                 client_total_samples += samples_count
#                 client_class_counts[label_idx] = samples_count
                
#                 print(f"  类别 {label_idx}: {samples_count} 样本")

#             total_samples_global += client_total_samples
#             client_class_summary.append(client_class_counts)
#             print(f"  客户端 {idx + 1} 总样本数: {client_total_samples}")

#         print(f"\n总样本数: {total_samples_global}")
#         return allocated_data