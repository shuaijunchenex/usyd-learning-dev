from abc import ABC, abstractmethod
from typing import Any

class BaseModelExtractor(ABC):
    """
    模型参数提取器接口，定义统一提取、注册、导出方法。
    """

    @abstractmethod
    def get_layer_dict(self) -> dict:
        """
        获取层级结构化的参数字典
        """
        pass

    @abstractmethod
    def save_to_json(self, filepath: str) -> None:
        """
        将权重保存为JSON
        """
        pass

    @abstractmethod
    def save_to_npz(self, filepath: str) -> None:
        """
        将权重保存为NPZ
        """
        pass

    @abstractmethod
    def register_custom_handler(self, layer_type: type, handler_fn: Any) -> None:
        """
        注册自定义提取器
        """
        pass

    @abstractmethod
    def set_model(self, model: Any) -> None:
        """
        更换模型并重新提取
        """
        pass
