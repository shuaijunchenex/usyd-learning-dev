from __future__ import annotations
from typing import Dict, List, Any, Optional
from collections import defaultdict

import torch
from fed_strategy.server_strategy import ServerStrategy
from fl_algorithms.aggregation.fed_aggregator_facotry import FedAggregatorFactory
from fl_algorithms.selection.fed_client_selector_factory import FedClientSelectorFactory
from model_trainer.model_evaluator import ModelEvaluator


class FedAvgServerStrategy(ServerStrategy):
    """
    Federated Averaging (FedAvg) server-side strategy.

    Lifecycle:
      - create(args, fn) -> _create_inner(args)
      - run(): 1 round orchestration (collect -> aggregate -> broadcast)
      - evaluate(): evaluate global model on server-side data
    """

    def __init__(self, server) -> None:
        super().__init__(server)
        self._strategy_type = "FedAvg"

        # 可选：在 create() 之前先占位，真正初始化放在 _create_inner
        self.model_evaluator: Optional[ModelEvaluator] = None

    # --------------------------------------------------
    def _create_inner(self, args) -> None:
        # self.model_evaluator = ModelEvaluator(
        #     args.model(),            
        #     args.data_loader(),      
        #     args.loss_func(),        
        #     args.device,             
        # )
        # self.selection_method = FedClientSelectorFactory()
        # self.aggregation_method = FedAggregatorFactory()
        self._server_node.node_var.client_selection
        self._server_node.node_var.aggregation_method

    # 保持向后兼容：等价于调用 create()
    def create_server_strategy(self, server_strategy_args, fn=None) -> FedAvgServerStrategy:
        """
        Backward-compatible factory-style entry.
        """
        return self.create(server_strategy_args, fn)

    # --------------------------------------------------
    def aggregation(self, client_weights: List[Any]) -> Dict[str, torch.Tensor]:
        """
        FedAvg 聚合。
        支持的输入形式（自动识别）：
          1) List[Dict[str, Tensor]]                      -> 等权平均
          2) List[Tuple[Dict[str, Tensor], weight]]       -> 按 weight 加权
          3) List[Dict{ 'state_dict':..., 'num_samples':... }]
             或 { 'weights':..., 'n':... } 等常见命名    -> 按 num_samples/n/weight 加权

        返回：
          聚合后的 state_dict（键->Tensor）
        """
        self._assert_created()
        if not client_weights:
            raise ValueError("aggregation() got empty client_weights")

        # 解析为统一的 [(state_dict, weight), ...]
        parsed: List[tuple[Dict[str, torch.Tensor], float]] = []
        for item in client_weights:
            sd: Dict[str, torch.Tensor]
            w: Optional[float] = None

            # 形式 2: (state_dict, weight)
            if isinstance(item, tuple) and len(item) == 2:
                sd, w = item  # type: ignore[assignment]
            # 形式 3: dict 携带 state_dict + 权重字段
            elif isinstance(item, dict) and (
                "state_dict" in item or "weights" in item or "updated_weights" in item
            ):
                sd = item.get("state_dict", item.get("weights", item.get("updated_weights")))
                # 尝试解析常见权重字段
                for key in ("num_samples", "n", "weight", "_weight", "samples", "data_sample_num"):
                    if key in item:
                        try:
                            w = float(item[key])
                            break
                        except Exception:
                            pass
                if w is None:
                    w = 1.0
            # 形式 1: 纯 state_dict
            elif isinstance(item, dict):
                sd = item
                w = 1.0
            else:
                raise TypeError(f"Unsupported client_weights item type: {type(item)}")

            if not isinstance(sd, dict):
                raise TypeError("Parsed state_dict is not a dict")

            parsed.append((sd, float(w)))

        # 归一化权重
        total_w = sum(w for _, w in parsed)
        if total_w <= 0:
            # 回退到等权
            total_w = float(len(parsed))
            parsed = [(sd, 1.0) for sd, _ in parsed]

        # 聚合（加权平均）
        agg: Dict[str, torch.Tensor] = defaultdict(lambda: None)
        for sd, w in parsed:
            alpha = w / total_w
            for k, v in sd.items():
                if not torch.is_tensor(v):
                    raise TypeError(f"State dict value for key {k} must be a Tensor, got {type(v)}")
                if agg[k] is None:
                    agg[k] = v.detach().clone() * alpha
                else:
                    agg[k] = agg[k] + v.detach() * alpha

        # 将 defaultdict 转回普通 dict
        return {k: v for k, v in agg.items()}

    def broadcast(self, aggregated_weights: Dict[str, torch.Tensor]) -> None:
        """
        将聚合后的全局权重应用到服务器模型，并尝试广播到客户端。
        """
        self._assert_created()
        model = self.server.node_var.model()

        # 1) 先应用到全局模型
        if hasattr(model, "load_state_dict"):
            model.load_state_dict(aggregated_weights)
        else:
            raise AttributeError("Server model has no load_state_dict(...)")

        # 2) 若 server 提供广播 API，则广播
        #    这里做兼容判断，你可替换为自己的方法名
        if hasattr(self.server, "broadcast_global_weights"):
            self.server.broadcast_global_weights(aggregated_weights)
        elif hasattr(self.server, "broadcast"):
            self.server.broadcast(aggregated_weights)
        else:
            # 如果没有广播 API，就保持 silent；或者在这里 raise 由你选择
            pass

    # --------------------------------------------------
    def run(self) -> Dict[str, Any]:
        """
        典型的单轮联邦流程（示例）：
          1) 收集客户端更新
          2) 聚合
          3) 广播新全局模型
          4) 返回本轮摘要
        你可按你的 server API 替换下面的占位方法名。
        """
        self._assert_created()

        # 1) 收集客户端更新
        if hasattr(self.server, "collect_client_updates"):
            client_updates = self.server.collect_client_updates()
        elif hasattr(self.server, "gather"):
            client_updates = self.server.gather()
        else:
            raise AttributeError("Server has no method to collect client updates")

        # 2) 聚合
        aggregated = self.aggregation(client_updates)

        # 3) 广播
        self.broadcast(aggregated)

        # 4) 可选：记录/返回摘要
        summary = {
            "num_clients": len(client_updates),
            "keys": list(aggregated.keys()),
        }
        return summary

    def evaluate(self) -> None:
        """
        在服务器侧对当前全局模型进行评估。
        """
        self._assert_created()
        if self.model_evaluator is None:
            raise RuntimeError("ModelEvaluator not initialized. Call create(...) first.")
        self.model_evaluator.evaluate()
        self.model_evaluator.print_results()
