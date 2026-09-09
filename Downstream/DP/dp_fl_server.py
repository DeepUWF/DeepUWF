"""
Federated Learning Server for DP FL Training
"""

import flwr as fl
from flwr.common import (
    FitRes,
    EvaluateRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.strategy import FedAvg
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from typing import Callable, Dict, List, Optional, Tuple, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)


def weighted_average(metrics: List[Tuple[int, Dict[str, Any]]]) -> Dict[str, float]:
    """
    Aggregate metrics from clients using weighted average
    
    Args:
        metrics: List of (num_examples, metrics_dict) from clients
        
    Returns:
        aggregated_metrics: Dict with aggregated metrics
    """
    if len(metrics) == 0:
        return {}
    
    # 权重平均
    total_examples = sum(num_examples for num_examples, _ in metrics)
    
    weighted_metrics = {}
    for metric_key in metrics[0][1].keys():
        if metric_key in ["client_id"]:
            continue
            
        weighted_sum = sum(
            num_examples * m[metric_key]
            for num_examples, metrics_dict in metrics
            if metric_key in (m := metrics_dict)
        )
        
        weighted_metrics[metric_key] = (
            weighted_sum / total_examples if total_examples > 0 else 0.0
        )
    
    logger.info(f"Aggregated metrics: {weighted_metrics}")
    return weighted_metrics


class DPFedAvgStrategy(FedAvg):
    """
    Custom FedAvg strategy with DP-specific configurations
    """
    
    def __init__(
        self,
        *,
        partial_weight_fraction: float = 1.0,
        **kwargs
    ):
        """
        Args:
            partial_weight_fraction: 部分权重共享比例
            **kwargs: 传递给FedAvg的其他参数
        """
        super().__init__(**kwargs)
        self.partial_weight_fraction = partial_weight_fraction
        self.round = 0
        
    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, fl.common.FitIns]]:
        """
        Configure training round
        """
        config = {
            "round": server_round,
            "partial_weight_fraction": self.partial_weight_fraction,
            "learning_rate": max(0.001 * (0.99 ** server_round), 0.00001),  # 衰减学习率
        }
        
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=min_num_clients,
        )
        
        fit_ins = fl.common.FitIns(parameters, config)
        
        logger.info(f"Round {server_round}: Sampled {len(clients)} clients for training")
        logger.info(f"Learning rate: {config['learning_rate']:.6f}")
        
        return [(client, fit_ins) for client in clients]
    
    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, fl.common.EvaluateIns]]:
        """
        Configure evaluation round
        """
        config = {
            "round": server_round,
            "partial_weight_fraction": self.partial_weight_fraction,
        }
        
        sample_size, min_num_clients = self.num_eval_clients(
            client_manager.num_available()
        )
        
        if sample_size == 0:
            return []
        
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=min_num_clients,
        )
        
        evaluate_ins = fl.common.EvaluateIns(parameters, config)
        
        return [(client, evaluate_ins) for client in clients]
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate training results from clients
        """
        logger.info(f"Round {server_round}: Aggregating {len(results)} training results")
        
        # 使用FedAvg聚合
        aggregated_params, metrics = super().aggregate_fit(
            server_round, results, failures
        )
        
        return aggregated_params, metrics
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """
        Aggregate evaluation results from clients
        """
        logger.info(f"Round {server_round}: Aggregating {len(results)} evaluation results")
        
        # 提取指标
        metrics_list = [
            (res.num_examples, res.metrics)
            for _, res in results
            if res.metrics is not None
        ]
        
        # 加权平均
        aggregated_metrics = weighted_average(metrics_list)
        
        # 返回loss（如果有）
        loss = aggregated_metrics.get("val_loss", 0.0)
        
        return loss, aggregated_metrics


def run_federated_server(
    server_address: str = "0.0.0.0:8080",
    num_rounds: int = 10,
    num_clients: int = 5,
    clients_per_round: int = 3,
    partial_weight_fraction: float = 1.0,
    min_available_clients: int = 2,
):
    """
    Launch federated learning server
    
    Args:
        server_address: Server address (e.g., "0.0.0.0:8080")
        num_rounds: 总轮数
        num_clients: 总客户端数
        clients_per_round: 每轮选择的客户端数
        partial_weight_fraction: 部分权重共享比例
        min_available_clients: 最少可用客户端数
    """
    
    logger.info(f"Starting Federated Learning Server")
    logger.info(f"Server address: {server_address}")
    logger.info(f"Number of rounds: {num_rounds}")
    logger.info(f"Clients per round: {clients_per_round}")
    logger.info(f"Partial weight fraction: {partial_weight_fraction}")
    
    # 初始化策略
    strategy = DPFedAvgStrategy(
        fraction_fit=clients_per_round / num_clients,
        fraction_evaluate=0.5,  # 评估时选择50%的客户端
        min_fit_clients=min_available_clients,
        min_evaluate_clients=min_available_clients,
        min_available_clients=min_available_clients,
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
        partial_weight_fraction=partial_weight_fraction,
    )
    
    # 启动服务器
    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        grpc_max_message_length=0x7fffffff,
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_address", type=str, default="0.0.0.0:8080")
    parser.add_argument("--num_rounds", type=int, default=50)
    parser.add_argument("--num_clients", type=int, default=5)
    parser.add_argument("--clients_per_round", type=int, default=3)
    parser.add_argument("--partial_weight_fraction", type=float, default=1.0)
    parser.add_argument("--min_available_clients", type=int, default=2)
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    run_federated_server(
        server_address=args.server_address,
        num_rounds=args.num_rounds,
        num_clients=args.num_clients,
        clients_per_round=args.clients_per_round,
        partial_weight_fraction=args.partial_weight_fraction,
        min_available_clients=args.min_available_clients,
    )
