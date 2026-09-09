import flwr as fl
from Model import models_vit, models_mae
import random

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg

# num_weights:
# encoder: 296
# encoder+decoder: 398

class CustomStrategy(fl.server.strategy.FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.round = 1

    def configure_fit(self, parameters, client_manager, server_round,):
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
            #num_clients=sample_size, min_num_clients=1
        )

        return [(client, fl.common.FitIns(parameters, {'round': self.round})) for client in clients]

    # config evaluation
    def configure_evaluate(self, server_round, parameters, client_manager):
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        config['round'] = self.round
        self.round += 1
        fit_ins = fl.common.FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
            #num_clients=sample_size, min_num_clients=1
        )
        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    # aggregate
    def aggregate_fit(self, rnd, results, failures):
        # Aggregate results
        aggregated_outputs = super().aggregate_fit(rnd, results, failures)
        # Return aggregated results
        return aggregated_outputs    


def weighted_average(metrics):
    '''
    metrics: List[Tuple[num_samples, Metric_Dict]]
    '''
    if len(metrics) == 0:
        return None
    if 'accuracy' in metrics[0][1]:
        # Multiply accuracy of each client by number of examples used
        accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]
        # Aggregate and return custom metric (weighted average)
        return {"accuracy": sum(accuracies) / sum(examples)}
    elif 'loss' in metrics[0][1]:
        losses = [num_examples * m["loss"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]
        return {"loss": sum(losses) / sum(examples)}


def launch_server(num_epochs, strat=fl.server.strategy.FedAvg, server_addr="0.0.0.0:8080", **kwargs):
    strategy = strat(
        fit_metrics_aggregation_fn = weighted_average, 
        evaluate_metrics_aggregation_fn = weighted_average,
        **kwargs,
    )

    fl.server.start_server(
        server_address=server_addr,
        config = fl.server.ServerConfig(num_rounds=num_epochs),
        strategy = strategy,
        grpc_max_message_length=0x7fffffff,
    )