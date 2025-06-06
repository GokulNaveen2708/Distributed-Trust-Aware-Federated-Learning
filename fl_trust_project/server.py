import flwr as fl
import torch
from typing import List, Tuple, Dict, Optional
from model import SimpleCNN
from trust_manager import TrustManager
import numpy as np
from collections import defaultdict


def weighted_average(results):
    """
    Aggregate client metrics into a dict. Flower (≥1.9) expects the
    aggregation fn to return Metrics, i.e. Dict[str, Scalar].

    Parameters
    ----------
    results : List[Tuple[int, Dict[str, float]]]
        Each item is (num_examples, {"accuracy": value, ...})
    """
    total_examples = 0
    total_acc = 0.0

    for num_examples, metrics in results:
        total_examples += num_examples
        total_acc += num_examples * metrics["accuracy"]

    aggregated_accuracy = total_acc / total_examples
    return {"accuracy": aggregated_accuracy}  # <-- return a dict


class TrustFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, gamma: float, beta: float, **kwargs):
        super().__init__(**kwargs)
        self.trust_mgr = TrustManager(gamma, beta)
        self.client_history = defaultdict(lambda: {"round": [], "loss": [], "accuracy": []})
        self.round = 0

    def aggregate_fit(self, rnd, results, failures):
        self.round = rnd
        # Compute reputation metric (Euclidean distance) for each client
        aggregated, _ = super().aggregate_fit(rnd, results, failures)
        if not aggregated:
            return None, {}
        global_weights = fl.common.parameters_to_ndarrays(aggregated)
        trusted_results = []
        for client_proxy, fit_res in results:
            cid = client_proxy.cid  # correct way to get client id
            local_weights = fl.common.parameters_to_ndarrays(fit_res.parameters)

            # Compute Euclidean distance between this client's weights and the global model
            rep = np.linalg.norm(
                np.concatenate([w.flatten() for w in local_weights]) -
                np.concatenate([w.flatten() for w in global_weights])
            )

            trust = self.trust_mgr.update(cid, rep)
            print(f"[Round {rnd}] Client {cid} rep={rep:.4f} trust={trust:.4f}")

            # Keep only trusted clients for re-aggregation
            if self.trust_mgr.is_trusted(cid):
                trusted_results.append((client_proxy, fit_res))

        # Re‑aggregate using only trusted clients
        return super().aggregate_fit(rnd, trusted_results, failures)

    def aggregate_evaluate(self, rnd, results, failures):
        # --- log per-client metrics ---
        for client_proxy, eval_res in results:
            cid = client_proxy.cid
            self.client_history[cid]["round"].append(rnd)
            self.client_history[cid]["loss"].append(eval_res.loss)
            acc = eval_res.metrics.get("accuracy", float("nan"))
            self.client_history[cid]["accuracy"].append(acc)
        # --- keep normal aggregation behaviour ---
        return super().aggregate_evaluate(rnd, results, failures)



def start_server(rounds: int, gamma: float, beta: float):
    model = SimpleCNN()
    strategy = TrustFedAvg(
        gamma=gamma,
        beta=beta,
        fraction_fit=1.0,
        min_fit_clients=2,
        min_available_clients=2,
        on_fit_config_fn=lambda rnd: {"epoch": 1},
        evaluate_metrics_aggregation_fn=weighted_average,  # ← rename
        initial_parameters=fl.common.ndarrays_to_parameters(
            [p.detach().numpy() for p in model.state_dict().values()]
        ),
    )

    hist = fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=rounds),
        strategy=strategy,
    )

    # Save history for later plotting
    import pickle, pathlib
    pathlib.Path("logs").mkdir(exist_ok=True)
    with open("logs/history.pkl", "wb") as f:
        pickle.dump(hist, f)
    pathlib.Path("logs").mkdir(exist_ok=True)
    # ---- dump per-client history ----
    plain_hist = {cid: rec for cid, rec in strategy.client_history.items()}
    with open("logs/client_history.pkl", "wb") as f:
        pickle.dump(plain_hist, f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.7)
    parser.add_argument("--beta", type=float, default=0.2)
    args = parser.parse_args()
    start_server(args.rounds, args.gamma, args.beta)
