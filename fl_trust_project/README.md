# Trust & Reputation Federated Learning Demo

This repo contains a minimal, self‑contained prototype that implements the **Trust & Reputation (T&R) mechanism** described in your paper:

* **Framework:** Flower (`flwr`)
* **Dataset:** PathMNIST (or any image dataset split among clients)
* **Model:** Simple CNN
* **Attack:** Label‑flipping for two clients
* **Mechanism:** Dynamic trust decay + reputation scoring; clients whose trust falls below `beta` are excluded from aggregation.

> ⚠️ **Prototype**: Designed to run on a laptop with CPU; tweak epochs/batch size for real experiments.

## Quick Start

```bash
# 1. Install deps (preferably in a venv)
pip install -r requirements.txt

# 2. Start the Flower server
python server.py --rounds 10 --gamma 0.7 --beta 0.2

# 3. In *separate* terminals, start 10 clients
for cid in $(seq 0 9); do
    python client.py --cid $cid &
done
wait
```

Clients 1 and 3 will automatically perform label‑flipping.

After training, metrics are logged to `logs/` and plots are saved in `plots/`.

## File Layout

```
fl_trust_project/
├── client.py            # Flower client (benign or malicious)
├── model.py             # CNN definition
├── server.py            # Custom TrustFedAvgStrategy
├── trust_manager.py     # Trust score bookkeeping
├── utils.py             # Loading PathMNIST + partitioning helper
├── requirements.txt
└── README.md
```
