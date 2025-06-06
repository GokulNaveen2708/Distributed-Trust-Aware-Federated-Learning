# 🔐 Enhancing Federated Learning Security with Trust & Reputation Mechanisms

Federated Learning (FL) enables decentralized model training while preserving data privacy — ideal for sectors like healthcare and finance. But what happens when untrustworthy clients poison the process?

This project introduces a **Trust and Reputation-based approach** that extends FedAvg to dynamically **filter malicious or low-quality clients**, boosting model reliability and convergence in adversarial environments.

---

## 🧠 Motivation

- **Problem**: FL is vulnerable to poisoning, inference attacks, and client unreliability.
- **Impact**: Can compromise performance in critical domains like healthcare and autonomous systems.
- **Goal**: Ensure only trustworthy clients influence the global model by scoring and filtering them based on behavior.

---

## ⚙️ Key Features

- ✅ **Trust & Reputation Scoring**: Based on Euclidean distance from global model.
- 🧹 **Dynamic Client Filtering**: Clients with low trust (below threshold β) are excluded.
- 🧪 **Attack Simulation**: Label-flipping adversaries introduced to validate robustness.
- 📉 **Trust Decay**: Smooth updates over time, enabling recovery from occasional noise.

---

## 🧪 Experimental Setup

- 🔍 **Framework**: [Flower FL](https://flower.dev/) for client-server orchestration.
- 🧬 **Dataset**: [PathMNIST](https://medmnist.com/) — 100K+ pathology images (9-class).
- 🤖 **Model**: CNN with ReLU, MaxPool, and FC layers.
- 💥 **Attack Type**: Label flipping and gradient perturbations.

**Hyperparameters**:
- Learning Rate: `0.001` with 0.95 decay
- Batch Size: `32`
- Trust Smoothing: `γ = 0.8`
- Trust Threshold: `β = 0.6`

---

## 📈 Results

| Scenario              | Accuracy (%) | Observation                                  |
|-----------------------|--------------|----------------------------------------------|
| Baseline (FedAvg)     | ~80%         | Poisoned clients degraded model performance  |
| Trust-Enhanced FedAvg | ~87%         | Malicious clients filtered, improved accuracy|

### 🔍 Additional Insights

- 📉 **Global loss dropped** from 1.0 to near 0.02 in just 3 rounds.
- 🧑‍⚕️ **Benign clients** maintained steady trust; adversaries showed steep trust decay.
- 🛰️ **Low overhead** in communication and computation.

---

## 📦 System Components

- **Trust Manager**: Maintains smoothed trust scores and enforces filtering.
- **Client Reputation Module**: Computes deviation from global model post-training.
- **Enhanced Protocol**: Transmits both model weights & trust metrics.
- **Logging System**: Tracks participation status, trust evolution, and attack isolation.

---

## 🔬 Evaluation Metrics

- 🔹 Client-wise accuracy and loss
- 🔹 Global model convergence
- 🔹 Trust score evolution over time
- 🔹 Comparison under benign vs. adversarial conditions

---

## 🚧 Future Work

- 🔐 Add cryptographic secure aggregation
- 🌐 Scale to 1000+ clients for IoT-scale simulation
- 🤖 Handle adaptive adversaries and sybil attacks
- 📡 Optimize further for edge deployments
- 🪞 Incorporate explainability into trust scores

---

## 👥 Team

- Gokula Chapala  
- Yasiru Karunawansa  
- Dhairya Lalwani  

📍 *Golisano College of Computing, RIT*

---

> 📝 *A full technical report, attack models, plots, and implementation details are included in the `/report` folder.*
