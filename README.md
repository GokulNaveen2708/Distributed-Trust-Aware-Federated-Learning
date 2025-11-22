# 🔐 Trust & Reputation for Secure Federated Learning

Federated Learning (FL) trains models without centralizing data—perfect for privacy‑sensitive domains like healthcare or finance. However, malicious or low-quality clients can poison training and degrade global performance. This project augments FedAvg with a **trust and reputation layer** that scores clients, filters unreliable participants, and improves robustness against adversarial behavior.

---

## 🧠 Why This Matters

- **Threats**: Poisoning, gradient manipulation, and inconsistent clients undermine FL reliability.
- **Goal**: Let trustworthy clients shape the global model while down-weighting or excluding risky participants.
- **Outcome**: Faster, more stable convergence even when adversaries are present.

---

## ⚙️ What the System Does

- ✅ **Trust & Reputation Scoring**: Evaluates each client via distance from the global model and recent behavior.
- 🧹 **Dynamic Client Filtering**: Automatically removes clients whose trust falls below a threshold (β).
- 🧪 **Attack Simulation**: Includes label-flipping and gradient perturbation adversaries to test defenses.
- 📉 **Trust Decay & Recovery**: Smooth updates (γ) prevent overreacting to noise while allowing redemption.

---

## 🧪 How We Experimented

- 🔍 **Framework**: [Flower FL](https://flower.dev/) for client–server orchestration.
- 🧬 **Dataset**: [PathMNIST](https://medmnist.com/) — 100K+ pathology images across 9 classes.
- 🤖 **Model**: CNN with ReLU, max pooling, and fully connected layers.
- 💥 **Attacks**: Label flipping and gradient perturbations to stress the trust pipeline.

**Key Hyperparameters**
- Learning rate: `0.001` with `0.95` decay
- Batch size: `32`
- Trust smoothing: `γ = 0.8`
- Trust threshold: `β = 0.6`

---

## 📈 What We Observed

| Scenario              | Accuracy (%) | Observation                                  |
|-----------------------|--------------|----------------------------------------------|
| Baseline (FedAvg)     | ~80%         | Poisoned clients degraded model performance. |
| Trust-Enhanced FedAvg | ~87%         | Filtering reduced attacker influence.        |

- 📉 **Global loss**: Fell from ~1.0 to ~0.02 within three rounds.
- 🧑‍⚕️ **Benign clients**: Maintained steady trust; adversaries decayed quickly.
- 🛰️ **Overhead**: Minimal additional communication and computation.

---

## 📦 System Components

- **Trust Manager**: Maintains smoothed trust scores and enforces filtering.
- **Client Reputation Module**: Measures deviation from the global model after each round.
- **Enhanced Protocol**: Exchanges model weights alongside trust metrics.
- **Logging & Analytics**: Tracks participation, trust trends, and attack isolation.

---

## 🔬 How We Measure Success

- 🔹 Client-wise accuracy and loss
- 🔹 Global convergence behavior
- 🔹 Trust score trajectories over time
- 🔹 Comparisons of benign vs. adversarial scenarios

---

## 🚧 Where We’re Going Next

- 🔐 Add cryptographic secure aggregation for stronger privacy.
- 🌐 Scale to 1000+ clients for IoT-scale testing.
- 🤖 Defend against adaptive adversaries and sybil attacks.
- 📡 Optimize for resource-constrained edge deployments.
- 🪞 Add explainability for trust decisions.

---

## 👥 Team

- Gokula Chapala
- Yasiru Karunawansa
- Dhairya Lalwani

📍 *Golisano College of Computing, RIT*

---

> 📝 *A full technical report, attack models, plots, and implementation details live in the `/report` folder.*
