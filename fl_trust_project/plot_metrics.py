import pickle, matplotlib.pyplot as plt, pathlib

hist = pickle.load(open("logs/history.pkl", "rb"))

# ----- LOSS -----
rounds = [r for r, _ in hist.losses_distributed]
losses = [v for _, v in hist.losses_distributed]

plt.figure(figsize=(6,4))
plt.plot(rounds, losses, marker="o")
plt.title("Global Loss vs Rounds")
plt.xlabel("Round"); plt.ylabel("Loss"); plt.grid(True); plt.tight_layout()
pathlib.Path("plots").mkdir(exist_ok=True)
plt.savefig("plots/global_loss.png", dpi=300)
plt.close()

# ----- ACCURACY -----
acc_tuples = hist.metrics_distributed["accuracy"]
rounds = [r for r, _ in acc_tuples]
accs   = [v for _, v in acc_tuples]

plt.figure(figsize=(6,4))
plt.plot(rounds, accs, marker="o")
plt.title("Global Accuracy vs Rounds")
plt.xlabel("Round"); plt.ylabel("Accuracy"); plt.ylim(0,1); plt.grid(True)
plt.tight_layout()
plt.savefig("plots/global_accuracy.png", dpi=300)
plt.close()

print("Saved plots to plots/global_loss.png and plots/global_accuracy.png")


hist = pickle.load(open("logs/client_history.pkl", "rb"))
pathlib.Path("plots").mkdir(exist_ok=True)

# -------- Loss per client ------------
plt.figure(figsize=(8,5))
for cid, rec in hist.items():
    plt.plot(rec["round"], rec["loss"], label=f"Client {cid}")
plt.title("Loss per Client vs Rounds")
plt.xlabel("Round"); plt.ylabel("Loss")
plt.legend(fontsize=8, ncol=2)
plt.grid(True); plt.tight_layout()
plt.savefig("plots/loss_per_client.png", dpi=300)

# -------- Accuracy per client --------
plt.figure(figsize=(8,5))
for cid, rec in hist.items():
    plt.plot(rec["round"], rec["accuracy"], label=f"Client {cid}")
plt.title("Accuracy per Client vs Rounds")
plt.xlabel("Round"); plt.ylabel("Accuracy")
plt.ylim(0,1)
plt.legend(fontsize=8, ncol=2)
plt.grid(True); plt.tight_layout()
plt.savefig("plots/accuracy_per_client.png", dpi=300)

print("Saved plots to plots/loss_per_client.png and plots/accuracy_per_client.png")
