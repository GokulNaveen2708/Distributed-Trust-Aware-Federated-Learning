
import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from model import SimpleCNN
from utils import load_partition, default_transforms
from sklearn.metrics import accuracy_score   # NEW
import argparse
import random

def load_data(cid: int, num_clients: int, poisoned_ids):
    # Placeholder: replace with PathMNIST download
    import torchvision.datasets as dsets
    trainset = dsets.MNIST(root="./data", train=True, download=True, transform=default_transforms())
    testset = dsets.MNIST(root="./data", train=False, download=True, transform=default_transforms())

    train_subset = load_partition(trainset, cid, num_clients)
    test_subset = load_partition(testset, cid, num_clients)

    # Poison if necessary (label flipping)
    if cid in poisoned_ids:
        for _, target in train_subset:
            target = 9 - target  # simple flip: digit d -> 9-d
    return train_subset, test_subset

class FLClient(fl.client.NumPyClient):
    def __init__(self, cid: int, poisoned_ids):
        self.cid = cid
        self.model = SimpleCNN(num_classes=10)
        self.poisoned_ids = poisoned_ids

        train_ds, test_ds = load_data(cid, num_clients=10, poisoned_ids=poisoned_ids)
        self.train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        self.test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
        self.criterion = nn.CrossEntropyLoss()

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, params):
        state_dict = self.model.state_dict()
        for k, v in zip(state_dict.keys(), params):
            state_dict[k] = torch.tensor(v)
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, params, config):
        self.set_parameters(params)
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        epoch = config.get("epoch", 1)
        for _ in range(epoch):
            for data, target in self.train_loader:
                optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                optimizer.step()

        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, params, config):
        """Return (loss, num_examples, metrics_dict)."""
        self.set_parameters(params)
        self.model.eval()

        running_loss, preds, targets = 0.0, [], []
        with torch.no_grad():
            for data, target in self.test_loader:
                output = self.model(data)
                loss = self.criterion(output, target)
                running_loss += loss.item() * data.size(0)

                preds.extend(output.argmax(1).cpu().tolist())
                targets.extend(target.cpu().tolist())

        num_examples = len(targets)
        avg_loss = running_loss / num_examples
        accuracy = accuracy_score(targets, preds)

        # Flower expects: loss (float), num_examples (int), metrics (dict)
        return float(avg_loss), num_examples, {"accuracy": float(accuracy)}


def start_client(cid: int, poisoned_ids):
    fl.client.start_numpy_client(server_address="localhost:8080", client=FLClient(cid, poisoned_ids))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cid", type=int, required=True)
    args = parser.parse_args()
    poisoned = [1, 3]  # example
    start_client(args.cid, poisoned)
