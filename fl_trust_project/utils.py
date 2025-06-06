
import torchvision.transforms as T
from torchvision.datasets import VisionDataset
from torch.utils.data import DataLoader, Subset
import numpy as np

def load_partition(dataset: VisionDataset, cid: int, num_clients: int, iid: bool = False):
    """
    Split dataset into partitions.
    If iid=False (default), each client gets data from two classes.
    """
    labels = np.array(dataset.targets)
    if iid:
        # simple round‑robin allocation
        idx = np.arange(len(dataset))[cid::num_clients]
    else:
        # non‑IID: pick 2 classes per client
        classes = np.unique(labels)
        classes_per_client = 2
        chosen = classes[(cid * classes_per_client) % len(classes):(cid * classes_per_client) % len(classes) + classes_per_client]
        idx = np.where(np.isin(labels, chosen))[0]
    return Subset(dataset, idx)

def default_transforms():
    return T.Compose([
        T.Resize((28, 28)),
        T.ToTensor(),
    ])
