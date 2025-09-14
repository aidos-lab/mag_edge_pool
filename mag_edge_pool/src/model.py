import os
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.utils import to_networkx
import numpy as np
import os
import random
import time
from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, TopKPooling, GeneralConv
from torch_geometric.nn import MLP
from mag_edge_pool.src.make_splits import make_splits
import json
from torch.nn import PReLU
from torch_geometric.nn import global_add_pool


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False