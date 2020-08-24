import torch
from model import Net

model = Net()


def visualize_feature_map(path):
    model = Net()
    model.load_state_dict(torch.load(path))