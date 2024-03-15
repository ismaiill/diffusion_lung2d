import os
import random

import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision import models


def evaluate_resnet_score(path: str) -> np.ndarray:
    """Evaluate the ResNet18 score."""
    # Load the data tensor and convert it to 3 channels
    data = torch.load(path)
    data = data.unsqueeze(0).unsqueeze(0)
    data = data.repeat(1, 3, 1, 1)

    # Obtain the numpy array of the score vector
    score = resnet(data)
    score = score.detach().numpy()
    return score


if __name__ == '__main__':
    # Load the pretrained ResNet18 model
    resnet = models.resnet18(pretrained=True)
    resnet.eval()

    directory = '/home/peter/data/2dmap'
    files = os.listdir(directory)
    paths = [os.path.join(directory, file) for file in files]
    paths = random.sample(paths, 100)

    score0 = evaluate_resnet_score(paths[0])
    for i, path in enumerate(paths):
        score = evaluate_resnet_score(path)
        print(path, np.linalg.norm(score - score0))
