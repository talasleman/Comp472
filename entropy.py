# entropy calculation
import numpy as np

def entropy(data):
    labels, counts = np.unique(data, return_counts=True)
    probabilities = counts / np.sum(counts)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy //