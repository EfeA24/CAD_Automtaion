import torch

def perform_aggregate(X, Y, type):
    """
    X: Shape (B,N,D)
    Y: Shape (B,N,D)
    """
    if type == "sum":
        return X+Y
    elif type == "mean":
        return 0.5*(X+Y)
    elif type == "max":
        return torch.maximum(X, Y)
