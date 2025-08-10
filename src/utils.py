# src/utils.py
import joblib
import torch
import os
import numpy as np
import scipy.sparse as sp

def save_artifact(path, obj):
    if isinstance(obj, torch.nn.Module):
        torch.save({'model_state_dict': obj.state_dict()}, path)
    else:
        joblib.dump(obj, path)

def load_artifact(path):
    if path.endswith('.pt') or path.endswith('.pth'):
        return torch.load(path)
    else:
        return joblib.load(path)
