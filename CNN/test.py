import pickle
import torch
import numpy as np

with open('/app/PoC/CNN/embedding_model', 'rb') as f:
        embedding_model = pickle.load(f)

weights = torch.FloatTensor(embedding_model.wv.vectors)

with open('/app/PoC/CNN/dataset', 'rb') as f:
        dataset = pickle.load(f)

X = list(dataset.keys())
Y = list(dataset.values())

print(X[:2])