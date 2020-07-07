import syft as sy

import gensim
import torch
import torch.nn as nn
from model import BiLSTM
import os
import numpy as np

sy.make_hook(globals())

EMBEDDING_DIM = 300
NUM_CLASSES = 39

embeddings = gensim.models.KeyedVectors.load_word2vec_format('../embeddings/german.model', binary=True)
word_embeddings = []

vector = np.zeros(EMBEDDING_DIM) # PADDING_TOKEN
word_embeddings.append(vector)

vector = np.random.uniform(-0.25, 0.25, EMBEDDING_DIM) # UNKNOWN_TOKEN
word_embeddings.append(vector)

# loop through each word in embeddings
for word in embeddings.vocab:
    vector = embeddings.wv[word]
    word_embeddings.append(vector)

model = BiLSTM(word_embeddings=torch.FloatTensor(word_embeddings), num_classes=NUM_CLASSES)

def set_model_params(module, params_list, start_param_idx=0):
    """ Set params list into model recursively
    """
    param_idx = start_param_idx

    for name, param in module._parameters.items():
        module._parameters[name] = params_list[param_idx]
        param_idx += 1

    for name, child in module._modules.items():
        if child is not None:
            param_idx = set_model_params(child, params_list, param_idx)

    return param_idx

@sy.func2plan()
def training_plan(X, y, batch_size, lr, model_params):
    #set_model_params(model, model_params)

    prediction = model(X)
    loss = nn.CrossEntropyLoss()(prediction, y)

    loss.backward()

    updated_params = [torch.optim.SGD(param, lr=learning_rate) for param in model_params]

    return (
        loss,
        *updated_params
    )

model_params = list(model.parameters())
X = torch.ones((1, 6), dtype=torch.long)
y = torch.zeros(6, dtype=torch.long)
lr = torch.Tensor([0.01])
batch_size = torch.Tensor([1])

print(model(X))

_ = training_plan.build(X, y, batch_size, lr, model_params, trace_autograd=True)

print(training_plan.code)
#print(training_plan.torchscript.code)
