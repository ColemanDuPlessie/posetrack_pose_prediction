# -*- coding: utf-8 -*-
"""
This needs documentation at some point
"""

import math
import torch
import matplotlib as plt
from models.LSTM import LSTMBenchmark
from models.Transformer_encoder import TransformerEncoder
from models.Informer import Informer
from KitchenModelComparisonTest import ModelWrapper

randomize_batch_series = True
batches_to_use = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

print("Pytorch running on " + str(device))

torch.set_default_tensor_type('torch.cuda.FloatTensor' if torch.cuda.is_available() else 'torch.FloatTensor')

input_size = 1
hidden_size = 768
num_layers = 5
min_pred_len = 128
max_pred_len = 1024

models = {LSTMBenchmark(hidden_size, input_size, input_size, num_layers): (min_pred_len,),
          TransformerEncoder(hidden_size, 8, input_size, num_layers, max_pred_len * 2): (min_pred_len,),
          Informer(input_size, input_size, input_size, 1, d_model=hidden_size, n_heads=8,
                   e_layers=math.ceil(num_layers / 2), d_layers=math.floor(num_layers / 2),
                   d_ff=hidden_size, activation="relu", positional_embedding_max_len=max_pred_len * 2): (
          min_pred_len, 1)}

pretrained_model_to_view = "TrainedSineWaveTransformer.pt"

timesteps = 2048
min_period = 128
dimension = 1
batch_size=1024
test_data = torch.tensor([[0.5+math.sin(index*2*math.pi/min_period/n)/2 for n in range(1, dimension+1)] for index in range(0, batch_size)])
ground_truth = torch.tensor([[0.5+math.sin(index*2*math.pi/min_period/n)/2 for n in range(1, dimension+1)] for index in range(0, batch_size)])
state_dict = torch.load(pretrained_model_to_view, map_location=device)
for model_type in models:
    try:
        model_type.load_state_dict(state_dict)
        model = model_type
        inputs = models[model]
        break
    except RuntimeError:
        continue

assert model in models
wrapped_model = ModelWrapper(model, "If you're seeing this, it's a bug", None, torch.nn.MSELoss(), {}, False)
print("Model %s loaded successfully!" % pretrained_model_to_view)

print("Beginning rendering!")
model(test_data)
plt.plot(ground_truth[0].squeeze(), label="ground truth")
plt.plot(test_data[0].squeeze(), label="prediction")
plt.legend()
plt.show()
