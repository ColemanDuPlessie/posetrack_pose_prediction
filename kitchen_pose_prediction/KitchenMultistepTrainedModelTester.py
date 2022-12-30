# -*- coding: utf-8 -*-
"""
This needs documentation at some point
"""

import torch
import matplotlib.pyplot as plt
from BatchManager import BatchManager
from models.LSTM import LSTMMultistep
from models.Transformer_encoder import TransformerEncoderMultistep
from models.benchmarks import SimpleRepeater
from KitchenModelComparisonMultistep import ModelWrapper, get_loss_filename

batches_to_use = 20 # All four of these numbers must be strictly > 0
min_steps = 1
max_steps = 10
steps_step = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Pytorch running on " + str(device))

torch.set_default_tensor_type('torch.cuda.FloatTensor' if torch.cuda.is_available() else 'torch.FloatTensor')

input_size = 153
hidden_size = 1024
num_layers = 6
min_pred_len = 100
max_pred_len = 1024

models = {LSTMMultistep(hidden_size, input_size, input_size, num_layers) : (min_pred_len,),
          TransformerEncoderMultistep(hidden_size, 8, input_size, num_layers, max_pred_len*2) : (min_pred_len,)}

pretrained_model_to_view = "EddiesTrained30fpsKitchenTransformer.pt"
dataset_to_view = "shuffled_velo_1024"

def load_data(start_batch, end_batch):
    return torch.utils.data.DataLoader(BatchManager("preprocessed_data/" + dataset_to_view, start_batch, end_batch))
         
data = load_data(603-batches_to_use, 603)
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
model = SimpleRepeater()
wrapped_model = ModelWrapper(model, model.__class__.__name__, None, torch.nn.MSELoss(), {})
print("Model %s loaded successfully!" % pretrained_model_to_view)

with torch.no_grad():
    prediction = [wrapped_model.test(data, min_pred_len, steps) for steps in range(min_steps, max_steps, steps_step)]

losses = wrapped_model.test_losses + wrapped_model.multistep_test_losses

plt.scatter(range(min_steps, max_steps, steps_step), losses)
plt.show()

with open(get_loss_filename(), "w") as writing:
    writing.write(wrapped_model.name + " " + str(losses))