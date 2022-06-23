# -*- coding: utf-8 -*-
"""
This needs documentation at some point
"""

import gc # This is an ugly hack
import math
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from ParseKitchenC3D import load_and_prepare_mocaps, train_test_split
from models.benchmarks import SimpleRepeater
from models.Transformer_encoder import TransformerEncoder
from models.Informer import Informer
# TODO from models.Pyraformer.Pyraformer_SS import Model as Pyraformer
# TODO from models.Pyraformer.Pyraformer_SS import pyraformer_params

min_seq_length = 100
predict_length = 1

input_size = 153

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) # TODO this may not count certain types of nested layers

def load_data(filenames, batch_size = 1024, train_qty = 0.67):
    """
    Returns a tuple of the form (train_data, test_data)
    """
    return train_test_split(Variable(load_and_prepare_mocaps(filenames, batch_size)), train_qty)

class ModelWrapper:
    """
    This class is just a simple wrapper for various models.
    """
    
    def __init__(self, model, name, optimizer_class, loss_function, optimizer_kwargs={}, store_losses=True):
        self.model = model
        self.name = name
        self._optimizer = None if optimizer_class is None else optimizer_class(model.parameters(), **optimizer_kwargs)
        self._loss_function = loss_function
        self._store_losses = store_losses
        if store_losses:
            self.train_losses = []
            self.test_losses = []
    
    def train(self, train_set, min_seq_len, predict_len=1, backprop=True):
        self.model.train()
        try:
            output = self.model(train_set[:, :-predict_len, :], min_seq_len)
        except TypeError:
            print("This feature actually worked!")
            output = self.model(train_set[:, :-predict_len, :])
            output = output[:, min_seq_len-1:, :]
        if backprop:
            if self._optimizer is not None:
                self._optimizer.zero_grad()
            loss = self._loss_function(output, train_set[:, min_seq_len+predict_len:])
            loss.backward()
            if self._optimizer is not None:
                self._optimizer.step()
        if self._store_losses:
            self.train_losses.append(loss.item())
        return output
    
    def test(self, test_set, min_seq_len, predict_len=1):
        self.model.eval()
        with torch.no_grad():
            try:
                output = self.model(test_set[:, :-predict_len, :], min_seq_len)
            except TypeError:
                print("This feature actually worked!")
                output = self.model(test_set[:, :-predict_len, :])
                output = output[:, min_seq_len-1:, :]
            loss = self._loss_function(output, test_set[:, min_seq_len+predict_len:])
        if self._store_losses:
            self.test_losses.append(loss.item())
        return output
    
    def to(self, device):
        self.model.to(device)
    
    def get_params_string(self):
        return "%s has %d parameters." % (self.name, count_parameters(self.model))
    
    def get_losses_string(self):
        return "%s train loss: %1.6f %s test loss: %1.6f" % (self.name, self.train_losses[-1], self.name, self.test_losses[-1])

class MultiModelHandler:
    """
    This class is a collection of ModelWrappers (found above).
    """
    
    def __init__(self, *model_wrappers):
        self.networks = model_wrappers
    
    def train(self, train_set, min_seq_len, predict_len=1, backprop=True):
        output = []
        for network in self.networks:
            output.append(network.train(train_set, min_seq_len, predict_len, backprop))
        return output
    
    def test(self, test_set, min_seq_len, predict_len=1):
        output = []
        for network in self.networks:
            output.append(network.test(test_set, min_seq_len, predict_len))
        return output
    
    def to(self, device):
        for network in self.networks: network.to(device)
    
    def get_params_string(self):
        ans = ""
        for network_idx, network in enumerate(self.networks):
            ans += "Model %d: %s " % (network_idx, network.get_params_string())
        return ans[:-1]
    
    def get_losses_string(self):
        ans = ""
        for network in self.networks:
            ans += "%s " % network.get_losses_string()
        return ans[:-1]
    
    def plot_losses_over_time(self):
        for network in self.networks:
            plt.plot(network.train_losses, label = "Train (%s)" % network.name)
            plt.plot(network.test_losses, label = "Test (%s)" % network.name)
        plt.legend()
        plt.yscale('log')
        plt.show()
    
    def save_models(self, filenames):
        for idx, network in enumerate(self.networks):
            if filenames[idx] is not None:
                torch.save(network.model.state_dict(), filenames[idx])

if __name__ == "__main__":
    num_epochs = 50 # TODO
    learning_rate = 0.01
    batch_size = 1024
    positional_embedding_max_len = batch_size * 2
    
    hidden_size = 1024 # TODO
    num_layers = 4 # TODO
    
    num_classes = 153
    
    networks = MultiModelHandler(ModelWrapper(TransformerEncoder(hidden_size, 8, input_size, num_layers, positional_embedding_max_len), "Transformer (encoder only)", torch.optim.Adam, torch.nn.MSELoss(), {"lr" : learning_rate}),
                ModelWrapper(Informer(input_size, input_size, input_size, 1, d_model = hidden_size, n_heads = 8,
                                         e_layers = math.ceil(num_layers/2), d_layers = math.floor(num_layers/2),
                                         d_ff = hidden_size, activation = "relu", positional_embedding_max_len = positional_embedding_max_len), "Informer", torch.optim.Adam, torch.nn.MSELoss(), {"lr" : learning_rate}),
                ModelWrapper(SimpleRepeater(input_size), "Benchmark", None, torch.nn.MSELoss()),)
    
# =============================================================================
#     network1 = TransformerEncoder(hidden_size, 8, input_size, num_layers, positional_embedding_max_len)
#     network2 = SimpleRepeater(input_size)
#     network3 = Pyraformer(pyraformer_params(num_classes, hidden_size, n_head=8, n_layer=num_layers, batch_size=batch_size, inner_hid=hidden_size, d_k=hidden_size/4, d_v=hidden_size/4, device='cpu')) # !!! Change device to cuda if you're running a model of any significant size.
#     network2 = Informer(input_size, input_size, input_size, 1,
#                         d_model = hidden_size, n_heads = 8,
#                         e_layers = math.ceil(num_layers/2),
#                         d_layers = math.floor(num_layers/2),
#                         d_ff = hidden_size, activation = "relu", device = torch.device("cpu"), 
#                         positional_embedding_max_len = positional_embedding_max_len)
#     
#     criterion = torch.nn.MSELoss()    # mean-squared error for regression
#     optimizer1 = torch.optim.Adam(network1.parameters(), lr=learning_rate)
#     # optimizer2 = torch.optim.Adam(network2.parameters(), lr=learning_rate)
#     optimizer3 = torch.optim.Adam(network3.parameters(), lr=learning_rate)
# =============================================================================
    
    print(networks.get_params_string())
    
    train_data, test_data = load_data(("mocap/brownies1.c3d", "mocap/brownies3.c3d", "mocap/brownies4.c3d",
                                       "mocap/eggs1.c3d", "mocap/eggs3.c3d", "mocap/eggs4.c3d", 
                                       "mocap/pizza1.c3d", "mocap/pizza2.c3d", "mocap/pizza3.c3d", "mocap/pizza4.c3d", 
                                       "mocap/salad1.c3d", "mocap/salad2.c3d", "mocap/salad3.c3d", "mocap/salad4.c3d", 
                                       "mocap/sandwich1.c3d", "mocap/sandwich2.c3d", "mocap/sandwich3.c3d", "mocap/sandwich4.c3d", ), batch_size)
    train_data.requires_grad = True
    test_data.requires_grad = True
    
    print("Beginning training...")
    
    # Train the model
    for epoch in range(num_epochs):
        networks.train(train_data, min_seq_length)
        networks.test(test_data, min_seq_length)
        print(networks.get_losses_string())
    
    networks.plot_losses_over_time()
    networks.save_models(("TrainedKitchenTransformer.pt", "TrainedKitchenInformer.pt", None))