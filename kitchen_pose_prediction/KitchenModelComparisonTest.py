# -*- coding: utf-8 -*-
"""
This needs documentation at some point
"""

import math
import matplotlib.pyplot as plt
import torch
from BatchManager import BatchManager
from torch.autograd import Variable
from models.benchmarks import SimpleRepeater
from models.LSTM import LSTMBenchmark
from models.Transformer_encoder import TransformerEncoder
from models.Informer import Informer
# TODO from models.Pyraformer.Pyraformer_SS import Model as Pyraformer
# TODO from models.Pyraformer.Pyraformer_SS import pyraformer_params

min_seq_length = 100
predict_length = 1

input_size = 153

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.set_default_tensor_type('torch.cuda.FloatTensor' if torch.cuda.is_available() else 'torch.FloatTensor')

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) # TODO this may not count certain types of nested layers

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
    
    def train(self, train_loader, min_seq_len, predict_len=1, backprop=True):
        self.model.train()
        outputs = []
        losses = []
        for train_set in train_loader:
            train_set = Variable(train_set.to(device)) # TODO globals are bad
            train_set.requires_grad = True
            try:
                output = self.model(train_set[:, :-predict_len, :], min_seq_len)
            except TypeError:
                print("This feature actually worked!")
                output = self.model(train_set[:, :-predict_len, :])
                output = output[:, min_seq_len-1:, :]
            outputs.append(output)
            if backprop:
                if self._optimizer is not None:
                    self._optimizer.zero_grad()
                loss = self._loss_function(output, train_set[:, min_seq_len+predict_len:])
                loss.backward()
                if self._optimizer is not None:
                    self._optimizer.step()
                losses.append(loss.item())
        output = torch.cat(outputs, dim=0)
        if self._store_losses:
            self.train_losses.append(sum(losses)/len(losses))
        return output
    
    def test(self, test_loader, min_seq_len, predict_len=1):
        self.model.eval()
        outputs = []
        losses = []
        with torch.no_grad():
            for test_set in test_loader:
                test_set = Variable(test_set.to(device)) # TODO globals are bad
                test_set.requires_grad = True
                try:
                    output = self.model(test_set[:, :-predict_len, :], min_seq_len)
                except TypeError:
                    print("This feature actually worked!")
                    output = self.model(test_set[:, :-predict_len, :])
                    output = output[:, min_seq_len-1:, :]
                loss = self._loss_function(output, test_set[:, min_seq_len+predict_len:])
                outputs.append(output)
                losses.append(loss.item())
        if self._store_losses:
            self.test_losses.append(sum(losses)/len(losses))
        return torch.stack(outputs)
    
    def to(self, device):
        self.model.to(device)
    
    def get_params_string(self):
        return "%s has %d parameters." % (self.name, count_parameters(self.model))
    
    def get_losses_string(self):
        return "%s train loss: %1.6f %s test loss: %1.6f" % (self.name, self.train_losses[-1], self.name, self.test_losses[-1])
    
    def get_simple_losses_str(self):
        """
        This function should be used for saving parseable data to a .txt file
        or similar. It should be called once after training. The function above
        this one should be used for a more human-readable output and called
        every epoch.
        """
        return ("%s train " + "%f " * len(self.train_losses) + "test " + "%f " * len(self.test_losses))[:-1] % (self.name, *self.train_losses, *self.test_losses)

class MultiModelHandler:
    """
    This class is a collection of ModelWrappers (found above).
    """
    
    def __init__(self, device, *model_wrappers):
        self.networks = model_wrappers
        for network in self.networks:
            network.to(device)
    
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
    
    def get_simple_losses_str(self):
        """
        This function should be used for saving parseable data to a .txt file
        or similar. It should be called once after training. The function above
        this one should be used for a more human-readable output and called
        every epoch.
        """
        ans = ""
        for network in self.networks:
            ans += "%s\n" % network.get_simple_losses_str()
        return ans[:-1]
    
    def plot_losses_over_time(self):
        for network in self.networks:
            plt.plot(network.train_losses, label = "Train (%s)" % network.name)
            plt.plot(network.test_losses, label = "Test (%s)" % network.name)
        plt.legend()
        plt.yscale('log')
        plt.show()
    
    def log_losses(self, log_file):
        with open(log_file, "w") as writing:
            writing.write(self.get_simple_losses_str())
            
    def save_models(self, filenames):
        for idx, network in enumerate(self.networks):
            if filenames[idx] is not None:
                torch.save(network.model.state_dict(), filenames[idx])

if __name__ == "__main__":
    print("Pytorch running on %s." % str(device))
    num_epochs = 100 # TODO
    learning_rate = 0.00001
    batch_size = 1024
    batches_at_once = 1
    positional_embedding_max_len = batch_size * 2
    
    hidden_size = 1024 # TODO
    num_layers = 6 # TODO
    
    num_classes = 153
    
    networks = MultiModelHandler(device, ModelWrapper(TransformerEncoder(hidden_size, 8, input_size, num_layers, positional_embedding_max_len), "Transformer (encoder only)", torch.optim.Adam, torch.nn.MSELoss(), {"lr" : learning_rate}),
                ModelWrapper(Informer(input_size, input_size, input_size, 1, d_model = hidden_size, n_heads = 8,
                                         e_layers = math.ceil(num_layers/2), d_layers = math.floor(num_layers/2),
                                         d_ff = hidden_size, activation = "relu", positional_embedding_max_len = positional_embedding_max_len), "Informer", torch.optim.Adam, torch.nn.MSELoss(), {"lr" : learning_rate}),
                ModelWrapper(LSTMBenchmark(hidden_size, input_size, num_classes, num_layers), "LSTM", torch.optim.Adam, torch.nn.MSELoss(), {"lr" : learning_rate}),
                ModelWrapper(SimpleRepeater(input_size), "Benchmark", None, torch.nn.MSELoss()))
    
    print(networks.get_params_string())
    
    train_data = torch.utils.data.DataLoader(BatchManager("preprocessed_data/velo_1024", 0, 438), batches_at_once, True, generator=torch.Generator(device=device))
    test_data = torch.utils.data.DataLoader(BatchManager("preprocessed_data/velo_1024", 438, 655), batches_at_once, True, generator=torch.Generator(device=device))
    
    print("Beginning training...")
    
    # Train the model
    for epoch in range(num_epochs):
        networks.train(train_data, min_seq_length)
        networks.test(test_data, min_seq_length)
        print(networks.get_losses_string())
    
    networks.plot_losses_over_time()
    networks.log_losses("losses.txt")
    networks.save_models(("TrainedKitchenTransformer.pt", "TrainedKitchenInformer.pt", "TrainedKitchenLSTM.pt", None))