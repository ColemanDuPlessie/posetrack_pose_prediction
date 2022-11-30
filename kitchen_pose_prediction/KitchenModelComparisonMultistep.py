# -*- coding: utf-8 -*-
"""
This needs documentation at some point
"""

import os
import matplotlib.pyplot as plt
import torch
from BatchManager import BatchManager
from torch.autograd import Variable
from models.benchmarks import SimpleRepeater
from models.LSTM import LSTMMultistep
from models.Transformer_encoder import TransformerEncoderMultistep

min_seq_length = 100
predict_length = 3
predict_freq = 10

input_size = 153

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.set_default_tensor_type('torch.cuda.FloatTensor' if torch.cuda.is_available() else 'torch.FloatTensor')

def get_loss_filename():
    i = 0
    while os.path.isfile("losses" + str(i) + ".txt"):
        i += 1
    return "losses" + str(i) + ".txt"

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) # TODO this may not count certain types of nested layers

class ModelWrapper:
    """
    This class is just a simple wrapper for various models.
    """
    
    def __init__(self, model, name, optimizer_class, loss_function, optimizer_kwargs={}, store_losses=True):
        self.model = model
        if device == "cuda" and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        self.name = name
        self._optimizer = None if optimizer_class is None else optimizer_class(model.parameters(), **optimizer_kwargs)
        self._loss_function = loss_function
        self._store_losses = store_losses
        if store_losses:
            self.train_losses = []
            self.test_losses = []
            self.multistep_test_losses = []
    
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
        run = self.model.forward if predict_len == 1 else lambda data_in, msl: self.model.forward_multistep(data_in, msl, predict_len)
        outputs = []
        losses = []
        with torch.no_grad():
            for test_set in test_loader:
                test_set = Variable(test_set.to(device)) # TODO globals are bad
                test_set.requires_grad = True
                try:
                    output = run(test_set[:, :-predict_len, :], min_seq_len)
                except TypeError:
                    print("This feature actually worked!")
                    output = run(test_set[:, :-predict_len, :])
                    output = output[:, min_seq_len-1:, :]
                loss = self._loss_function(output, test_set[:, min_seq_len+predict_len:])
                outputs.append(output)
                losses.append(loss.item())
        if self._store_losses:
            if predict_len == 1:
                self.test_losses.append(sum(losses)/len(losses))
            else:
                self.multistep_test_losses.append(sum(losses)/len(losses))
        return torch.stack(outputs)
    
    def to(self, device):
        self.model.to(device)
    
    def get_params_string(self):
        return "%s has %d parameters." % (self.name, count_parameters(self.model))
    
    def get_losses_string(self):
        return "%s train loss: %1.6f %s test loss: %1.6f, %s most recent multistep test loss: %1.6f" % (self.name, self.train_losses[-1], self.name, self.test_losses[-1], self.name, self.multistep_test_losses[-1])
    
    def get_simple_losses_str(self):
        """
        This function should be used for saving parseable data to a .txt file
        or similar. It should be called once after training. The function above
        this one should be used for a more human-readable output and called
        every epoch.
        """
        return ("%s train " + "%f " * len(self.train_losses) + "test " + "%f " * len(self.test_losses) + "multistep " + "%f " * len(self.multistep_test_losses))[:-1] % (self.name, *self.train_losses, *self.test_losses, *self.multistep_test_losses)

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
    
    def plot_losses_over_time(self, multisteps_per_step = None):
        for network in self.networks:
            if multisteps_per_step == None:
                multisteps_per_step = len(network.test_losses)//len(network.multistep_test_losses)
            plt.plot(network.train_losses, label = "Train (%s)" % network.name)
            plt.plot(network.test_losses, label = "Test (%s)" % network.name)
            plt.plot(range(0, len(network.test_losses), multisteps_per_step), network.multistep_test_losses, label = "Multistep Test (%s)" % network.name)
        plt.legend()
        plt.yscale('log')
        plt.show()
    
    def log_losses(self, log_file):
        with open(log_file, "w") as writing:
            writing.write(self.get_simple_losses_str())
            
    def save_models(self, filenames):
        for idx, network in enumerate(self.networks):
            if filenames[idx] is not None:
                torch.save(network.model.module.state_dict() if isinstance(network.model, torch.nn.DataParallel) else network.model.state_dict(), filenames[idx])

if __name__ == "__main__":
    print("Pytorch running on %s." % str(device))
    num_epochs = 100 # TODO
    learning_rate = 0.00005
    batch_size = 1024
    batches_at_once = max((1, torch.cuda.device_count() if device == "cuda" else 0))
    positional_embedding_max_len = batch_size * 2
    
    hidden_size = 1024 # TODO
    num_layers = 6 # TODO
    
    num_classes = 153
    
    networks = MultiModelHandler(device,
                ModelWrapper(TransformerEncoderMultistep(hidden_size, 8, input_size, num_layers, positional_embedding_max_len), "Transformer (encoder only)", torch.optim.Adam, torch.nn.MSELoss(), {"lr" : learning_rate}),
                ModelWrapper(LSTMMultistep(hidden_size, input_size, num_classes, num_layers), "LSTM", torch.optim.Adam, torch.nn.MSELoss(), {"lr" : learning_rate*50}),
                ModelWrapper(SimpleRepeater(input_size), "Benchmark", None, torch.nn.MSELoss()))
    schedulers = []
    for model in networks.networks:
        if model._optimizer is None: continue
        schedulers.append(torch.optim.lr_scheduler.ReduceLROnPlateau(model._optimizer, mode='min', factor=0.9, patience=6, cooldown=6, verbose=True, threshold=0.001, threshold_mode='rel'))
    print(networks.get_params_string())
    
    train_data = torch.utils.data.DataLoader(BatchManager("preprocessed_data/30fps_velo_1024", 0, 402), batches_at_once, True, generator=torch.Generator(device=device), num_workers=os.cpu_count()-1)
    test_data = torch.utils.data.DataLoader(BatchManager("preprocessed_data/30fps_velo_1024", 402, 603), batches_at_once, True, generator=torch.Generator(device=device), num_workers=os.cpu_count()-1)
    multistep_test_data = torch.utils.data.DataLoader(BatchManager("preprocessed_data/30fps_velo_1024", 583, 603), batches_at_once, True, generator=torch.Generator(device=device), num_workers=os.cpu_count()-1)

    print("Beginning training...")
    
    # Train the model
    for epoch in range(num_epochs):
        networks.train(train_data, min_seq_length)
        networks.test(test_data, min_seq_length)
        sched_idx = 0
        for model in networks.networks:
            if model._optimizer is None: continue
            schedulers[sched_idx].step(model.test_losses[-1])
            sched_idx += 1
        if epoch % predict_freq == 0:
            networks.test(multistep_test_data, min_seq_length, predict_length)
        print(networks.get_losses_string())
    
    networks.plot_losses_over_time(predict_freq)
    networks.log_losses(get_loss_filename())
    networks.save_models(("TrainedKitchenTransformer.pt", "TrainedKitchenInformer.pt", "TrainedKitchenLSTM.pt", None))