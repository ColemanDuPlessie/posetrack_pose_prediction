# -*- coding: utf-8 -*-
"""
This needs documentation at some point
"""

import gc # This is an ugly hack
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from ParseKitchenC3D import load_and_prepare_mocaps, train_test_split
from models.benchmarks import SimpleRepeater
from models.Transformer_encoder import TransformerEncoder
from models.Pyraformer.Pyraformer_SS import Model as Pyraformer
from models.Pyraformer.Pyraformer_SS import pyraformer_params
# TODO from models.Informer import Informer

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

if __name__ == "__main__":
    num_epochs = 2 # TODO
    learning_rate = 0.01
    batch_size = 1024
    positional_embedding_max_len = batch_size * 2
    
    hidden_size = 1024
    num_layers = 4
    
    num_classes = 153
    
    network1 = TransformerEncoder(hidden_size, 8, input_size, num_layers, positional_embedding_max_len)
    network2 = SimpleRepeater(input_size)
    network3 = Pyraformer(pyraformer_params(num_classes, hidden_size, n_head=8, n_layer=num_layers, batch_size=batch_size, inner_hid=hidden_size, d_k=hidden_size/4, d_v=hidden_size/4, device='cpu')) # !!! Change device to cuda if you're running a model of any significant size.
# =============================================================================
#     network2 = Informer(input_size, input_size, input_size, 1,
#                         d_model = hidden_size, n_heads = 8,
#                         e_layers = math.ceil(num_layers/2),
#                         d_layers = math.floor(num_layers/2),
#                         d_ff = hidden_size, activation = "relu", device = torch.device("cpu"), 
#                         positional_embedding_max_len = positional_embedding_max_len)
#     
# =============================================================================
    criterion = torch.nn.MSELoss()    # mean-squared error for regression
    optimizer1 = torch.optim.Adam(network1.parameters(), lr=learning_rate)
    # optimizer2 = torch.optim.Adam(network2.parameters(), lr=learning_rate)
    optimizer3 = torch.optim.Adam(network3.parameters(), lr=learning_rate)
    
    print("Model 1 (Transformer) has %d parameters. Model 2 (Benchmark) has %d parameters. Model 3 (Pyraformer) has %d parameters" % (count_parameters(network1), count_parameters(network2), count_parameters(network3)))
    
    train_losses1 = []
    test_losses1  = []
    train_losses2 = []
    test_losses2  = []
    train_losses3 = []
    test_losses3  = []
    
    train_data, test_data = load_data("mocap/brownies1.c3d", batch_size)
    
    print("Beginning training...")
    
    # Train the model
    for epoch in range(num_epochs):
        network1.train()
        network2.train()
        network3.train()
        outputs1 = network1(train_data[:, :-predict_length], min_seq_length)
        outputs2 = network2(train_data[:, :-predict_length], min_seq_length)
        outputs3 = network3(train_data[:, :-predict_length], min_seq_length)
        
        optimizer1.zero_grad()
        # optimizer2.zero_grad()
        optimizer3.zero_grad()
        gc.collect()
        
        # obtain the loss function
        loss1 = criterion(outputs1, train_data[:, min_seq_length+predict_length:])
        loss2 = criterion(outputs2, train_data[:, min_seq_length+predict_length:])
        loss3 = criterion(outputs3, train_data[:, min_seq_length+predict_length:])
        
        loss1.backward()
        # loss2.backward()
        loss3.backward()
        
        optimizer1.step()
        # optimizer2.step()
        optimizer3.step()
        
        train_losses1.append(loss1.item())
        train_losses2.append(loss2.item())
        train_losses3.append(loss3.item())
        
        gc.collect()
        network1.eval()
        network2.eval()
        network3.eval()
        
        with torch.no_grad():
            train_predict1 = network1(test_data[:, :-predict_length], min_seq_length)
            train_predict2 = network2(test_data[:, :-predict_length], min_seq_length)
            train_predict3 = network3(test_data[:, :-predict_length], min_seq_length)
            test_loss1 = criterion(train_predict1, test_data[:, min_seq_length+predict_length:])
            test_loss2 = criterion(train_predict2, test_data[:, min_seq_length+predict_length:])
            test_loss3 = criterion(train_predict3, test_data[:, min_seq_length+predict_length:])
            test_losses1.append(test_loss1.item())
            test_losses2.append(test_loss2.item())
            test_losses3.append(test_loss3.item())
            if epoch % 1 == 0:
                print("Epoch: %d, Transformer train loss: %1.5f, Transformer test loss: %1.5f, Benchmark train loss: %1.5f, Benchmark test loss: %1.5f, Pyraformer train loss: %1.5f, Pyraformer test loss: %1.5f" % (epoch, loss1.item(), test_loss1.item(), loss2.item(), test_loss2.item(), loss3.item(), test_loss3.item()))
    
    plt.plot(train_losses1, label = "Train (Transformer)")
    plt.plot(test_losses1, label = "Test (Transformer)")
    plt.plot(train_losses2, label = "Train (Benchmark)")
    plt.plot(test_losses2, label = "Test (Benchmark)")
    plt.plot(train_losses3, label = "Train (Pyraformer)")
    plt.plot(test_losses3, label = "Test (Pyraformer)")
    plt.legend()
    plt.yscale('log')
    plt.show()
    # TODO torch.save(network1.state_dict(), "TrainedKitchenTransformer.pt")
    # TODO torch.save(network2.state_dict(), "TrainedKitchenInformer.pt")
    torch.save(network3.state_dict(), "TrainedKitchenPyraformer.pt")