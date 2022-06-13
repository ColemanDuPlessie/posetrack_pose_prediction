# -*- coding: utf-8 -*-
"""
This needs documentation at some point
"""

import gc # This is an ugly hack
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
from ParseKitchenC3D import load_and_preprocess_mocap
from Models import TransformerEncoder, SimpleRepeater# TODO Informer

min_seq_length = 100
predict_length = 1

input_size = 198 - 14*3 # TODO: This is an ugly hack

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def batch_timeseries_data(unbatched_data, batch_size = 256):
    """
    Takes unbatched_data, a Tensor of shape (1, L, N), where L is the length
    of the timeseries input and n is the input size of the neural network.
    Returns a Tensor of shape (L/B, B, N), where B is batch_size. If there
    are frames 'left over' after dividing unbatched_data into L/B batches,
    they are discarded.
    If L < B, batch_timeseries_data will raise a RuntimeError.
    """
    ans = []
    for batch_num in range(unbatched_data.size()[1]//batch_size):
        ans.append(unbatched_data[0, batch_num*batch_size:(batch_num+1)*batch_size])
    return torch.stack(ans)

def normalize_data(examples):
    sc = MinMaxScaler()
    return sc.fit_transform(np.array([i.flatten() for i in examples]))

def load_data(filename):
    """
    This code is in its own function so that the data's intermediate forms
    will be garbage collected when the function ends, freeing up RAM to
    use while training.
    
    Returns a tuple of the form (train_data, test_data)
    """
    print("Loading data...")
    data = load_and_preprocess_mocap(filename)
    
    print("Normalizing data...")
    data = normalize_data(data)
    
    data = data[:, 14*3:] # TODO: This is an ugly hack
    
    train_size = int(len(data) * 0.67)
    test_size = len(data)-train_size
    
    print("Converting train set to Tensor...")
    
    train_data = Variable(torch.Tensor(data[0:train_size])).reshape(1, -1, input_size)
    
    print("Converting test set to Tensor...")
    
    test_data = Variable(torch.Tensor(data[train_size:])).reshape(1, -1, input_size)
    
    return train_data, test_data

if __name__ == "__main__":
    num_epochs = 10 # TODO
    learning_rate = 0.01
    batch_size = 1024
    positional_embedding_max_len = batch_size * 2
    
    hidden_size = 256 # TODO (was 512)
    num_layers = 2
    
    num_classes = 198 - 14*3
    
    network1 = TransformerEncoder(hidden_size, 8, input_size, num_layers, positional_embedding_max_len)
    network2 = SimpleRepeater(input_size)
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
    
    print("Model 1 (Transformer) has %d parameters. Model 2 (Informer) has %d parameters" % (count_parameters(network1), count_parameters(network2)))
    
    train_losses1 = []
    test_losses1  = []
    train_losses2 = []
    test_losses2  = []
    
    train_data, test_data = load_data("mocap/brownies1.c3d")
    train_data = batch_timeseries_data(train_data, batch_size)
    test_data = batch_timeseries_data(test_data, batch_size)
    
    print("Beginning training...")
    
    # Train the model
    for epoch in range(num_epochs):
        network1.train()
        network2.train()
        outputs1 = network1(train_data[:, :-predict_length], min_seq_length)
        outputs2 = network2(train_data[:, :-predict_length], min_seq_length)
        
        optimizer1.zero_grad()
        # optimizer2.zero_grad()
        gc.collect()
        
        # obtain the loss function
        loss1 = criterion(outputs1, train_data[:, min_seq_length+predict_length:])
        loss2 = criterion(outputs2, train_data[:, min_seq_length+predict_length:])
        
        loss1.backward()
        # loss2.backward()
        
        optimizer1.step()
        # optimizer2.step()
        
        train_losses1.append(loss1.item())
        train_losses2.append(loss2.item())
        
        gc.collect()
        network1.eval()
        network2.eval()
        
        with torch.no_grad():
            train_predict1 = network1(test_data[:, :-predict_length], min_seq_length)
            train_predict2 = network2(test_data[:, :-predict_length], min_seq_length)
            test_loss1 = criterion(train_predict1, test_data[:, min_seq_length+predict_length:])
            test_loss2 = criterion(train_predict2, test_data[:, min_seq_length+predict_length:])
            test_losses1.append(test_loss1.item())
            test_losses2.append(test_loss2.item())
            if epoch % 1 == 0:
                print("Epoch: %d, Transformer train loss: %1.5f, Transformer test loss: %1.5f, Informer train loss: %1.5f, Informer test loss: %1.5f" % (epoch, loss1.item(), test_loss1.item(), loss2.item(), test_loss2.item()))
    
    plt.plot(train_losses1, label = "Train (Transformer)")
    plt.plot(test_losses1, label = "Test (Transformer)")
    plt.plot(train_losses2, label = "Train (Benchmark)")
    plt.plot(test_losses2, label = "Test (Benchmark)")
    plt.legend()
    plt.yscale('log')
    plt.show()
    # TODO torch.save(network1.state_dict(), "TrainedKitchenTransformer.pt")
    # TODO torch.save(network1.state_dict(), "TrainedKitchenInformer.pt")