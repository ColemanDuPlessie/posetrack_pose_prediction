# -*- coding: utf-8 -*-
"""
This needs documentation at some point
"""

import gc # This is an ugly hack
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
from ParseKitchenC3D import load_and_preprocess_mocap
from Models import TwoLayerLSTM

seq_length = 100
predict_length = 1

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
    
    train_size = int(len(data) * 0.67)
    test_size = len(data)-train_size
    
    print("Converting train set to Tensor...")
    
    train_data = Variable(torch.Tensor(data[0:train_size]))
    
    print("Converting test set to Tensor...")
    
    test_data = Variable(torch.Tensor(data[train_size:]))
    
    return train_data, test_data

if __name__ == "__main__":
    num_epochs = 100
    learning_rate = 0.01
    
    input_size = 198
    
    hidden_size = 100
    # TODO num_layers = 2 num_layers must be two with the current architecture
    
    num_classes = 198
    
    lstm = TwoLayerLSTM(hidden_size, input_size)
    
    criterion = torch.nn.MSELoss()    # mean-squared error for regression
    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
    #optimizer = torch.optim.SGD(lstm.parameters(), lr=learning_rate)
    
    train_losses = []
    test_losses  = []
    
    train_data, test_data = load_data("mocap/brownies_.c3d")
    
    print("Beginning training...")
    
    # Train the model
    for epoch in range(num_epochs):
        lstm.train()
        outputs = lstm(train_data[:-predict_length], seq_length)
        outputs = outputs.reshape(outputs.size(0), outputs.size(2))
        
        optimizer.zero_grad()
        gc.collect()
        
        # obtain the loss function
        loss = criterion(outputs, train_data[seq_length+predict_length:])
        
        loss.backward()
        
        optimizer.step()
        
        train_losses.append(loss.item())
        
        gc.collect()
        lstm.eval()
        
        with torch.no_grad():
            train_predict = lstm(test_data[:-predict_length], seq_length)
            train_predict = train_predict.reshape(train_predict.size(0), train_predict.size(2))
            test_loss = criterion(train_predict, test_data[seq_length+predict_length:])
            test_losses.append(test_loss.item())
            if epoch % 1 == 0:
                print("Epoch: %d, train loss: %1.5f, test loss: %1.5f" % (epoch, loss.item(), test_loss.item()))
    
    plt.plot(train_losses, label = "Train")
    plt.plot(test_losses, label = "Test")
    plt.legend()
    plt.yscale('log')
    plt.show()
    torch.save(lstm.state_dict(), "TrainedKitchenLSTM.pt")