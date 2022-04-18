# -*- coding: utf-8 -*-
"""
This needs documentation at some point
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
from ParseKitchenC3D import load_and_preprocess_mocap

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
    
    train_data = Variable(torch.Tensor(data[0:train_size])).view(train_size, 1, 198)
    
    print("Converting test set to Tensor...")
    
    test_data = Variable(torch.Tensor(data[train_size:])).view(test_size, 1, 198)
    
    return train_data, test_data

class LSTM(nn.Module):
    def __init__(self, hidden_layers=64, frame_dimension = 1):
        super(LSTM, self).__init__()
        self.hidden_layers = hidden_layers
        # lstm1, lstm2, linear are all layers in the network
        self.lstm1 = nn.LSTMCell(frame_dimension, self.hidden_layers)
        self.lstm2 = nn.LSTMCell(self.hidden_layers, self.hidden_layers)
        self.linear = nn.Linear(self.hidden_layers, frame_dimension)
        
    def forward(self, y, pre_output_len=1):
        global view
        view = y
        outputs, n_samples = [], y.size(0)
        h_t = torch.zeros(n_samples, self.hidden_layers, dtype=torch.float32)
        c_t = torch.zeros(n_samples, self.hidden_layers, dtype=torch.float32)
        h_t2 = torch.zeros(n_samples, self.hidden_layers, dtype=torch.float32)
        c_t2 = torch.zeros(n_samples, self.hidden_layers, dtype=torch.float32)
        
        for i, frame in enumerate(y.split(1, dim=1)):
            h_t, c_t = self.lstm1(frame, (h_t, c_t)) # initial hidden and cell states
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2)) # new hidden and cell states
            output = self.linear(h_t2) # output from the last FC layer
            if i >= pre_output_len: outputs.append(output)

        # transform list to tensor    
        outputs = torch.cat(outputs, dim=1)
        return outputs

if __name__ == "__main__":
    num_epochs = 5
    learning_rate = 0.01
    
    input_size = 198
    hidden_size = 2
    # TODO num_layers = 2 num_layers must be two with the current architecture
    
    num_classes = 198
    
    lstm = LSTM(hidden_size, input_size)
    
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
        outputs = lstm(train_data, seq_length)
        optimizer.zero_grad()
        
        # obtain the loss function
        loss = criterion(outputs, train_data[:-seq_length])
        
        loss.backward()
        
        optimizer.step()
        
        train_losses.append(loss.item())
        
        lstm.eval()
        with torch.no_grad():
            train_predict = lstm(test_data)
            test_loss = criterion(train_predict, test_data[:-seq_length])
            test_losses.append(test_loss.item())
            if epoch % 1 == 0:
                print("Epoch: %d, train loss: %1.5f, test loss: %1.5f" % (epoch, loss.item(), test_loss.item()))
    
    plt.plot(train_losses, label = "Train")
    plt.plot(test_losses, label = "Test")
    plt.legend()
    plt.yscale('log')
    plt.show()
    torch.save(lstm.state_dict(), "TrainedKitchenLSTM.pt")