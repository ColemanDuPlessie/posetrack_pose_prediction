# -*- coding: utf-8 -*-
"""
This needs documentation at some point
"""

import gc # This is an ugly hack
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
from ParseKitchenC3D import load_and_preprocess_mocap

min_seq_length = 100
predict_length = 1

input_size = 198

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

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
    
    train_size = int(len(data) * 0.67)
    test_size = len(data)-train_size
    
    print("Converting train set to Tensor...")
    
    train_data = Variable(torch.Tensor(data[0:train_size])).reshape(1, -1, input_size)
    
    print("Converting test set to Tensor...")
    
    test_data = Variable(torch.Tensor(data[train_size:])).reshape(1, -1, input_size)
    
    return train_data, test_data

class Transformer(nn.Module):
    def __init__(self, hidden_size = 64, heads = 4, frame_dimension = 1, layers = 2):
        super(Transformer, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = frame_dimension
        self.encoding = nn.Linear(frame_dimension, hidden_size)
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(hidden_size, heads, hidden_size, batch_first=True), layers)
        
    def forward(self, y, pre_output_len=1):
        frames = self.encoding(y)
        mask = generate_square_subsequent_mask(frames.size()[1])
        ans = self.transformer(frames, mask)[:, pre_output_len:]
        ans = torch.matmul(ans-self.encoding.bias, torch.linalg.pinv(self.encoding.weight).t()) # This line effectively runs the self.encoding layer in reverse

        return ans

if __name__ == "__main__":
    num_epochs = 5
    learning_rate = 0.01
    batch_size = 1024
    
    hidden_size = 256
    num_layers = 2
    
    num_classes = 198
    
    network = Transformer(hidden_size, 4, input_size, num_layers)
    
    criterion = torch.nn.MSELoss()    # mean-squared error for regression
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    
    train_losses = []
    test_losses  = []
    
    train_data, test_data = load_data("mocap/brownies_.c3d")
    train_data = batch_timeseries_data(train_data, batch_size)
    test_data = batch_timeseries_data(test_data, batch_size)
    
    print("Beginning training...")
    
    # Train the model
    for epoch in range(num_epochs):
        network.train()
        outputs = network(train_data[:, :-predict_length], min_seq_length)
        
        optimizer.zero_grad()
        gc.collect()
        
        # obtain the loss function
        loss = criterion(outputs, train_data[:, min_seq_length+predict_length:])
        
        loss.backward()
        
        optimizer.step()
        
        train_losses.append(loss.item())
        
        gc.collect()
        network.eval()
        
        with torch.no_grad():
            train_predict = network(test_data[:, :-predict_length], min_seq_length)
            test_loss = criterion(train_predict, test_data[:, min_seq_length+predict_length:])
            test_losses.append(test_loss.item())
            if epoch % 1 == 0:
                print("Epoch: %d, train loss: %1.5f, test loss: %1.5f" % (epoch, loss.item(), test_loss.item()))
    
    plt.plot(train_losses, label = "Train")
    plt.plot(test_losses, label = "Test")
    plt.legend()
    plt.yscale('log')
    plt.show()
    torch.save(network.state_dict(), "TrainedKitchenTransformer.pt")