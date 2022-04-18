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
from ParseKitchenC3D import load_and_preprocess_mocap, create_sliding_window

seq_length = 100
predict_length = 1

def normalize_data(examples):
    sc = MinMaxScaler()
    sc.fit([frame for in_or_out in examples for example in in_or_out for frame in example])
    
    inputs, outputs = examples
    
    x = [sc.transform(item) for item in inputs]
    y = [sc.transform(item) for item in outputs]
    return x, y

def load_data(filename):
    """
    This code is in its own function so that the data's intermediate forms
    will be garbage collected when the function ends, freeing up RAM to
    use while training.
    
    Returns an array of the form (trainX, trainY, testX, testY)
    """
    print("Loading data...")
    data = load_and_preprocess_mocap(filename)
    
    data = create_sliding_window(data, seq_length, predict_length, flatten = True)
    
    print("Normalizing data...")
    x, y = normalize_data(data)
    
    train_size = int(len(y) * 0.67)
    
    print("Converting train set to Tensor...")
    
    trainX = Variable(torch.Tensor(np.array(x[0:train_size])))
    trainY = Variable(torch.Tensor(np.array(y[0:train_size])))
    
    print("Converting test set to Tensor...")
    
    testX = Variable(torch.Tensor(np.array(x[train_size:len(x)])))
    testY = Variable(torch.Tensor(np.array(y[train_size:len(y)])))
    
    return trainX, trainY, testX, testY

class LSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        
        self.fc = nn.Linear(hidden_size*seq_length, num_classes)

    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        
        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        # hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
        hn = output
        hn = hn.reshape(hn.size()[0], self.hidden_size*self.seq_length)
        out = self.fc(hn)
        return out

if __name__ == "__main__":
    num_epochs = 5
    learning_rate = 0.01
    
    input_size = 198
    hidden_size = 2
    num_layers = 2
    
    num_classes = 198
    
    lstm = LSTM(num_classes, input_size, hidden_size, num_layers)
    
    criterion = torch.nn.MSELoss()    # mean-squared error for regression
    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
    #optimizer = torch.optim.SGD(lstm.parameters(), lr=learning_rate)
    
    train_losses = []
    test_losses  = []
    
    trainX, trainY, testX, testY = load_data("mocap/brownies_.c3d")
    
    print("Beginning training...")
    
    # Train the model
    for epoch in range(num_epochs):
        lstm.train()
        outputs = lstm(trainX)
        optimizer.zero_grad()
        
        # obtain the loss function
        loss = criterion(outputs, trainY)
        
        loss.backward()
        
        optimizer.step()
        
        train_losses.append(loss.item())
        
        lstm.eval()
        with torch.no_grad():
            train_predict = lstm(testX)
            test_loss = criterion(train_predict, testY)
            test_losses.append(test_loss.item())
            if epoch % 1 == 0:
                print("Epoch: %d, train loss: %1.5f, test loss: %1.5f" % (epoch, loss.item(), test_loss.item()))
    
    plt.plot(train_losses, label = "Train")
    plt.plot(test_losses, label = "Test")
    plt.legend()
    plt.yscale('log')
    plt.show()
    torch.save(lstm.state_dict(), "TrainedKitchenLSTM.pt")