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
from ParsePosetrackJSON import load_to_list, create_sliding_windows

seq_length = 7
predict_length = 1

if __name__ == "__main__":
    print("Loading data...")
    data = load_to_list("posetrack_data/annotations/train/", 8)
    
    data = create_sliding_windows(data, seq_length, predict_length)
    sc = MinMaxScaler()
    sc.fit([frame for in_or_out in data for example in in_or_out for frame in example])
    
    print("Converting to Tensor...")
    inputs, outputs = data
    
    x = [sc.transform(item) for item in inputs]
    y = [sc.transform(item) for item in outputs]
    
    train_size = int(len(y) * 0.67)
    test_size = len(y) - train_size
    
    dataX = Variable(torch.Tensor(np.array(x)))
    dataY = Variable(torch.Tensor(np.array(y)))
    
    trainX = Variable(torch.Tensor(np.array(x[0:train_size])))
    trainY = Variable(torch.Tensor(np.array(y[0:train_size])))
    
    testX = Variable(torch.Tensor(np.array(x[train_size:len(x)])))
    testY = Variable(torch.Tensor(np.array(y[train_size:len(y)])))

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
    num_epochs = 50
    learning_rate = 0.01
    
    input_size = 30
    hidden_size = 200
    num_layers = 3
    
    num_classes = 30
    
    lstm = LSTM(num_classes, input_size, hidden_size, num_layers)
    
    criterion = torch.nn.MSELoss()    # mean-squared error for regression
    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
    #optimizer = torch.optim.SGD(lstm.parameters(), lr=learning_rate)
    
    train_losses = []
    test_losses  = []
    
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
    torch.save(lstm.state_dict(), "TrainedPosetrackLSTM.pt")