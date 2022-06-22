# -*- coding: utf-8 -*-
"""
This needs documentation at some point
"""

import gc # This is an ugly hack
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from ParseKitchenC3D import load_and_prepare_mocaps, train_test_split
from models.Transformer_encoder import TransformerEncoder

min_seq_length = 100
predict_length = 1

input_size = 153

def load_data(filenames, batch_size = 1024, train_qty = 0.67):
    """
    Returns a tuple of the form (train_data, test_data)
    """
    return train_test_split(Variable(load_and_prepare_mocaps(filenames, batch_size)), train_qty)

if __name__ == "__main__":
    num_epochs = 100
    learning_rate = 0.01
    batch_size = 1024
    positional_embedding_max_len = batch_size * 2
    
    hidden_size = 512
    num_layers = 2
    
    num_classes = 153
    
    network = TransformerEncoder(hidden_size, 8, input_size, num_layers, positional_embedding_max_len)
    
    criterion = torch.nn.MSELoss()    # mean-squared error for regression
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    
    train_losses = []
    test_losses  = []
    
    train_data, test_data = load_data("mocap/brownies1.c3d", batch_size)
    
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
    plt.title('With useless points')
    plt.show()
    torch.save(network.state_dict(), "TrainedKitchenTransformer.pt")