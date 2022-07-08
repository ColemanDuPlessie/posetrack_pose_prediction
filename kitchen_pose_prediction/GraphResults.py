# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 07:56:04 2022

@author: coley
"""

import matplotlib.pyplot as plt

def parse_losses(file):
    results = {}
    with open(file, 'r') as data: data_str = data.read()
    mode = "name" # mode can be one of "name", "train", or "test".
    target = ""
    idx = 0
    while idx < len(data_str):
        if mode == "name":
            if data_str[idx+1:idx+6] == "train":
                mode = "train"
                results[target] = ([], [])
                idx += 7
            else:
                target += data_str[idx]
                idx += 1
        elif mode == "train":
            if data_str[idx:idx+4] == "test":
                mode = "test"
                idx += 5
            else:
                next_loss = ""
                while data_str[idx] != " ":
                    next_loss += data_str[idx]
                    idx += 1
                idx += 1
                results[target][0].append(float(next_loss))
        elif mode == "test":
            if data_str[idx-1] == "\n":
                mode = "name"
                target = ""
            else:
                next_loss = ""
                while idx < len(data_str) and data_str[idx] != " " and data_str[idx] != "\n":
                    next_loss += data_str[idx]
                    idx += 1
                idx += 1
                results[target][1].append(float(next_loss))
    return results

def graph_parsed_losses(parsed_losses):
    for name, losses in parsed_losses.items():
        plt.plot(losses[0], label = "Train (%s)" % name)
        plt.plot(losses[1], label = "Test (%s)" % name)
    plt.legend()
    plt.yscale('log')
    plt.show()

if __name__ == "__main__":
    graph_parsed_losses(parse_losses("losses.txt"))