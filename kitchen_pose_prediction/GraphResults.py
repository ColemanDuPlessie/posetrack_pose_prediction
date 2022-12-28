# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 07:56:04 2022

@author: coley
"""

import matplotlib.pyplot as plt

COLORS = {
    "Transformer (encoder only)" : {"Train" : "#bb0000",
                     "Test" : "#00bb00",
                     "Multistep" : "#0000bb"},
    "LSTM" : {"Train" : "#bb6600",
                     "Test" : "#bbbb00",
                     "Multistep" : "#6600bb"},
    "Informer" : {"Train" : "#bb0099",
                     "Test" : "#00aaaa",
                     "Multistep" : "#0066bb"},
    "Benchmark" : {"Train" : "#bb6666",
                     "Test" : "#003300",
                     "Multistep" : "#6666bb"},
    }

def parse_losses(file):
    results = {}
    with open(file, 'r') as data: data_str = data.read()
    mode = "name" # mode can be one of "name", "train", "test", or "multistep".
    target = ""
    idx = 0
    while idx < len(data_str):
        if mode == "name":
            if data_str[idx+1:idx+6] == "train":
                mode = "train"
                results[target] = ([], [], [])
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
                results[target] = (results[target][0], results[target][1])
                target = ""
            elif data_str[idx:idx+9] == "multistep":
                mode = "multistep"
                idx += 10
            else:
                next_loss = ""
                while idx < len(data_str) and data_str[idx] != " " and data_str[idx] != "\n":
                    next_loss += data_str[idx]
                    idx += 1
                idx += 1
                results[target][1].append(float(next_loss))
        elif mode == "multistep":
            if data_str[idx-1] == "\n":
                mode = "name"
                target = ""
            else:
                next_loss = ""
                while idx < len(data_str) and data_str[idx] != " " and data_str[idx] != "\n":
                    next_loss += data_str[idx]
                    idx += 1
                idx += 1
                results[target][2].append(float(next_loss))
    return results

def parse_multistep_losses(filenames):
    losses = []
    for filename in filenames:
        losses.append(parse_multistep_loss(filename))
    return losses

def parse_multistep_loss(filename):
    with open(filename, 'r') as data: data_str = data.read()
    mode = "name" # mode can be one of "name", "train", or "test".
    target = ""
    idx = 0
    while idx < len(data_str) and data_str[idx] != "]":
        if mode == "name":
            if data_str[idx+1] == "[":
                mode = "test"
                if "Multistep" in target:
                    target = target[:-9]
                results = []
                idx += 2
            else:
                target += data_str[idx]
                idx += 1
        elif mode == "test":
            next_loss = ""
            while idx < len(data_str) and data_str[idx] != "," and data_str[idx] != "]":
                next_loss += data_str[idx]
                idx += 1
            idx += 2
            results.append(float(next_loss))
    return results

def graph_parsed_losses(parsed_losses, title = None):
    for name, losses in parsed_losses.items():
        plt.plot(losses[0], label = "Train (%s)" % name, color = COLORS[name]["Train"])
        plt.plot(losses[1], label = "Test (%s)" % name, color = COLORS[name]["Test"])
        if len(losses) == 3:
            plt.plot(range(0, len(losses[2]*10), 10), losses[2], label = "Multistep Test (%s)" % name, color = COLORS[name]["Multistep"])
    # plt.legend()
    plt.yscale('log')
    plt.xlabel("Epochs")
    plt.ylabel("Mean Squared Error")
    plt.ylim(0.00001, 0.0002)
    plt.title(title)
    plt.show()

def graph_multistep_losses(parsed_multistep_losses, names, title = None):
    for i in range(len(parsed_multistep_losses)):
        name = names[i]
        losses = parsed_multistep_losses[i]
        plt.plot(range(1, len(losses)+1),losses, label = name, color = COLORS[name]["Multistep"])
    plt.legend()
    plt.yscale('log')
    plt.xlabel("Prediction Length (frames)")
    plt.ylabel("Mean Squared Error")
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    graph_parsed_losses(parse_losses("losses8.txt"))
    graph_parsed_losses(parse_losses("losses0.txt"))
    graph_multistep_losses(parse_multistep_losses(["losses2.txt", "losses1(benchmark).txt"]), ["Transformer (encoder only)", "Benchmark"])