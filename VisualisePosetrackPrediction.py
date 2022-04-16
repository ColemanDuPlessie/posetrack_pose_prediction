# -*- coding: utf-8 -*-
"""
This needs documentation at some point
"""

import tkinter as tk
import numpy as np
from ParsePosetrackJSON import load_to_list, create_sliding_windows
from PosetrackLSTMTest import LSTM
from sklearn.preprocessing import MinMaxScaler
from torch import load, Tensor

skeleton = ((14, 12),
            (12, 10),
            (15, 13),
            (13, 11),
            (10, 11),
            (4, 10),
            (5, 11),
            (4, 5),
            (4, 6),
            (5, 7),
            (6, 8),
            (7, 9),
            (2, 3),
            (1, 2),
            (1, 3))

root = tk.Tk()
root.title("Posetrack Prediction Visualizer")

height = 900
width = 1200

canvas = tk.Canvas(root, bg = "white", height = height, width = width)
canvas.pack()

def draw_line(start, end, color):
    canvas.create_line(start[0]*width, start[1]*height, end[0]*width, end[1]*height, fill=color, width = 4)
    # canvas.create_line(start[0], start[1], end[0], end[1], fill=color, width = 4)

def draw_skeleton(coords, color):
    for line in skeleton:
        draw_line((coords[line[0]*2-2], coords[line[0]*2-1]), (coords[line[1]*2-2], coords[line[1]*2-1]), color)

class Person:
    
    def __init__(self, frame_generator, color = "red"):
        self.frame_generator = frame_generator
        self.color = color
    
    def draw(self):
        draw_skeleton(next(self.frame_generator), self.color)

data = load_to_list("posetrack_data/annotations/train/", 8)
# data = [[[max(coord, 0) for coord in frame] for frame in video] for video in data]
# max_pos = max(coord for video in data for frame in video for coord in frame)
max_pos = 1280
min_pos = 0
inputs, outputs = create_sliding_windows(data, 7, 1)
sc = MinMaxScaler()
sc.fit([frame for video in data for frame in video])
# data = [[[(coord-min_pos)/(max_pos-min_pos) for coord in frame] for frame in video] for video in data]
x = [sc.transform(item) for item in inputs]
y = [sc.transform(item) for item in outputs]
model = LSTM(30, 30, 200, 3)
model.load_state_dict(load("TrainedPosetrackLSTM.pt"))
model.eval()
predicted = model(Tensor(np.array(x))).detach().numpy()

# ground_truth = Person([(coord-min_pos)/(max_pos-min_pos) for coord in sc.inverse_transform(item)[0]] for item in y)
# prediction = Person(([(float(coord)-min_pos)/(max_pos-min_pos) for coord in sc.inverse_transform(item)[0]] for item in predicted), "blue")
ground_truth = Person(item[0] for item in y)
prediction = Person((item for item in predicted), "blue")
# prediction = Person(([float(coord) for coord in model(Tensor([item]))[0]] for item in x), "blue")

def render(delay):
    canvas.delete("all")
    ground_truth.draw()
    prediction.draw()
    root.after(delay, render, delay)

root.after(35, render, 35)

root.mainloop()