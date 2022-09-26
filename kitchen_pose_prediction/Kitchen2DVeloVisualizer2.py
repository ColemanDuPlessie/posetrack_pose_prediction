# -*- coding: utf-8 -*-
"""
This needs documentation at some point
"""

import math
import random
import tkinter as tk
import torch
from sklearn.preprocessing import MinMaxScaler
from BatchManager import BatchManager
from kitchen_pose_prediction.ReallySimpleModelComparisonTest import SineWaveMaker
from models.LSTM import LSTMBenchmark
from models.Transformer_encoder import TransformerEncoder
from models.Informer import Informer
from ReallySimpleModelComparisonTest import ModelWrapper, MultiModelHandler

input_size = 153
hidden_size = 768
num_layers = 5
min_pred_len = 128
max_pred_len = 1024
learning_rate = 0.00005
batch_size = 1024
positional_embedding_max_len = batch_size * 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pretrained_model_to_view = "TrainedKitchenTransformer2.pt"

models = {LSTMBenchmark(hidden_size, input_size, input_size, num_layers): (min_pred_len,),
          TransformerEncoder(hidden_size, 8, input_size, num_layers, max_pred_len * 2): (min_pred_len,),
          Informer(input_size, input_size, input_size, 1, d_model=hidden_size, n_heads=8,
                   e_layers=math.ceil(num_layers / 2), d_layers=math.floor(num_layers / 2),
                   d_ff=hidden_size, activation="relu", positional_embedding_max_len=max_pred_len * 2): (
          min_pred_len, 1)}

batches_at_once = 1

data = torch.utils.data.DataLoader(SineWaveMaker(timesteps=2048, min_period=128, dimension=1), batches_at_once, True, generator=torch.Generator(device=device))

state_dict = torch.load(pretrained_model_to_view, map_location=device)
for model_type in models:
    try:
        model_type.load_state_dict(state_dict)
        model = model_type
        inputs = models[model]
        break
    except RuntimeError:
        continue

assert model in models
network = ModelWrapper(TransformerEncoder(hidden_size, 8, input_size, num_layers, positional_embedding_max_len, dropout=0.0), "Transformer (0% dropout)", torch.optim.Adam, torch.nn.MSELoss(), {"lr": learning_rate})
for epoch in range(2000):
    network.test(data, min_pred_len, 1)
a = 1
prediction_generator = (frame for frame in network.test(data, min_pred_len, 1).reshape(-1, input_size))

# randomize_batch_series = True
# batches_to_use = 2
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# print("Pytorch running on " + str(device))
#
# torch.set_default_tensor_type('torch.cuda.FloatTensor' if torch.cuda.is_available() else 'torch.FloatTensor')
#
# input_size = 153
# hidden_size = 1024
# num_layers = 6
# min_pred_len = 128
# max_pred_len = 1024
#
# models = {LSTMBenchmark(hidden_size, input_size, input_size, num_layers): (min_pred_len,),
#           TransformerEncoder(hidden_size, 8, input_size, num_layers, max_pred_len * 2): (min_pred_len,),
#           Informer(input_size, input_size, input_size, 1, d_model=hidden_size, n_heads=8,
#                    e_layers=math.ceil(num_layers / 2), d_layers=math.floor(num_layers / 2),
#                    d_ff=hidden_size, activation="relu", positional_embedding_max_len=max_pred_len * 2): (
#           min_pred_len, 1)}
#
# pretrained_model_to_view = "TrainedKitchenTransformer2.pt"
# dataset_to_view = "shuffled_velo_1024"
#
# root = tk.Tk()
# root.title("Kitchen Prediction Visualizer (Velo)")
# root.geometry("900x1200")
#
# height = 800
# width = 1200
# scale = 60
#
# grid_spacing = 120
# grid = [(x * grid_spacing, y * grid_spacing) for y in range(5, -1, -1) for x in range(9)][:51]
#
# offset_x = 150
# offset_y = 150
#
# projection = (0, 2)  # 0 = x, 1 = y, 2 = z
#
# canvas = tk.Canvas(root, bg="white", height=height, width=width)
#
# batches_at_once = 1
#
# def load_data():
#     return torch.utils.data.DataLoader(SineWaveMaker(timesteps=4096, min_period=128, dimension=1), batches_at_once, True, generator=torch.Generator(device=device))
#
#
# def draw_arrow(origin, tip, color):
#     """
#     origin and tip are 2-element tuples representing x-y coordinates.
#     color is a color (hex code or name)
#     """
#     # print("X: " + str(tip[0]-origin[0]) + " Y: " + str(tip[1]-origin[1]))
#     origin_x = float(origin[0]) - offset_x
#     origin_y = height - float(origin[1]) - offset_y
#     tip_x = float(tip[0]) - offset_x
#     tip_y = height - float(tip[1]) - offset_y
#     canvas.create_line(origin_x, origin_y, tip_x, tip_y, fill=color, width=2)
#
#
# class Person:
#
#     def __init__(self, frame_generator, color="red"):
#         self.frame_generator = frame_generator
#         self.color = color
#
#     def draw(self):
#         frame = next(self.frame_generator)
#         frame = [point * 2 - 1 for point in frame]
#         for point in range(0, len(frame), 3):
#             origin = grid[point // 3]
#             draw_arrow(origin, (
#             origin[0] + frame[point + projection[0]] * scale, origin[1] + frame[point + projection[1]] * scale),
#                        self.color)
#
#
# sc = MinMaxScaler()
# data = load_data()  # TODO get scaler to inverse-transform
# state_dict = torch.load(pretrained_model_to_view, map_location=device)
# for model_type in models:
#     try:
#         model_type.load_state_dict(state_dict)
#         model = model_type
#         inputs = models[model]
#         break
#     except RuntimeError:
#         continue
#
# assert model in models
# wrapped_model = ModelWrapper(model, "If you're seeing this, it's a bug", None, torch.nn.MSELoss(), {}, False)
# print("Model %s loaded successfully!" % pretrained_model_to_view)
#
# ground_truth_generator = (frame for batch in data for frame in batch.squeeze()[min_pred_len+1:])
# with torch.no_grad():
#     prediction_generator = (frame for frame in wrapped_model.test(data, min_pred_len, 1).reshape(-1, input_size))
# ground_truth = Person(ground_truth_generator, "green")
# prediction = Person(prediction_generator, "red")
#
# rendering = True
#
#
# def render(delay=None):
#     global rendering
#     canvas.delete("all")
#     ground_truth.draw()
#     prediction.draw()
#     if delay is not None and rendering is True:
#         root.after(delay, render, delay)
#     elif delay is not None:
#         rendering = True
#
#
# next_frame = tk.Button(root, text="Next Frame", command=render)
# next_frame.pack(side=tk.TOP)
# play = tk.Button(root, text="Play", command=lambda: render(8))
# play.pack(side=tk.TOP)
#
#
# def pause_func():
#     global rendering
#     rendering = False
#
#
# pause = tk.Button(root, text="Pause", command=pause_func)
# pause.pack(side=tk.TOP)
# canvas.pack(side=tk.BOTTOM)
#
# print("Beginning rendering!")
# render()
# # root.after(8, render, 8)
#
# root.mainloop()