# -*- coding: utf-8 -*-
"""
This needs documentation at some point
"""

import tkinter as tk
import torch
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
from ParseKitchenC3D import preprocess_mocap, load_mocap
from kitchen_pose_prediction.ReallySimpleModelComparisonTest import SineWaveMaker
from models.LSTM import TwoLayerLSTM
from models.Transformer_encoder import TransformerEncoder
from models.Informer import Informer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_size = 198 - 14*3 # TODO: This is an ugly hack
min_pred_len = 128
max_pred_len = 1024

models = {TwoLayerLSTM(100, input_size) : (min_pred_len,),
          TransformerEncoder(256, 8, input_size, 2, 2048) : (min_pred_len,),
          Informer(input_size, input_size, input_size, 1, d_model = 256, n_heads = 8, e_layers = 1, d_layers = 1, d_ff = 256, activation = "relu", device = torch.device("cpu"), positional_embedding_max_len = 2048) : (min_pred_len, 1)}

pretrained_model_to_view = "TrainedKitchenTransformer.pt"

root = tk.Tk()
root.title("Kitchen Prediction Visualizer")

height = 900
width = 1200
scale = 0.25

projection = (0, 2) # 0 = x, 1 = y, 2 = z

canvas = tk.Canvas(root, bg = "white", height = height, width = width)
canvas.pack()

def batch_timeseries_data_with_overlap(unbatched_data, batch_size = 256, overlap = 100):
    """
    Behaves as batch_timeseries_data, except the nth batch and the n-1th batch
    share a number of elements equal to overlap.
    """
    ans = []
    for batch_num in range((unbatched_data.shape[0]+overlap)//(batch_size-overlap)):
        ans.append(torch.tensor(unbatched_data[batch_num*(batch_size-overlap):(batch_num+1)*(batch_size-overlap)+overlap]))
    return torch.stack(ans)

def load_data(filename):
    ans = torch.tensor(preprocess_mocap(load_mocap(filename)))
    ans = ans.reshape(ans.shape[0], -1)
    return Variable(ans)

def draw_point(x, y, color):
    x = float(x)*scale + width/2
    y = -float(y)*scale + height/2
    canvas.create_oval(x-2, y-2, x+2, y+2, fill = color, width = 0)

class Person:
    
    def __init__(self, frame_generator, color = "red"):
        self.frame_generator = frame_generator
        self.color = color
    
    def draw(self):
        frame = next(self.frame_generator)
        for point in range(0, len(frame), 3):
            draw_point(frame[point+projection[0]], frame[point+projection[1]], self.color)
         
sc = MinMaxScaler()
data = torch.utils.data.DataLoader(SineWaveMaker(timesteps=4096, min_period=128, dimension=1), 1, True, generator=torch.Generator(device=device))
data = sc.fit_transform(data)
state_dict = torch.load(pretrained_model_to_view)
for model_type in models:
    try:
        model_type.load_state_dict(state_dict)
        model = model_type
        inputs = models[model]
        break
    except RuntimeError:
        continue

assert model in models
model.eval()
print("Model %s loaded successfully!" % pretrained_model_to_view)

ground_truth_generator = (frame for frame in sc.inverse_transform(data)[min_pred_len:])
batched_data = batch_timeseries_data_with_overlap(data, max_pred_len, min_pred_len)
total_expected_len = batched_data.shape[0] * (batched_data.shape[1] - min_pred_len)
with torch.no_grad():
    prediction_generator = (frame for frame in sc.inverse_transform(model(batched_data, *inputs).reshape(total_expected_len, input_size)))
ground_truth = Person(ground_truth_generator)
prediction = Person(prediction_generator, "blue")

def render(delay):
    canvas.delete("all")
    ground_truth.draw()
    prediction.draw()
    root.after(delay, render, delay)

print("Beginning rendering!")
root.after(8, render, 8)

root.mainloop()