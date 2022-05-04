# -*- coding: utf-8 -*-
"""
This needs documentation at some point
"""

import c3d
import numpy as np

def load_mocap(filename):
    """
    Takes the name of a .c3d file and returns an ordered iterable of tuples.
    Each tuple represents a frame and has three elements. The first element is
    the (1-indexed) frame number, the second element is a numpy array of size
    (67, 5), although the last two columns and last row seem to only contain
    zeros. The third element of each frame is a numpy array that seems to
    always be empty.
    """
    with open(filename, "rb") as raw_data:
        reader = c3d.Reader(raw_data)
        return list(reader.read_frames())

def preprocess_mocap(raw_data):
    return (np.delete(np.delete(frame[1], 66, 0), np.s_[3:], 1) for frame in raw_data)

def load_and_preprocess_mocap(filename):
    return preprocess_mocap(load_mocap(filename))

def create_sliding_window(data, window_input_len, window_output_len, flatten = False):
    ans = [[], []]
    if flatten: data = [frame.flatten() for frame in data]
    for i in range(len(data)-window_input_len-window_output_len):
        _x = data[i:(i+window_input_len)]
        _y = data[i+window_input_len:i+window_input_len+window_output_len]
        ans[0].append(_x)
        ans[1].append(_y)
    return ans

if __name__ == "__main__":
    with open('mocap/brownies_.c3d', 'rb') as file:
        reader = c3d.Reader(file)
        frame_qty = sum(1 for i in reader.read_frames())
        for i, points in enumerate(reader.read_frames()):
            print('Frame {}: {}'.format(i, points[1].round(2)))