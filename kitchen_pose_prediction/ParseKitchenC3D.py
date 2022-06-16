# -*- coding: utf-8 -*-
"""
This needs documentation at some point
"""

import c3d
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

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
    return [np.delete(np.delete(frame[1], 66, 0), np.s_[3:], 1) for frame in raw_data]

def load_and_preprocess_mocap(filename):
    return preprocess_mocap(load_mocap(filename))

def remove_bad_mocap(mocap, epsilon = 1e-8, max_natural_movement = 1): # TODO I'm pretty sure that the epsilon is wrong...
    sc = MinMaxScaler()
    flattened_input = np.array([i.flatten() for i in mocap])
    input_frames = sc.fit_transform(flattened_input)
    ans = []
    velo_list = []
    current_frames = [mocap[0]]
    for idx in range(1, len(input_frames)):
        total_velo = np.linalg.norm(input_frames[idx]-input_frames[idx-1]) # Eucledian distance
        velo_list.append(total_velo)
        if total_velo < epsilon or total_velo > max_natural_movement:
            if len(current_frames) != 0:
                ans.append(current_frames)
                current_frames = []
        else:
            current_frames.append(input_frames[idx])
    if len(current_frames) != 0:
        ans.append(current_frames)
        current_frames = []
    plt.xscale("log")
    plt.hist(velo_list, log = True, bins = 2500)
    transformed_ans = []
    for item in ans:
        transformed_ans.append(sc.inverse_transform([frame.flatten() for frame in item]))
    return transformed_ans

def find_velocity(raw_position):
    """
    Takes a preprocessed motion capture (from the load_and_preprocess_mocap
    function) and returns a generator whose length is one less than the input's
    length and whose contents are the same shape, but now correspond to the
    velocity of points, instead of their absolute position.
    """
    return [raw_position[idx] - raw_position[idx-1] for idx in range(1, len(raw_position))]

def remove_bad_velo_mocap(mocap, epsilon = 4, max_natural_movement = 10): # TODO I'm pretty sure that the epsilon is wrong...
    sc = MinMaxScaler()
    flattened_input = np.array([i.flatten() for i in mocap])
    input_frames = sc.fit_transform(flattened_input)
    ans = []
    current_frames = [mocap[0]]
    for idx in range(1, len(input_frames)):
        total_velo = np.linalg.norm(input_frames[idx]) # Eucledian distance
        if total_velo < epsilon or total_velo > max_natural_movement:
            if len(current_frames) != 0:
                ans.append(current_frames)
                current_frames = []
        else:
            current_frames.append(input_frames[idx])
    if len(current_frames) != 0:
        ans.append(current_frames)
        current_frames = []
    transformed_ans = []
    for item in ans:
        transformed_ans.append(sc.inverse_transform([frame.flatten() for frame in item]))
    return transformed_ans

def load_velo_mocaps(filenames):
    ans = []
    for filename in filenames:
        ans.extend(remove_bad_velo_mocap(find_velocity(load_and_preprocess_mocap(filename))))
    return ans

if __name__ == "__main__":
    print([item for item in load_velo_mocaps(["mocap/brownies1.c3d"])])