# -*- coding: utf-8 -*-
"""
This needs documentation at some point

TODO Our preprocessing is currently in the style of a pipeline. We can probably
get significant speed improvements by combining or reordering some steps of
the pipeline.
"""

import math
import c3d
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

def load_mocap(filename):
    """
    Takes the name of a .c3d file and returns a 3 dimensional np.array. The
    first dimension corresponds to time, and has a variable length, depending
    on the file. The second dimension corresponds to the different points. It
    will be truncated to length 66. Finally, the third dimension corresponds
    to the x, y and z axes. Its length will always be three.
    """
    with open(filename, "rb") as raw_data:
        reader = c3d.Reader(raw_data)
        return np.array([np.delete(np.delete(frame[1], 66, 0), np.s_[3:], 1) for frame in reader.read_frames()])

def preprocess_mocap(unprocessed, nonhuman_point_qty = 15, epsilon = 0.01):
    """
    Takes unprocessed, a motion capture (such as one created by load_mocap),
    and returns that motion capture with the nonhuman_point_qty points that it
    is least certain are part of a human taken out.
    """
    motionless_frames = np.array([0 for i in range(len(unprocessed[0]))]) # We use a numpy array so that we can use argsort to quickly find the indices of the n lowest numbers.
    for frame_idx in range(1, len(unprocessed)): # TODO: Doing this as a nested loop is not time-efficient. I'll need to come back to this and rewrite it to not use nested loops.
        dist = unprocessed[frame_idx] - unprocessed[frame_idx-1]
        for point_idx in range(len(unprocessed[0])):
            eucledian_dist = math.sqrt(dist[point_idx][0]**2+dist[point_idx][1]**2+dist[point_idx][2]**2)
            if eucledian_dist < epsilon:
                motionless_frames[point_idx] += 1
    nonhuman_idxs = sorted(motionless_frames.argsort()[-nonhuman_point_qty:], reverse = True)
    ans = unprocessed
    for idx in nonhuman_idxs:
        ans = np.delete(ans, idx, 1)
    return ans

def remove_bad_mocap(preprocessed, epsilon = 1e-8):
    bad_frames = set()
    for frame_idx in range(1, len(preprocessed)): # TODO: This should also probably not be a loop.
        movement = sum(sum(abs(preprocessed[frame_idx] - preprocessed[frame_idx-1]))) # TODO: Taxicab distance may not be ideal
        if movement < epsilon:
            bad_frames.add(frame_idx)
            bad_frames.add(frame_idx-1)
    bad_frames = sorted(bad_frames, reverse = True)
    ans = []
    remaining_preprocessed = preprocessed
    for frame in bad_frames:
        truncated_bit = remaining_preprocessed[frame+1:]
        if len(truncated_bit) > 0:
            ans.append(truncated_bit)
        remaining_preprocessed = remaining_preprocessed[:frame]
    ans.append(remaining_preprocessed)
    return ans

def convert_mocap_to_velo(mocap):
    """
    Takes mocap, a motion capture (like the output of remove_bad_mocap)
    and returns a np.array of the same shape, except with one fewer frame. The
    output will be the derivative of the input.
    """
    ans = mocap.copy()
    for frame_idx in range(1, len(ans)):
        ans[frame_idx] -= ans[frame_idx-1]
    return ans[1:]
    
def convert_mocap_to_tensor(mocap):
    """
    This function will work on a motion capture in any stage of preprocessing.
    It returns a 2 dimensional tensor, of dimension (n, 3m), where n is the
    number of frames in the motion capture and m is the number of points.
    """
    return torch.Tensor(mocap).reshape(mocap.shape[0], -1)
    
def load_and_preprocess_mocaps(files, velo=False):
    """
    This function takes files (either a string or an iterable of strings
    corresponding to filenames) and outputs a list of Tensors. If a file
    contains a period of time where nothing moves or a quick jump somewhere,
    it will split the tensor in half around the ugly part and remove it (this
    means that you are likely to receive more tensors than the number of files
    you pass in).
    """
    second_step_individual = (lambda x: convert_mocap_to_tensor(convert_mocap_to_velo(x))) if velo else convert_mocap_to_tensor
    if isinstance(files, str):
        return [second_step_individual(mocap) for mocap in remove_bad_mocap(preprocess_mocap(load_mocap(files)))]
    ans = []
    for file in files:
        ans.extend(load_and_preprocess_mocaps(file, velo))
    return ans

def batch_tensors(loaded_tensors, batch_size = 1024):
    """
    Takes loaded_tensors, an iterable of tensors of variable sizes (such as
    from load_and_preprocess_mocap) and returns a tensor consisting of the
    batched data. Any excess data that doesn't fit evenly into a batch is
    thrown out.
    """
    ans = []
    for tensor in loaded_tensors:
        for batch_num in range(tensor.size()[0]//batch_size):
            ans.append(tensor[batch_num*batch_size:(batch_num+1)*batch_size])
    return torch.stack(ans)

def normalize_data(batched_data, return_scaler = False):
    """
    Takes batched_data, a Tensor, and returns a copy scaled so that each point
    stays in the 0-1 range. If return_scaler is True, it will return a
    two-element tuple whose first element is the normalized data and whose
    second element is a sklearn.preprocessing.MinMaxScaler fit to the data.
    """
    sc = MinMaxScaler()
    cpu_batched_data = batched_data.to('cpu')
    for batch in cpu_batched_data: sc.partial_fit(batch)
    ans = torch.Tensor(np.array([sc.transform(batch) for batch in cpu_batched_data]))
    if return_scaler: return (ans, sc)
    else: return ans

def load_and_prepare_mocaps(files, batch_size = 1024, return_scaler = False, velo = False):
    """
    This is probably the function you want to use. It reads the .c3d file or
    files specified by the files parameter, preprocesses and cleans the data,
    batches it (the batch size can be specified with the batch_size parameter),
    and normalizes it to fit in 0-1 range (if return_scaler is True, this
    function will return a tuple whose first element is the data and whose
    second element is a sklearn.preprocessing.MinMaxScaler fit to it).
    """
    return normalize_data(batch_tensors(load_and_preprocess_mocaps(files, velo), batch_size), return_scaler)

def train_test_split(batches, train_qty = 0.67):
    train_size = int(len(batches) * train_qty)
    train_data = batches[:train_size]
    test_data = batches[train_size:]
    
    return train_data, test_data

if __name__ == "__main__":
    batch_size = 1024
    mocaps = ("mocap/brownies1.c3d", "mocap/brownies3.c3d", "mocap/brownies4.c3d",
                                       "mocap/eggs1.c3d", "mocap/eggs3.c3d", "mocap/eggs4.c3d", 
                                       "mocap/pizza1.c3d", "mocap/pizza2.c3d", "mocap/pizza3.c3d", "mocap/pizza4.c3d", 
                                       "mocap/salad1.c3d", "mocap/salad2.c3d", "mocap/salad3.c3d", "mocap/salad4.c3d", 
                                       "mocap/sandwich1.c3d", "mocap/sandwich2.c3d", "mocap/sandwich3.c3d", "mocap/sandwich4.c3d")
    loaded_mocaps = load_and_prepare_mocaps(mocaps, batch_size, velo=True)
    print("Done loading!")
    loaded_mocaps = loaded_mocaps[torch.randperm(loaded_mocaps.shape[0])]
    print("Done shuffling!")
    for num, batch in enumerate(loaded_mocaps):
        torch.save(batch.clone(), "preprocessed_data/60fps_velo_1024/batch%s.pt" % str(num).zfill(4))