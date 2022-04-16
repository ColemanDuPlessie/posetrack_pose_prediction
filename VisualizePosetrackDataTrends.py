# -*- coding: utf-8 -*-
"""
This needs documentation at some point
"""

from ParsePosetrackJSON import load_to_list
import matplotlib.pyplot as plt

data = load_to_list("posetrack_data/annotations/train/", 8)

data = [frame for video in data for frame in video]

xs = [frame[i] for i in range(0, 30, 2) for frame in data]
ys = [frame[i] for i in range(1, 30, 2) for frame in data]
print(max(xs), max(ys))
plt.hist(xs, bins=100)
plt.hist(ys, bins=100)