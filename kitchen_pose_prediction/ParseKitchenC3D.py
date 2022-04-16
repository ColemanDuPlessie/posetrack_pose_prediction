# -*- coding: utf-8 -*-
"""
This needs documentation at some point
"""

import c3d

with open('mocap/brownies_.c3d', 'rb') as file:
    reader = c3d.Reader(file)
    frame_qty = sum(1 for i in reader.read_frames())
    for i, points in enumerate(reader.read_frames()):
        print('Frame {}: {}'.format(i, points[1].round(2)))