# -*- coding: utf-8 -*-
"""
This needs documentation at some point
"""

import c3d

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
    return reader.read_frames()

if __name__ == "__main__":
    with open('mocap/brownies_.c3d', 'rb') as file:
        reader = c3d.Reader(file)
        frame_qty = sum(1 for i in reader.read_frames())
        for i, points in enumerate(reader.read_frames()):
            print('Frame {}: {}'.format(i, points[1].round(2)))