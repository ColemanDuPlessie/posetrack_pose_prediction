# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 10:24:02 2022

@author: coley
"""

import json
import os

def convert_raw_dict(parsed_json):
    ans = [[]]
    categories = {category["id"] : tuple(category["keypoints"]) for category in parsed_json["categories"]}
    for annotation in parsed_json["annotations"]:
        while len(ans) < annotation["track_id"] + 1:
            ans.append([])
        ans[annotation["track_id"]].append(({categories[annotation["category_id"]][i] : (annotation["keypoints"][i*3], annotation["keypoints"][i*3+1]) if annotation["keypoints"][i*3+2] == 1 else None for i in range(len(annotation["keypoints"])//3)}, annotation["bbox"] if "bbox" in annotation.keys() else None, annotation["bbox_head"]))
    return ans

def load_json(filename):
    with open(filename, "r") as to_parse:
        parsed_json = json.load(to_parse)
    return convert_raw_dict(parsed_json)

def load_directory(directory):
    ans = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            ans.extend(load_json(os.path.join(directory, filename)))
    return ans

def _split_video_on_condition(video, condition):
    """
    Takes video, an ordered iterable of frames, and condition, a function that
    takes a frame as its only parameter, and returns all the sub-videos created
    by removing all frames which meet the condition.
    """
    if len(video) == 0: return []
    for idx, frame in enumerate(video):
        if condition(frame):
            if idx == 0: return _split_video_on_condition(video[1:], condition)
            elif idx == len(video) - 1: return [video[:idx]]
            else: return _split_video_on_condition(video[:idx], condition) + _split_video_on_condition(video[idx+1:], condition)
    return [video]

def _split_video_on_empty_frame(video):
    if len(video) == 0: return []
    for idx, frame in enumerate(video):
        if all(item == None for item in frame[0].values()):
            if idx == 0: return _split_video_on_empty_frame(video[1:])
            elif idx == len(video) - 1: return [video[:idx]]
            else: return _split_video_on_empty_frame(video[:idx]) + _split_video_on_empty_frame(video[idx+1:])
    return [video]

def frame_is_empty(frame): return all(item == None for item in frame[0].values())
def frame_is_imperfect(frame): return tuple(frame[0].values()).count(None) > 2 or any(point != None and any(coord < 0 for coord in point) for point in frame[0].values())

def split_videos_on_empty_frames(inputs):
    outputs = []
    for video in inputs:
        outputs.extend(_split_video_on_condition(video, frame_is_empty))
    return outputs

def split_videos_on_imperfect_frames(inputs):
    outputs = []
    for video in inputs:
        outputs.extend(_split_video_on_condition(video, frame_is_imperfect))
    return outputs

def remove_videos_below_length(inputs, length):
    return [video for video in inputs if len(video) >= length]

def convert_frame_to_list(frame):
    return [item for key in frame[0].keys() if key not in ("left_ear", "right_ear") for item in frame[0][key]]

def create_sliding_windows(data, window_input_len, window_output_len):
    ans = [[], []]
    for video in data:
        for i in range(len(video)-window_input_len-window_output_len):
            _x = video[i:(i+window_input_len)]
            _y = video[i+window_input_len:i+window_input_len+window_output_len]
            ans[0].append(_x)
            ans[1].append(_y)
    return ans

def load_to_list(dirname, min_clip_len):
    return [[convert_frame_to_list(frame) for frame in video] for video in remove_videos_below_length(split_videos_on_imperfect_frames(load_directory(dirname)), min_clip_len)]

if __name__ == "__main__":
    
    ready_to_input = load_directory("posetrack_data/annotations/train/")
    
    missing = {}
    total = 0
    perfect = 0
    nonexistant = 0
    for key in ready_to_input[0][0][0].keys():
        missing[key] = 0
        for video in ready_to_input:
            for frame in video:
                if frame[0][key] == None:
                    missing[key] += 1
                    
    for video in ready_to_input:
        for frame in video:
            total += 1
            if tuple(frame[0].values()).count(None) <= 2: # left_ear and right_ear never appear in the training set
                perfect += 1
            if all(item == None for item in frame[0].values()):
                nonexistant += 1
    print(missing, total, perfect, nonexistant)
    
    ready_to_input = remove_videos_below_length(split_videos_on_imperfect_frames(ready_to_input), 8)
    
    missing = {}
    total = 0
    perfect = 0
    nonexistant = 0
    brokenness = [0 for i in range(15)]
    lengths = [0 for i in range(32)]
    perfect_videos = 0
    perfect_video_frames = 0
    for key in ready_to_input[0][0][0].keys():
        missing[key] = 0
        for video in ready_to_input:
            for frame in video:
                if frame[0][key] == None:
                    missing[key] += 1   
    for video in ready_to_input:
        lengths[len(video)] += 1
        for frame in video:
            total += 1
            if tuple(frame[0].values()).count(None) <= 2: # left_ear and right_ear never appear in the training set
                perfect += 1
            else:
                brokenness[tuple(frame[0].values()).count(None)-2] += 1
            if all(item == None for item in frame[0].values()):
                nonexistant += 1
    for video in ready_to_input:
        for frame in video:
            if tuple(frame[0].values()).count(None) > 2:
                break
        else:
            perfect_videos += 1
            perfect_video_frames += len(video)
    print("\n\n", missing, total, perfect, nonexistant, brokenness, "\n", lengths, "\n", perfect_videos, perfect_video_frames)