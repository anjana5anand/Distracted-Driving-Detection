# %%
import pandas as pd
import os
import re
import shutil
import numpy as np

import cv2
from imutils.video import VideoStream

import time



# %%

# labels_path = "/mnt/d/University Stuff/OneDrive - SSN Trust/7th Semester/Capstone Project/Data/AI City Full Data/Labels/A1"
labels_path = "/media/viplab/DATADRIVE1/driver_action_recognition/2024-data_labelsAndinstructions/labels/A1/A1_fixed"
# labels_path = "/home/mohitavva/capstone/project/new/labels_fixed"
# videos_path = "/mnt/d/University Stuff/OneDrive - SSN Trust/7th Semester/Capstone Project/Data/AI City Full Data/A1"
# videos_path = "/mnt/d/Capstone_Data/Raw_Footage/A1"
videos_path = "/media/viplab/DATADRIVE1/driver_action_recognition/raw_videos/A1"
# view_filter = "rear"
# saved_output = "/mnt/d/Capstone_Data/Cut_Video_1/"
# saved_output = "mnt/C/Users/mohit/Desktop/Capstone_Data"
# saved_output = "/media/viplab/DATADRIVE1/driver_action_recognition/mobilenet_approach/cut_frames_side_new/"
# saved_output = "/media/viplab/DATADRIVE1/driver_action_recognition/mobilenet_approach/neural_net_frames/cut_frames_rear/"
missing_videos = []
frame_rate = 30  
fps = 5
# frame_skip = 6
# frame_shift = 1
# crop_size = (512, 512)
class_names = ['class_0', 'class_1', 'class_2', 'class_3', 'class_4', 'class_5', 'class_6', 'class_7', 'class_8', 'class_9', 'class_10', 'class_11', 'class_12', 'class_13', 'class_14', 'class_15']
save_type = "image"
# save_type = "video"
skipped_videos = []
test_counter = 0
processed_list = []


# %%
user_labels = os.listdir(labels_path)
# user_labels_length = len(user_labels)

def video_duration(filename1, filename2, filename3, frame_rate):
    video = cv2.VideoCapture(filename1)
    # duration = video.get(cv2.CAP_PROP_POS_MSEC)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    duration1 = round(frame_count/frame_rate)

    video = cv2.VideoCapture(filename2)
    # duration = video.get(cv2.CAP_PROP_POS_MSEC)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    duration2 = round(frame_count/frame_rate)

    video = cv2.VideoCapture(filename3)
    # duration = video.get(cv2.CAP_PROP_POS_MSEC)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    duration3 = round(frame_count/frame_rate)

    duration = max(duration1, duration2, duration3)

    return duration
#/home/viplab/Documents/driver_action_recognition/data_processing/array_generation/arrays
def time_processor(time):
    hours, minutes, seconds = map(int, time.split(":"))
    time_seconds = hours*60*60 + minutes*60 + seconds
    return time_seconds


for user in user_labels:

    user_id = user.split(".")[0]
    user_data_path = labels_path + "/" + user
    user_video_path = videos_path + "/" + user_id

    dir_files = os.listdir(user_video_path)
    sorted_files = sorted(dir_files, key=lambda x: (int(x.split('_')[-1].split('.')[0]), x))
    user_data = pd.read_csv(user_data_path)
    
    for files in range(0, len(sorted_files), 3):
        video_path_temp_1 = user_video_path  + "/" + sorted_files[files]
        video_path_temp_2 = user_video_path  + "/" + sorted_files[files+1]
        video_path_temp_3 = user_video_path  + "/" + sorted_files[files+2]

        duration = video_duration(video_path_temp_1, video_path_temp_2, video_path_temp_3, frame_rate)

        s = sorted_files[files].split(".")[0]
        cleaned = s.replace('_NoAudio_', '_') 

        user_data_filted = user_data[user_data['Filename'] == cleaned]
        start_time = 0
        end_time = 0

        # class_array = np.zeros(duration*fps, dtype=int)
        class_array = np.full(duration*fps, -1)
        # print("Duration (seconds): ", duration, ", Array_Length: ", len(class_array))
        for i in range(len(user_data_filted)):
            start_time = user_data_filted['Start Time'].iloc[i]
            end_time = user_data_filted['End Time'].iloc[i]
            action_class = int(user_data_filted['Label (Primary)'].iloc[i].split(" ")[-1])
            start_time = time_processor(start_time)
            end_time = time_processor(end_time)
            class_array[start_time*fps: end_time*fps] = action_class


        save_path = "./arrays_5/" + user_id + "_" + sorted_files[files].split(".")[0][-1]
        np.save(save_path, class_array)
        processed_list.append([sorted_files[files], sorted_files[files+1], sorted_files[files+2], save_path])
        # processed_list.append(sorted_files[files], sorted_files[files+1], sorted_files[files+2], len(class_array))
        test_counter += 1

print("Finished_processing", int(test_counter/2))
np.savetxt("class_arrays_8.csv", processed_list, delimiter =", ", fmt ='% s')
