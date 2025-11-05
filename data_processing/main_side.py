import pandas as pd
import os
import re
import shutil

import cv2
from imutils.video import VideoStream

import time

from utils.preprocess import *
# from utils.detectron import *


# labels_path = "/mnt/d/University Stuff/OneDrive - SSN Trust/7th Semester/Capstone Project/Data/AI City Full Data/Labels/A1"
# labels_path = "/media/viplab/DATADRIVE1/driver_action_recognition/2024-data_labelsAndinstructions/labels/A1/A1_fixed"
labels_path = "/media/viplab/DATADRIVE1/driver_action_recognition/2024-data_labelsAndinstructions/labels/A1/A1_fixed (Copy)"
# videos_path = "/mnt/d/University Stuff/OneDrive - SSN Trust/7th Semester/Capstone Project/Data/AI City Full Data/A1"
# videos_path = "/mnt/d/Capstone_Data/Raw_Footage/A1"
videos_path = "/media/viplab/DATADRIVE1/driver_action_recognition/raw_videos/A1"
view_filter = "side"
# saved_output = "/mnt/d/Capstone_Data/Cut_Video_1/"
# saved_output = "mnt/C/Users/mohit/Desktop/Capstone_Data"
# saved_output = "/media/viplab/DATADRIVE1/driver_action_recognition/mobilenet_approach/cut_frames_side_new/"
# saved_output = "/media/viplab/DATADRIVE1/driver_action_recognition/mobilenet_approach/test_1/"
saved_output = f"/media/viplab/Storage1/driver_action_recognition/crop_new/cut_frames_{view_filter}/"
missing_videos = []
frame_rate = 30
frame_shift = 1
frame_skip = 6
crop_size = (512, 512)
class_names = ['class_0', 'class_1', 'class_2', 'class_3', 'class_4', 'class_5', 'class_6', 'class_7', 'class_8', 'class_9', 'class_10', 'class_11', 'class_12', 'class_13', 'class_14', 'class_15']
save_type = "image"
# save_type = "video"
skipped_videos = []


#with open("dash_missing_videos.txt", 'r') as f:
#        c = f.read()
# print("Missing File:", c)

#with open("dash_skipped_videos.txt", 'r') as f:
#    c = f.read()
# print("Skipped File:", c)





def main():
    user_labels = os.listdir(labels_path)

    replace_directory(saved_output)

    user_labels = os.listdir(labels_path)
    # user_labels_length = len(user_labels)

    for user in user_labels:
        print(user)
        user_id = user[:-4]
        user_data_path = labels_path + "/" + user
        user_video_path = videos_path + "/" + user[:-4]
        user_data = pd.read_csv(user_data_path)

        print(os.path.isdir(user_video_path))
        print(user_video_path)
        if(os.path.isdir(user_video_path)):
            user_video_files = os.listdir(user_video_path)
            for user_video in user_video_files:
                if(re.search(view_filter, user_video, re.IGNORECASE) == None):
                    skipped_videos.append(user_video)
                    continue

                #String Preprocessing

                user_video_process = user_video[:-4]
                print("Hi", user_video_process)
                user_video_process = re.sub("_NoAudio_", "_", user_video_process)
                print(user_video_process)

                user_video_path_full = user_video_path + "/"+ user_video
                print(user_video_path_full)
                user_data_filtered = user_data[user_data['Filename'] == user_video_process]
                if user_data_filtered.empty:
                    user_video_process = user_video_process.replace("_","", 1)
                    user_data_filtered = user_data[user_data['Filename'] == user_video_process]
                
                # print(user_data_filtered)
                # print(user_id)
                # print(user_data_path)    
                image_processor(user_video_path_full, user_data_filtered, user_id, saved_output, frame_rate, frame_skip, crop_size, save_type, frame_shift, view_filter)
                # print("Done: ", user)
                # print()

                
        else:
            missing_videos.append(user)
    print(missing_videos)
#    with open("dash_missing_videos.txt", 'w') as f:
#        for name in missing_videos:
#            f.write(name + '\n')
#    with open("dash_skipped_videos.txt", 'w') as f:
#        for name in skipped_videos:
#            f.write(name + '\n')
	

if __name__ == "__main__":
    main()
