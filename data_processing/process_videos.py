import os
import shutil
from glob import glob
import re
source_root = "/media/viplab/DATADRIVE1/driver_action_recognition/raw_videos/A1/"
des_root = "/media/viplab/DATADRIVE1/driver_action_recognition/raw_videos/A1_changed/"
pattern = re.compile(r"(dash|rear|right).*?(user_id_\d+).*?_(\d+)\.mp4", re.IGNORECASE)

for user_folder in os.listdir(source_root):
    user_path = os.path.join(source_root, user_folder)
    if not os.path.isdir(user_path):
        continue

    for file_name in os.listdir(user_path):
        match = pattern.match(file_name)
        if match:
            view, user_id, video_number = match.groups()
            new_folder_name = f"{user_id}_{video_number}"
            new_folder_path = os.path.join(des_root, new_folder_name)
            os.makedirs(new_folder_path, exist_ok=True)

            # Move the file into its new folder
            src_file = os.path.join(user_path, file_name)
            dst_file = os.path.join(new_folder_path, file_name)
            shutil.move(src_file, dst_file)
    # if not os.listdir(user_path):
    #     os.rmdir(user_path)
