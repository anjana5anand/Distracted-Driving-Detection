import os
import pandas as pd
import subprocess
import datetime
import re

labels_dir = '/media/viplab/DATADRIVE1/driver_action_recognition/2024-data_labelsAndinstructions/labels/A1/A1_fixed_split'
data_dir = '/media/viplab/DATADRIVE1/driver_action_recognition/crop_video/train'
output_dir = '/media/viplab/DATADRIVE1/driver_action_recognition/split_video/train'
os.makedirs(output_dir, exist_ok=True)

# Function to convert hh:mm:ss or hh:mm:ss.ms to seconds
def time_str_to_seconds(time_str):
    """Convert hh:mm:ss or hh:mm:ss.ms to seconds as float"""
    try:
        t = datetime.datetime.strptime(time_str, "%H:%M:%S.%f")
    except ValueError:
        t = datetime.datetime.strptime(time_str, "%H:%M:%S")
    return int(t.hour * 3600 + t.minute * 60 + t.second) #+ t.microsecond / 1e6

def seconds_to_hms(seconds):
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02}:{m:02}:{s:02}"

# Function to get video duration using ffprobe
def get_video_duration(video_path):
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries',
             'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', video_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT)
        return float(result.stdout.decode().strip())
    except:
        return None
    
# Process each user and video
for user in os.listdir(data_dir):
    user_video_dir = os.path.join(data_dir, user)
    # print(user_video_dir)
    if not os.path.isdir(user_video_dir):
        print(user_video_dir)
        continue

    for video_file in os.listdir(user_video_dir):
        if not video_file.endswith('.MP4'):
            continue

        video_path = os.path.join(user_video_dir, video_file)
        duration = get_video_duration(video_path)
        # print(video_path, duration)
        if duration is None:
            print(f"❌ Failed to get duration for {video_path}")
            continue

        # Build path to corresponding CSV
        csv_name = video_file.replace('.MP4', '.csv')
        label_csv_path = os.path.join(labels_dir, user, csv_name)

        if not os.path.exists(label_csv_path):
            print(f"⚠️ CSV not found for {video_file}, skipping.")
            continue

        # Load the CSV with start_time, end_time, and labels
        df = pd.read_csv(label_csv_path)

        for idx, row in df.iterrows():
            # Convert time strings to seconds
            start = int(time_str_to_seconds(row['Start Time']))
            end = int(time_str_to_seconds(row['End Time']))
            label = row['Label (Primary)'].split(' ')[-1]

            # Skip invalid clip ranges
            if start >= duration:
                print(f"⏭️ Clip {idx} of {video_file} starts after video ends — skipping. {start} {end} {duration}")
                continue
            if end > duration:
                print(f"⚠️ Clip {idx} of {video_file} exceeds duration — trimming end time from {end} to {duration}. {start} {end} {duration}")
                end = duration
            if end <= start: #end - start <= 1: 
                print(f"❌ Clip {idx} of {video_file} has invalid start/end — skipping. {start} {end} {duration}")
                continue

            # Create output folder for this label if it doesn't exist
            if re.search('Dash', video_path):
                label_dir = os.path.join(output_dir, "Dash")
            elif re.search('Rear', video_path):
                label_dir = os.path.join(output_dir, "Rear")
            else:
                label_dir = os.path.join(output_dir, "Side")

            label_dir = os.path.join(label_dir, label)
            os.makedirs(label_dir, exist_ok=True)
            # print(label_dir)
            # Output clip path
            end = int(end)
            if label in [1, 7, 9, 10, 13]:
                skip = 2
            else:
                skip = 3
            for i in range(start, end, skip):
                b = seconds_to_hms(i)
                output_name = f"{video_file.replace('.MP4', '')}_{idx}_{i}.MP4"
                output_path = os.path.join(label_dir, output_name)
                cmd = [
                'ffmpeg',
                '-i', video_path,
                '-ss', b,
                '-t', str(3),
                '-c:v', 'libx264',
                '-an',
                output_path,
                '-y'  # overwrite if exists
                ]
                # print(b)
                subprocess.run(cmd)


# import os
# import pandas as pd
# import subprocess
# import datetime

# output_dir = '/media/viplab/DATADRIVE1/driver_action_recognition/split_videos'
# labels_dir = '/media/viplab/DATADRIVE1/driver_action_recognition/2024-data_labelsAndinstructions/labels/A1/A1_fixed_split'
# data_dir = '/media/viplab/DATADRIVE1/driver_action_recognition/crop_videos/train'

# os.makedirs(output_dir, exist_ok=True)

# # Function to convert hh:mm:ss or hh:mm:ss.ms to seconds
# def time_str_to_seconds(time_str):
#     """Convert hh:mm:ss or hh:mm:ss.ms to seconds as float"""
#     try:
#         t = datetime.datetime.strptime(time_str, "%H:%M:%S.%f")
#     except ValueError:
#         t = datetime.datetime.strptime(time_str, "%H:%M:%S")
#     return t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6

# # Function to get video duration using ffprobe
# def get_video_duration(video_path):
#     try:
#         result = subprocess.run(
#             ['ffprobe', '-v', 'error', '-show_entries',
#              'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', video_path],
#             stdout=subprocess.PIPE,
#             stderr=subprocess.STDOUT)
#         return float(result.stdout.decode().strip())
#     except:
#         return None
    
# # Process each user and video
# for user in os.listdir(data_dir):
#     user_video_dir = os.path.join(data_dir, user)
#     # print(user_video_dir)
#     if not os.path.isdir(user_video_dir):
#         print(user_video_dir)
#         continue

#     for video_file in os.listdir(user_video_dir):
#         if not video_file.endswith('.MP4'):
#             continue

#         video_path = os.path.join(user_video_dir, video_file)
#         duration = get_video_duration(video_path)
#         # print(video_path, duration)
#         if duration is None:
#             print(f"❌ Failed to get duration for {video_path}")
#             continue

#         # Build path to corresponding CSV
#         csv_name = video_file.replace('.MP4', '.csv')
#         label_csv_path = os.path.join(labels_dir, user, csv_name)

#         if not os.path.exists(label_csv_path):
#             print(f"⚠️ CSV not found for {video_file}, skipping.")
#             continue

#         # Load the CSV with start_time, end_time, and labels
#         df = pd.read_csv(label_csv_path)

#         for idx, row in df.iterrows():
#             # Convert time strings to seconds
#             start = time_str_to_seconds(row['Start Time'])
#             end = time_str_to_seconds(row['End Time'])
#             label = row['Label (Primary)'].split(' ')[-1]

#             # Skip invalid clip ranges
#             if start >= duration:
#                 print(f"⏭️ Clip {idx} of {video_file} starts after video ends — skipping. {start} {end} {duration}")
#                 continue
#             if end > duration:
#                 print(f"⚠️ Clip {idx} of {video_file} exceeds duration — trimming end time from {end} to {duration}. {start} {end} {duration}")
#                 end = duration
#             if end <= start:
#                 print(f"❌ Clip {idx} of {video_file} has invalid start/end — skipping. {start} {end} {duration}")
#                 continue

#             # Create output folder for this label if it doesn't exist
#             label_dir = os.path.join(output_dir, label)
#             os.makedirs(label_dir, exist_ok=True)

#             # Output clip path
#             # for i in range(start, end, 3):

#             output_name = f"{user}_{video_file.replace('.MP4', '')}_{idx}.MP4"
#             output_path = os.path.join(label_dir, output_name)
#             print(output_path)
#             # # Run ffmpeg to extract the clip
#             # print(str(start))
#             cmd = [
#                 'ffmpeg',
#                 '-i', video_path,
#                 '-ss', str(row['Start Time']),
#                 '-to', str(row['End Time']),
#                 '-c:v', 'libx264',
#                 '-an',
#                 output_path,
#                 '-y'  # overwrite if exists
#             ]
#             subprocess.run(cmd)
