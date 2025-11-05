import pandas as pd
import os
import re


labels_path = "/media/viplab/DATADRIVE1/driver_action_recognition/2024-data_labelsAndinstructions/labels/A1/A1_fixed/"
labels_dir = os.listdir(labels_path)

for i in range(len(labels_dir)):
    labels_full_path = labels_path + labels_dir[i]
    print(labels_full_path)
    
    df = pd.read_csv(labels_full_path)
    
    # print(df['Filename'])
    
    view = df['Camera View']
    file_name = df['Filename']
    
    
    file_name = list(file_name)
    camera_view = list(view)

    for j in range(len(file_name)):

        if(re.search("dash", camera_view[j], re.IGNORECASE)):
            org_filename = df["Filename"][j]
            org_filename = org_filename.split("_")
            org_filename[0] = "Dashboard"   
            modified_filename = "_".join(org_filename)
            df.loc[j, "Filename"] = modified_filename
            # print(df['Filename'][j])
        
        if(re.search("rear", camera_view[j], re.IGNORECASE)):
            org_filename = df["Filename"][j]
            org_filename = org_filename.split("_")
            org_filename[0] = "Rear"   
            modified_filename = "_".join(org_filename)
            df.loc[j, "Filename"] = modified_filename
            # print(df['Filename'][j])
            
        if(re.search("right", camera_view[j], re.IGNORECASE)):
            org_filename = df["Filename"][j]
            org_filename = org_filename.split("_")
            org_filename[0] = "Right"
            modified_filename = "_".join(org_filename)
            df.loc[j, "Filename"] = modified_filename
            # print(df['Filename'][j])
    
    df.to_csv(labels_full_path, index= False)
        

