import pandas as pd
import os

input_directory = '/media/viplab/DATADRIVE1/driver_action_recognition/2024-data_labelsAndinstructions/labels/A1/A1_fixed'  # Replace with your directory path
output_directory = '/media/viplab/DATADRIVE1/driver_action_recognition/2024-data_labelsAndinstructions/labels/A1/A1_fixed_split'  # Replace with your desired output directory path

os.makedirs(output_directory, exist_ok=True)

for file_name in os.listdir(input_directory):
    if file_name.endswith('.csv'):
        # Full path to the current CSV file
        file_path = os.path.join(input_directory, file_name)
        
        # Load the CSV file into a DataFrame
        df = pd.read_csv(file_path)

        # Group by the 'Filename' column
        grouped = df.groupby('Filename')

        # Iterate over each group and modify the 'Filename' and save it to a separate CSV file
        for name, group in grouped:
            # Modify the filename: Add 'NoAudio_' before the number part (e.g., user_id_28557_5)
            modified_name = '_'.join(name.split('_')[:-1]) + '_NoAudio_' + name.split('_')[-1]

            # Clean the filename if there are any unwanted characters
            safe_name = modified_name.replace('/', '_').replace(' ', '_')

            # Extract the directory part from the modified filename
            # print('_'.join(name.split('_')[-4:]))
            dir_name = '_'.join(name.split('_')[-4:])  # Extract 'user_id_28557'
            
            # Create the directory for the specific user_id_number (e.g., user_id_28557_5)
            user_directory = os.path.join(output_directory, dir_name)
            os.makedirs(user_directory, exist_ok=True)

            # Create the output file path within the user's specific directory
            output_file_path = os.path.join(user_directory, f'{safe_name}.csv')
            
            # Save the group to a new CSV file
            group.to_csv(output_file_path, index=False)
            print(f"Saved {output_file_path}")
