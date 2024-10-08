#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# to filter and extract dataset
"""
Created on Wed Aug 14 18:15:40 2024

@author: ybys
"""
"""



raw = mne.io.read_raw_edf(input_file_path, preload=False)

# Select the relevant channels
channels_to_keep = ['Tracheal', 'Mic']
raw = raw.pick_channels(channels_to_keep)

# Save the extracted channels to a new EDF file
raw.save(output_file_path, overwrite=True)

print(f"Selected channels saved to {output_file_path}")


"""
import mne
import os
import glob

def extract_and_save_mic_channel(input_file_path, output_file_path):
    """
    Extract the 'Mic' channel from an EDF file and save it to a new FIF file.
    
    Parameters:
    - input_file_path: str, path to the input EDF file
    - output_file_path: str, path to the output FIF file
    """
    # Just load the EDF file with preload=False to minimize memory usage
    raw = mne.io.read_raw_edf(input_file_path, preload=False)
    
    # Select 'Mic' channel only
    raw = raw.pick_channels(['Mic'])
    
    # Save to a new FIF file
    raw.save(output_file_path, overwrite=True)
    
    print(f"Selected 'Mic' channel saved to {output_file_path}")  


# Note: I completely missed load more option before,
#       here the new code is slightly modified
'''
def process_patient_data(patient_id, base_directory):
    
    # Get all EDF files in each folder
    edf_files = glob.glob(os.path.join(base_directory, patient_id, '*.edf'))
    
    # Get all EDF files in the base directory (and subdirectories)
    edf_files = glob.glob(os.path.join(base_directory, '**', '*.edf'), recursive=True)
    
    
    for edf_file in edf_files:
        # Output file name: replace edf with fif
        file_name = os.path.basename(edf_file).replace('.edf', '.fif')
        output_file_path = os.path.join(base_directory, patient_id, file_name)
        
        """
        # Check if the FIF file already exists
        if not os.path.exists(output_file_path):
          # Extract and save the 'Mic' channel
            extract_and_save_mic_channel(edf_file, output_file_path)
        else:
            print(f"File {output_file_path} already exists, skipping.")
        """
        
        
        # Extract and save the 'Mic' channel
        extract_and_save_mic_channel(edf_file, output_file_path)
        

base_directory = '/Users/ybys/Desktop/TP/PSG_Audio/APNEA_EDF'

# List of patient IDs
patient_ids = ['0995', '0999', '1000', '1006', '1008', '1010', '1014', '1016']

# Process data for each patient
for patient_id in patient_ids:
    process_patient_data(patient_id, base_directory)
'''
    
def process_patient_data(base_directory):
    """
    Process all EDF files in the base directory, extracting the 'Mic' channel.
    
    Parameters:
    - base_directory: str, path to the base directory containing the patient folders
    """
    # Get all EDF files in the base directory (and subdirectories)
    edf_files = glob.glob(os.path.join(base_directory, '**', '*.edf'), recursive=True)
    
    for edf_file in edf_files:
        # Extract patient ID from the filename, the ID is in positions 6-10
        # For example, file name '00001016-100507[001].edf'
        file_name = os.path.basename(edf_file)
        patient_id = file_name[5:9]

         # Output file name: replace edf with fif
        output_file_path = os.path.join(base_directory, patient_id, file_name.replace('.edf', '.fif'))
        
        # Check if the output directory exists, create if not
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        
        # Extract and save the 'Mic' channel
        extract_and_save_mic_channel(edf_file, output_file_path)

# Base directory where all the patient data is stored
base_directory = '/Users/ybys/Desktop/TP/PSG_Audio/APNEA_EDF'

# Process data for all patients
process_patient_data(base_directory)