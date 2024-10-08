#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
# to parse the rml file
"""
Created on Sat Aug 17 18:02:28 2024

@author: ybys
"""


import mne
import numpy as np
import os
import matplotlib.pyplot as plt  # Optional, for visualization
import glob

def clean_mic_signal(fif_path):
    # Load the filtered EDF file with preload=False
    raw = mne.io.read_raw_fif(fif_path, preload=False)

    # Extract the "Mic" channel
    mic = 'Mic'
    mic_signal = raw[mic][0][0]  # Load the data into memory

    # Set an amplitude threshold to identify and remove artifacts
    amplitude_threshold = np.std(mic_signal) * 5
    cleaned_signal = np.where(np.abs(mic_signal) > amplitude_threshold, 0, mic_signal)

    # Detect missing data (e.g., complete zero segments)
    missing_data_indices = np.where(cleaned_signal == 0)[0]

    # Initialize cleaned_signal_interpolated
    cleaned_signal_interpolated = cleaned_signal.copy()

    # If missing data is significant, apply interpolation
    if len(missing_data_indices) > 0:
        cleaned_signal_interpolated[missing_data_indices] = np.interp(missing_data_indices,
                                                                     np.nonzero(cleaned_signal)[0],
                                                                     cleaned_signal[np.nonzero(cleaned_signal)])

    # Create a new Raw object with the cleaned data
    # Prepare to create a new raw data array with the same shape as original
    cleaned_data = raw.get_data()  # Get the data as a numpy array
    cleaned_data[raw.ch_names.index(mic), :] = cleaned_signal_interpolated  # Update the Mic channel data

    # Create a new Info object to hold the metadata
    cleaned_info = raw.info.copy()  # Copy the info from the original raw object

    # Create a new Raw object with the cleaned data
    cleaned_raw = mne.io.RawArray(cleaned_data, cleaned_info)

    # Save the cleaned data to a new FIF file
    cleaned_fif_path = fif_path.replace('.fif', '_cleaned.fif')
    cleaned_raw.save(cleaned_fif_path, overwrite=True)

    print(f"Cleaned signal saved to: {cleaned_fif_path}")

    # Optional: Plot the cleaned signal
    plt.figure(figsize=(12, 4))
    plt.plot(cleaned_signal_interpolated)
    plt.title(f'Cleaned Mic Signal: {cleaned_fif_path}')
    plt.xlabel('Time (Samples)')
    plt.ylabel('Amplitude')
    plt.show()

# Define the base directory where patient folders are located
base_directory = "/Users/ybys/Desktop/TP/PSG_Audio/APNEA_EDF"

# Note: I completely missed load more option before,
#       here the new code is slightly modified
'''
# List of patient numbers to process
patient_id = ["0995", "0999", "1000", "1006", "1008", "1010", "1014", "1016"]

# Process each patient's folder
for patient_number in patient_id:
    patient_folder = os.path.join(base_directory, patient_number)
    # Loop through all .fif files in the patient's folder
    for file_name in os.listdir(patient_folder):
        if file_name.endswith('.fif'):
            fif_path = os.path.join(patient_folder, file_name)
            clean_mic_signal(fif_path)
'''

# Use glob to find all FIF files recursively in subdirectories
fif_files = glob.glob(os.path.join(base_directory, '**', '*.fif'), recursive=True)

# Loop through each FIF file and clean the Mic channel
for fif_file in fif_files:
    clean_mic_signal(fif_file)
    

    

