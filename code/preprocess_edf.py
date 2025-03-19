import os
import re
import numpy as np
import pyedflib

from scipy.io.wavfile import write

data_dir = '../data'

# Load patient mappings from .rml file
patient_mapping = {}

# Extract patient IDs
for file_name in os.listdir(data_dir):
    if file_name.endswith('.rml'):
        patient_id = file_name.split('.')[0]
        # print(patient_id)
        patient_mapping[patient_id] = []

# Populate the mapping with corresponding .edf files
for file_name in os.listdir(data_dir):
    if file_name.endswith('.edf'):
        # Use regex to extract patient ID from the file name
        match = re.match(r'(\d{8}-\d{6})\[\d+\]\.edf', file_name)
        if match:
            patient_id = match.group(1)
            # print(patient_id)
            if patient_id in patient_mapping:
                patient_mapping[patient_id].append(os.path.join(data_dir, file_name))

channels = ['Mic', 'Snore', 'Tracheal']

for channel in channels:
    # Concatenate signals for each patient
    for patient_id, edf_pathes in patient_mapping.items():
        if patient_id == '00001339-100507':
            continue

        concatenated_signal = []
        sample_rate = 0

        edf_pathes.sort()

        for edf_path in edf_pathes:
            edf_file = pyedflib.EdfReader(edf_path)

            signal_labels = edf_file.getSignalLabels()
            signal_index = signal_labels.index(channel)
            signal_data = edf_file.readSignal(signal_index)
            # print(len(signal_data))
            concatenated_signal.append(signal_data)

            sample_rate = edf_file.getSampleFrequency(signal_index)
            physical_min = edf_file.getPhysicalMinimum(signal_index)
            physical_max = edf_file.getPhysicalMaximum(signal_index)

            edf_file.close()
        
        # Concatenate all segments
        print(f'Processing {patient_id} -  Channel {channel}')
        full_signal = np.concatenate(concatenated_signal)
        print(full_signal.shape)

        # Normalize the concatenated signal to the range of int16 for .wav
        full_signal_normalized = (full_signal - physical_min) / (physical_max - physical_min) * 2 - 1
        assert np.min(full_signal_normalized) >= -1.0
        assert np.max(full_signal_normalized) <= 1.0
        full_signal_normalized = np.clip(full_signal_normalized, -1.0, 1.0)
        full_signal_normalized = (full_signal_normalized * 32767).astype(np.int16)

        # Save as a single .wav file
        output_file_path = os.path.join(data_dir, f'{patient_id}_{channel.lower()}.wav')
        write(output_file_path, int(sample_rate), full_signal_normalized)

        print(f'Saved concatenated .wav for patient {patient_id}')
