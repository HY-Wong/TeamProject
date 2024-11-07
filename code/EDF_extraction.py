import os
import numpy as np
import pyedflib

# Define your data directory
data_dir = '/Users/ybys/Desktop/TP/PSG_Audio/APNEA_EDF'

# Dictionary to store concatenated data for 'Mic' and 'Tracheal' channels for each patient
concatenated_data = {}

# Traverse all patient folders and EDF files
for patient_folder in os.listdir(data_dir):
    patient_path = os.path.join(data_dir, patient_folder)
    
    if not os.path.isdir(patient_path):
        continue
    
    concatenated_mic_signal = []
    concatenated_tracheal_signal = []
    sample_rate_mic = 0
    sample_rate_tracheal = 0

    for file_name in sorted(os.listdir(patient_path)):
        if file_name.endswith('.edf'):
            edf_path = os.path.join(patient_path, file_name)
            edf_file = pyedflib.EdfReader(edf_path)

            signal_labels = edf_file.getSignalLabels()

            # Check for 'Mic' and 'Tracheal' channels
            if 'Mic' in signal_labels:
                mic_index = signal_labels.index('Mic')
                mic_data = edf_file.readSignal(mic_index)
                concatenated_mic_signal.append(mic_data)
                sample_rate_mic = edf_file.getSampleFrequency(mic_index)
            else:
                print(f"'Mic' channel not found in {edf_path}. Skipping.")

            if 'Tracheal' in signal_labels:
                tracheal_index = signal_labels.index('Tracheal')
                tracheal_data = edf_file.readSignal(tracheal_index)
                concatenated_tracheal_signal.append(tracheal_data)
                sample_rate_tracheal = edf_file.getSampleFrequency(tracheal_index)
            else:
                print(f"'Tracheal' channel not found in {edf_path}. Skipping.")

            edf_file.close()

    # Concatenate all segments and store if both channels are found
    if concatenated_mic_signal and concatenated_tracheal_signal:
        full_mic_signal = np.concatenate(concatenated_mic_signal)
        full_tracheal_signal = np.concatenate(concatenated_tracheal_signal)

        # Store both channels in a dictionary and save in a single .npy file
        concatenated_data = {
            'Mic': {'data': full_mic_signal, 'sample_rate': sample_rate_mic},
            'Tracheal': {'data': full_tracheal_signal, 'sample_rate': sample_rate_tracheal}
        }
        
        # Save the concatenated data dictionary to a single .npy file
        output_file_path = os.path.join(data_dir, f"{patient_folder}_mic_tracheal.npy")
        np.save(output_file_path, concatenated_data)
        
        print(f'Saved concatenated Mic and Tracheal data for {patient_folder} to {output_file_path}')
