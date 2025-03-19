import numpy as np
import os
import glob
import pandas as pd
from scipy.io import wavfile  # For reading .wav files

def summarize_statistics(channel_data, step_name, acq_id, channel):
    if isinstance(channel_data, np.ndarray) and len(channel_data) > 0:
        stats = {
            'Acq_id': acq_id,
            'Step': step_name,
            'Channel': channel,
            'Mean Amplitude': np.mean(channel_data),
            'STD': np.std(channel_data),
            'Min Amplitude': np.min(channel_data),
            'Max Amplitude': np.max(channel_data),
            'Non-Zero Count': np.count_nonzero(channel_data),
            'Zero Count': np.sum(channel_data == 0),
            'Missing Data Percentage': np.sum(channel_data == 0) / len(channel_data) * 100
        }
        return stats
    else:
        print(f"Warning: Channel data for {step_name} is not valid. Data: {channel_data}")
        return None

def clean_data(wav_file):
    # Extract 'Acq_id' and channel from the filename
    filename = os.path.basename(wav_file)
    acq_id = filename.split('_')[0]
    channel = 'Snore'

    # Load the .wav file for the Snore channel
    sampling_rate, snore_signal = wavfile.read(wav_file)

    original_stats = summarize_statistics(snore_signal, 'Original', acq_id, channel)
    if original_stats is None:
        return None  # Skip if original stats are invalid

    # 1. Set an amplitude threshold to identify and remove artifacts
    amplitude_threshold = np.std(snore_signal) * 5
    cleaned_signal = np.where(np.abs(snore_signal) > amplitude_threshold, 0, snore_signal)

    # Artifact Removal Stats
    artifact_removed_stats = summarize_statistics(cleaned_signal, 'Artifact Removed', acq_id, channel)
    if artifact_removed_stats is None:
        return None  # Skip if artifact stats are invalid

    # 2. Detect missing data (e.g., complete zero segments)
    missing_data_indices = np.where(cleaned_signal == 0)[0]

    # Interpolate over missing data
    cleaned_signal_interpolated = cleaned_signal.copy()
    non_zero_indices = np.nonzero(cleaned_signal)[0]
    if len(missing_data_indices) > 0 and len(non_zero_indices) > 0:
        cleaned_signal_interpolated[missing_data_indices] = np.interp(
            missing_data_indices, non_zero_indices, cleaned_signal[non_zero_indices])

    # Post Missing Data Interpolation Stats
    interpolated_stats = summarize_statistics(cleaned_signal_interpolated, 'Interpolated', acq_id, channel)
    if interpolated_stats is None:
        return None  # Skip if interpolation stats are invalid

    # 3. Normalize the signal to the range [-1, 1]
    cleaned_signal_normalized = cleaned_signal_interpolated / np.max(np.abs(cleaned_signal_interpolated))
    normalization_stats = summarize_statistics(cleaned_signal_normalized, 'Normalized', acq_id, channel)
    if normalization_stats is None:
        return None  # Skip if normalization stats are invalid

    # Save the cleaned signal back to a new .wav file
    os.makedirs(output_directory, exist_ok=True)
    cleaned_wav_path = os.path.join(output_directory, filename)
    wavfile.write(cleaned_wav_path, sampling_rate, (cleaned_signal_normalized * 32767).astype(np.int16))

    print(f"Cleaned signal saved to: {cleaned_wav_path}")

    return original_stats, artifact_removed_stats, interpolated_stats, normalization_stats

# Define the base directory where patient folders are located
base_directory = "../data"
output_directory = "../mic_cleaned"

# Use glob to find all 'snore' .wav files recursively in subdirectories
wav_files = glob.glob(os.path.join(base_directory, '**', '*_snore.wav'), recursive=True)

# Initialize a list to hold the summary of statistics for all files
all_files_summary = []

# Loop through each .wav file and clean the 'Snore' channel
for wav_file in wav_files:
    stats = clean_data(wav_file)
    if stats is not None:
        all_files_summary.extend(stats)

# Create a DataFrame to summarize the statistics for all files
summary_df = pd.DataFrame(all_files_summary)
summary_csv_path = os.path.join('../data', "cleaning_summary_statisticsSnore.csv")
summary_df.to_csv(summary_csv_path, index=False)

print(f"Summary statistics saved to: {summary_csv_path}")
