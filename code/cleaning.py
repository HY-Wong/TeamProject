import numpy as np
import os
import glob
import pandas as pd
from scipy.io import wavfile  # For reading and writing .wav files

def summarize_statistics(channel_data, step_name, acq_id, channel_name):
    if isinstance(channel_data, np.ndarray) and len(channel_data) > 0:
        stats = {
            'Acq_id': acq_id,
            'Step': step_name,
            'Channel': channel_name,
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

def clean_channel_data(wav_file, channel_name):
    # Extract 'Acq_id' from the file path
    acq_id = os.path.basename(wav_file).split('_')[0]

    # Load the .wav file
    sample_rate, signal = wavfile.read(wav_file)
    
    # Ensure the signal is in float format for processing
    if signal.dtype != np.float32:
        signal = signal.astype(np.float32)

    original_stats = summarize_statistics(signal, 'Original', acq_id, channel_name)
    if original_stats is None:
        return None  # Skip if original stats are invalid

    # 1. Set an amplitude threshold to identify and remove artifacts
    amplitude_threshold = np.std(signal) * 5
    cleaned_signal = np.where(np.abs(signal) > amplitude_threshold, 0, signal)

    # Artifact Removal Stats
    artifact_removed_stats = summarize_statistics(cleaned_signal, 'Artifact Removed', acq_id, channel_name)
    
    if artifact_removed_stats is None:
        return None  # Skip if artifact stats are invalid

    # 2. Detect missing data (e.g., complete zero segments)
    missing_data_indices = np.where(cleaned_signal == 0)[0]

    # Initialize cleaned_signal_interpolated
    cleaned_signal_interpolated = cleaned_signal.copy()

    # Only interpolate if there are valid non-zero points in the signal
    non_zero_indices = np.nonzero(cleaned_signal)[0]
    if len(missing_data_indices) > 0 and len(non_zero_indices) > 0:
        cleaned_signal_interpolated[missing_data_indices] = np.interp(
            missing_data_indices, non_zero_indices, cleaned_signal[non_zero_indices])

    # Post Missing Data Interpolation Stats
    interpolated_stats = summarize_statistics(cleaned_signal_interpolated, 'Interpolated', acq_id, channel_name)
    
    if interpolated_stats is None:
        return None  # Skip if interpolation stats are invalid

    # 3. Normalize the signal to the range [-1, 1]
    cleaned_signal_normalized = cleaned_signal_interpolated / np.max(np.abs(cleaned_signal_interpolated))
    normalization_stats = summarize_statistics(cleaned_signal_normalized, 'Normalized', acq_id, channel_name)
    
    if normalization_stats is None:
        return None  # Skip if normalization stats are invalid

    # Save the cleaned signal back to a new .wav file
    cleaned_wav_path = wav_file.replace('.wav', '_cleaned.wav')
    wavfile.write(cleaned_wav_path, sample_rate, cleaned_signal_normalized.astype(np.float32))

    print(f"Cleaned signals saved to: {cleaned_wav_path}")

    return original_stats, artifact_removed_stats, interpolated_stats, normalization_stats

# Define the base directory where patient folders are located
base_directory = "/Users/ybys/Desktop/TP/PSG_Audio/APNEA_EDF"

# Use glob to find all .wav files recursively in subdirectories
wav_files = glob.glob(os.path.join(base_directory, '**', '*.wav'), recursive=True)

# Initialize a list to hold the summary of statistics for all files
all_files_summary = []

# Process each .wav file and summarize statistics for both channels
for wav_file in wav_files:
    # Identify if the file is mic or tracheal based on file naming
    if '_mic.wav' in wav_file:
        channel_name = 'Mic'
    elif '_tracheal.wav' in wav_file:
        channel_name = 'Tracheal'
    else:
        continue  # Skip files that don't match the expected channel names

    # Clean and summarize the data for the identified channel
    stats = clean_channel_data(wav_file, channel_name)
    if stats is not None:
        all_files_summary.extend(stats)

# Create a DataFrame to summarize the statistics for all files and channels
summary_df = pd.DataFrame(all_files_summary)
summary_csv_path = os.path.join(base_directory, "cleaning_summary_statistics.csv")
summary_df.to_csv(summary_csv_path, index=False)

print(f"Summary statistics saved to: {summary_csv_path}")
