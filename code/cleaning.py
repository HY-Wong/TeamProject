import mne
import numpy as np
import os
import glob
import pandas as pd  # For statistics summary

def summarize_statistics(channel_data, step_name, acq_id):
    # Calculate basic statistics for the channel
    stats = {
        'Acq_id': acq_id,
        'Step': step_name,
        'Mean Amplitude': np.mean(channel_data),
        'STD': np.std(channel_data),
        'Min Amplitude': np.min(channel_data),
        'Max Amplitude': np.max(channel_data),
        'Non-Zero Count': np.count_nonzero(channel_data),
        'Zero Count': np.sum(channel_data == 0),
        'Missing Data Percentage': np.sum(channel_data == 0) / len(channel_data) * 100
    }
    return stats

def clean_data(fif_path):
    
    # Extract 'Acq_id' from the file path
    acq_id = os.path.basename(fif_path).replace('.fif', '')

    # Skip processing if already cleaned
    if fif_path.endswith('_cleaned.fif'):
        print(f"File {fif_path} is already cleaned. Skipping...")
        return None

    # Load the filtered FIF file with preload=False
    raw = mne.io.read_raw_fif(fif_path, preload=False)

    # Define channels to clean
    channels_to_clean = ['Snore', 'Tracheal', 'Mic']
    
    # Initialize statistics storage
    summary_stats = []

    # Initialize a list to hold cleaned data
    cleaned_data = raw.get_data()  # Get the original data as a numpy array

    for channel in channels_to_clean:
        mic_signal = cleaned_data[raw.ch_names.index(channel), :]  # Load the channel data
        
        
        original_stats = summarize_statistics(mic_signal, f'Original {channel}', acq_id)
        summary_stats.append(original_stats)

        # 1. Set an amplitude threshold to identify and remove artifacts
        amplitude_threshold = np.std(mic_signal) * 5
        cleaned_signal = np.where(np.abs(mic_signal) > amplitude_threshold, 0, mic_signal)

        #    Artifact Removal Stats
        artifact_removed_stats = summarize_statistics(cleaned_signal, f'Artifact Removed {channel}', acq_id)
        summary_stats.append(artifact_removed_stats)

        # 2. Detect missing data (e.g., complete zero segments)
        missing_data_indices = np.where(cleaned_signal == 0)[0]

        #    Initialize cleaned_signal_interpolated
        cleaned_signal_interpolated = cleaned_signal.copy()

        #    Only interpolate if there are valid non-zero points in the signal
        non_zero_indices = np.nonzero(cleaned_signal)[0]
        if len(missing_data_indices) > 0 and len(non_zero_indices) > 0:
            cleaned_signal_interpolated[missing_data_indices] = np.interp(
                missing_data_indices, non_zero_indices, cleaned_signal[non_zero_indices])

        #    Post Missing Data Interpolation Stats
        interpolated_stats = summarize_statistics(cleaned_signal_interpolated, f'Interpolated {channel}', acq_id)
        summary_stats.append(interpolated_stats)

        # 3. Normalize the signal to the range [-1, 1]
        cleaned_signal_normalized = cleaned_signal_interpolated / np.max(np.abs(cleaned_signal_interpolated))
        normalization_stats = summarize_statistics(cleaned_signal_normalized, f'Normalized {channel}', acq_id)
        summary_stats.append(normalization_stats)

        #    Update the cleaned data array
        cleaned_data[raw.ch_names.index(channel), :] = cleaned_signal_normalized

    # Create a new Info object to hold the metadata
    cleaned_info = raw.info.copy()  # Copy the info from the original raw object

    # Create a new Raw object with the cleaned data
    cleaned_raw = mne.io.RawArray(cleaned_data, cleaned_info)

    # Save the cleaned data to the new FIF file
    cleaned_fif_path = fif_path.replace('.fif', '_cleaned.fif')
    cleaned_raw.save(cleaned_fif_path, overwrite=True)

    print(f"Cleaned signals saved to: {cleaned_fif_path}")

    return summary_stats

# Define the base directory where patient folders are located
base_directory = "/Users/ybys/Desktop/TP/PSG_Audio/APNEA_EDF"

# Use glob to find all FIF files recursively in subdirectories
fif_files = glob.glob(os.path.join(base_directory, '**', '*.fif'), recursive=True)

# Initialize a list to hold the summary of statistics for all files
all_files_summary = []

# Loop through each FIF file and clean the specified channels
for fif_file in fif_files:
    stats = clean_data(fif_file)
    if stats is not None:
        all_files_summary.extend(stats)

# Create a DataFrame to summarize the statistics for all files and channels
summary_df = pd.DataFrame(all_files_summary)
summary_csv_path = os.path.join(base_directory, "cleaning_summary_statistics.csv")
summary_df.to_csv(summary_csv_path, index=False)

print(f"Summary statistics saved to: {summary_csv_path}")
