import os
import json
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Define paths to audio and label files
audio_folder_path = '../mic_cleaned'
label_folder_path = '../respiratory'

label_map = {'ObstructiveApnea': 0, 'Hypopnea': 1, 'CentralApnea': 2, 'MixedApnea': 3}

# Load audio files and extract spectrograms in smaller chunks
all_labels, all_spectrograms = [], []
for filename in sorted(os.listdir(audio_folder_path)):
    if filename.endswith('.wav'):
        print(f'[INFO] Processing {filename}')

        # Load label files
        patient_id = filename.split('.')[0]
        with open(os.path.join(label_folder_path, f'{patient_id}.json')) as f:
            annotations = json.load(f)

        labels, intervals = [], []
        for annotation in annotations:
            if annotation['type'] in label_map.keys():
                intervals.append((annotation['start'], annotation['start'] + annotation['duration']))
                labels.append(annotation['type'])

        # Downsample the audio to 16 kHz in smaller chunks
        audio, sr = librosa.load(os.path.join(audio_folder_path, filename), sr=16000)
        chunk_size = int(10 * sr)  # 10 seconds per chunk
        chunk_labels, chunk_spectrograms = [], []

        for label, (interval_start, interval_end) in zip(labels, intervals):
            chunk_start, chunk_end = int(interval_start * sr), int(interval_end * sr)
            for i in range(chunk_start, chunk_end, chunk_size):
                chunk = audio[i:i+chunk_size]
                # Truncate
                if len(chunk) != chunk_size:
                    break

                # Compute the spectrogram for the resampled chunk
                spectrogram = librosa.feature.melspectrogram(y=chunk, sr=16000)
                spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
                # print(f'[INFO] Spectrogram shape: {spectrogram.shape}')
                # print(f'[INFO] Max: {spectrogram.max():.4f}, Min: {spectrogram.min():.4f}')
                chunk_spectrograms.append(spectrogram)
                chunk_labels.append(label_map[label])
        
        all_spectrograms.append(np.array(chunk_spectrograms))
        all_labels.append(np.array(chunk_labels))
        
# Convert spectrograms and labels to tensors
all_spectrograms = np.concatenate(all_spectrograms, axis=0)
all_labels = np.concatenate(all_labels, axis=0)
print(f'[INFO] All spectrograms shape: {all_spectrograms.shape}')
print(f'[INFO] All labels shape: {all_labels.shape}')

# Split data into training and validation sets
train_spectrograms, val_spectrograms, train_labels, val_labels = train_test_split(all_spectrograms, all_labels, test_size=0.2, random_state=42)
np.savez_compressed('train.npz', spectrograms=train_spectrograms, labels=train_labels)
np.savez_compressed('val.npz', spectrograms=val_spectrograms, labels=val_labels)

unique, counts = np.unique(train_labels, return_counts=True)
print(f'[INFO] Before resampling: {dict(zip(unique, counts))}')

# Apply SMOTE for resampling on training data
flattened_spectrograms = train_spectrograms.reshape(train_spectrograms.shape[0], -1)
smote = SMOTE(sampling_strategy='auto', random_state=42,)
flattened_spectrograms, train_labels = smote.fit_resample(flattened_spectrograms, train_labels)
train_spectrograms = flattened_spectrograms.reshape(-1, train_spectrograms.shape[1], train_spectrograms.shape[2])

unique, counts = np.unique(train_labels, return_counts=True)
print(f'[INFO] After resampling: {dict(zip(unique, counts))}')

print(f'[INFO] Resampled train spectrograms shape: {train_spectrograms.shape}')
print(f'[INFO] Resampled train labels shape: {train_labels.shape}')
np.savez_compressed('train_resampled.npz', spectrograms=train_spectrograms, labels=train_labels)