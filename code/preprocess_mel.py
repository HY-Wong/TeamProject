import os
import json
import torch
import torchaudio
import torchaudio.transforms as T

from multiprocessing import Pool
from functools import partial


def process_file(json_file, json_dir, audio_dir, output_dir, sr=16000):
    if not json_file.endswith(".json"):
        return
    
    print(f'[INFO] processing {json_file}')
    mel_transform = T.MelSpectrogram(sample_rate=sr, n_fft=1024, hop_length=512, n_mels=128)
    db_transform = T.AmplitudeToDB()
    
    # Load annotations
    json_path = os.path.join(json_dir, json_file)
    with open(json_path, "r") as f:
        annotations = json.load(f)

    # Load corresponding audio
    audio_file = json_file.replace(".json", ".wav")
    audio_path = os.path.join(audio_dir, audio_file)
    waveform, sample_rate = torchaudio.load(audio_path)

    # Resample if necessary
    if sample_rate != sr:
        resampler = T.Resample(orig_freq=sample_rate, new_freq=sr)
        waveform = resampler(waveform)
    print(waveform.shape)

    segments_data = []
    for annotation in annotations:
        start = annotation["start"]
        end = annotation["end"]
        label = annotation["label"]

        # Convert time to sample indices
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        segment = waveform[:, start_sample:end_sample]

        # Extract mel spectrogram
        segment = mel_transform(segment)
        segment = db_transform(segment)
        output_file = os.path.join(output_dir, f'{json_file.replace(".json", "")}_{start}.pt')
        torch.save({"mel": segment, "label": label}, output_file)
        segments_data.append({"mel": segment, "label": label})


def preprocess_and_store(json_dir, audio_dir, output_dir, sr=16000):
    os.makedirs(output_dir, exist_ok=True)
    json_files = sorted(os.listdir(json_dir))
    
    # use multiprocessing pool to parallelize the processing
    pool = Pool(processes=5)
    worker = partial(
        process_file,
        json_dir=json_dir, 
        audio_dir=audio_dir, 
        output_dir=output_dir,
        sr=sr
    )
    pool.map(worker, json_files)
    pool.close()
    pool.join()


preprocess_and_store(
    json_dir="../apnea/train",
    audio_dir="../mic_cleaned",
    output_dir="../preprocessed_apnea_mel/train"
)
preprocess_and_store(
    json_dir="../apnea/val",
    audio_dir="../mic_cleaned",
    output_dir="../preprocessed_apnea_mel/val"
)