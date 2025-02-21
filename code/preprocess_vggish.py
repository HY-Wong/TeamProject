import os
import json
import torch
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm
from torchaudio.prototype.pipelines import VGGISH


input_proc = VGGISH.get_input_processor()
model = VGGISH.get_model()


def preprocess_and_store(json_dir, audio_dir, output_dir, sr=16000, n_mels=128, n_fft=2048, hop_length=512):
    """
    Preprocesses audio data by extracting segments and storing mel spectrograms as tensors.
    Args:
        json_dir (str): Path to the directory containing JSON label files.
        audio_dir (str): Path to the directory containing audio files.
        output_dir (str): Path to the directory where preprocessed data will be stored.
        sr (int): Sample rate to resample the audio.
    """
    os.makedirs(output_dir, exist_ok=True)

    for json_file in tqdm(sorted(os.listdir(json_dir)), desc="Preprocessing JSON Files"): 
        if not json_file.endswith(".json"):
            continue
        if json_file < '00001301-100507.json':
            continue

        print(f'[INFO] processing {json_file}')
        
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
        
        for annotation in annotations:
            start = annotation["start"]
            end = annotation["end"]
            label = annotation["label"]

            # Convert time to sample indices
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            segment = waveform[:, start_sample:end_sample]
            segment = segment.squeeze(0)

            # Extract VGGish embeddings
            segment = model(input_proc(segment))
            output_file = os.path.join(output_dir, f'{json_file.replace(".json", "")}_{start}.pt')
            torch.save({"mel": segment, "label": label}, output_file)


preprocess_and_store(
    json_dir="../os_apnea/train",
    audio_dir="../mic_cleaned",
    output_dir="../preprocessed_apnea_vggish/train"
)
preprocess_and_store(
    json_dir="../os_apnea/val",
    audio_dir="../mic_cleaned",
    output_dir="../preprocessed_apnea_vggish/val"
)