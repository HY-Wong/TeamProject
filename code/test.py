import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sr = 16000
test_time = 1800 # Test 30 minutes of the recording
test_step = 10 # Shift the window by 30 seconds

patient_id = '1500'
start_time = 3000
# patient_id = '1502'
# start_time = 0

json_file = f'../data/0000{patient_id}-100507_respiratory.json'
audio_file = f'../data/0000{patient_id}-100507_mic_cleaned.wav'

# LSTM Model Definition
class LSTM(nn.Module):
    def __init__(self, input_size=128, hidden_size=128, num_layers=2):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 2)
        
    def forward(self, x):
        if len(x.shape) == 4:
            x = x.squeeze(1)    # Remove channel -> (batch_size, n_mels, time)
        x = x.permute(0, 2, 1)  # Reshape -> (batch_size, time, n_mels)
        
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Take the last time step output
        
        x = F.relu(self.fc1(x), inplace=True)
        x = self.fc2(x)
        return x


with open(json_file, 'r') as f:
    annotations = json.load(f)

gt_series = torch.zeros(test_time)

for annotation in annotations:
    start = annotation['start']
    end = start + annotation['duration']
    label = annotation['type']

    if start >= start_time + test_time:
        break

    if label == 'ObstructiveApnea' and end >= start_time:
        start = max(math.floor(start), start_time)
        end = min(math.ceil(end), start_time + test_time)
        gt_series[start - start_time : end - start_time] = 1
        print(start, end)

waveform, sample_rate = torchaudio.load(audio_file)
waveform = waveform[:, sample_rate * start_time : sample_rate * (start_time + test_time)]

if sample_rate != sr:
    resampler = T.Resample(orig_freq=sample_rate, new_freq=sr)
    waveform = resampler(waveform)
print(waveform.shape)

mel_transform = T.MelSpectrogram(sample_rate=sr, n_fft=1024, hop_length=512, n_mels=128)
db_transform = T.AmplitudeToDB()

model = LSTM().to(device)
model.load_state_dict(torch.load('../output/lstm_obstructive_apnea.pth', map_location=device, weights_only=True))
model.eval()

preds = torch.zeros(test_time, dtype=torch.int8)
counts = torch.zeros(test_time, dtype=torch.int8)

num_windows = int((test_time - 60) / test_step + 1)

for i in range(num_windows):
    print(f'Windows {i + 1}')
    start = i * test_step
    end = i * test_step + 60
    segment = waveform[:, start * sr : end * sr].clone()
    segment = mel_transform(segment)
    segment = db_transform(segment)
    segment = segment.to(device)

    with torch.no_grad():
        probs = torch.softmax(model(segment), dim=-1)
    
    preds[start:end] += torch.argmax(probs.cpu(), dim=1)
    counts[start:end] += 1

pred_series = (preds >= counts / 2).int()

gt_series = gt_series.numpy()
pred_series = pred_series.numpy()
waveform = waveform.numpy()
time_audio = torch.arange(waveform.shape[1]) / sr  # Time axis for audio

# Create the plot
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot audio waveform (Primary Y-axis)
ax1.plot(time_audio, waveform[0], color="black", alpha=0.5, lw=0.5, label="Audio Waveform")  # First channel
ax1.set_xlabel("Time (seconds)")
ax1.set_ylabel("Audio Amplitude", color="black")
ax1.tick_params(axis='y', labelcolor="black")

# Create secondary Y-axis for apnea labels
ax2 = ax1.twinx()
ax2.plot(gt_series, label="Ground Truth", linestyle="dashed", lw=2.0, color="blue", alpha=0.7)
ax2.plot(pred_series, label="Prediction", lw=2.0, color="red", alpha=0.7)
ax2.set_ylabel("Apnea Label", color="black")
ax2.tick_params(axis='y', labelcolor="black")

# Set x-axis range from 0 to 1800 seconds
ax1.set_xlim([0, test_time])

# Legends
fig.legend(loc="upper right")

# Show the plot
plt.savefig(f'../output/{patient_id}.png')