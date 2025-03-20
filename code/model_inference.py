import os
import json
import math
import torch
import numpy as np
import torchaudio
from torchaudio.transforms import Resample, MelSpectrogram, AmplitudeToDB

# Global configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sr = 16000  # Target sample rate

##############################
# Helper Function: Audio Loading
##############################
def clean_data_for_mic_in_memory(wav_file):
    """
    Loads and normalizes the WAV file.
    Returns a tensor of shape [1, num_samples] and the sample rate.
    """
    import numpy as np
    from scipy.io import wavfile
    filename = os.path.basename(wav_file)
    sampling_rate, mic_signal = wavfile.read(wav_file)
    mic_signal = mic_signal.astype(np.float32)
    mic_signal = mic_signal / (np.max(np.abs(mic_signal)) + 1e-6)
    cleaned_waveform = torch.from_numpy(mic_signal).unsqueeze(0)
    return cleaned_waveform, sampling_rate

##############################
# Model Definitions
##############################
class CRNN(torch.nn.Module):
    def __init__(self):
        super(CRNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=(3,3), padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=(3,3), padding=1)
        self.pool = torch.nn.MaxPool2d((2,2))
        self.dropout = torch.nn.Dropout(0.3)
        self.rnn = torch.nn.LSTM(input_size=64, hidden_size=128, batch_first=True, num_layers=2)
        self.fc = torch.nn.Linear(128, 2)
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.permute(0, 2, 1, 3)  # (B, time, channel, freq)
        x = torch.mean(x, dim=-1)
        x, _ = self.rnn(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x

class LSTM(torch.nn.Module):
    def __init__(self, input_size=128, hidden_size=128, num_layers=2):
        super(LSTM, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc1 = torch.nn.Linear(hidden_size, 64)
        self.fc2 = torch.nn.Linear(64, 2)
    def forward(self, x):
        x = x.squeeze(1)  # (B, n_mels, T)
        x = x.permute(0, 2, 1)  # (B, T, n_mels)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class MLP(torch.nn.Module):
    def __init__(self, input_shape):
        super(MLP, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(input_shape, 128)
        self.dropout = torch.nn.Dropout(0.5)
        self.fc2 = torch.nn.Linear(128, 2)
    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class GatingNetwork(torch.nn.Module):
    def __init__(self, input_dim=3, hidden_dim=8):
        super(GatingNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

def logits_to_positive_prob(logits):
    probs = torch.softmax(logits, dim=1)
    return probs[:, 1]

def load_all_models(device):
    from torchaudio.prototype.pipelines import VGGISH as TA_VGGISH
    model_crnn = CRNN().to(device)
    model_crnn.load_state_dict(torch.load("./my_api_project/crnn_model.pth", map_location=device))
    model_crnn.eval()
    
    model_lstm = LSTM().to(device)
    model_lstm.load_state_dict(torch.load("./my_api_project/lstm_model.pth", map_location=device))
    model_lstm.eval()
    
    model_vggish = MLP(62 * 128).to(device)
    model_vggish.load_state_dict(torch.load("./my_api_project/vggish_model.pth", map_location=device))
    model_vggish.eval()
    
    meta_learner = GatingNetwork(input_dim=3, hidden_dim=8).to(device)
    meta_learner.load_state_dict(torch.load("./my_api_project/final_meta_learner.pth", map_location=device))
    meta_learner.eval()
    
    vggish_input_proc = TA_VGGISH.get_input_processor()
    vggish_extractor = TA_VGGISH.get_model()
    vggish_extractor.eval()
    
    return model_crnn, model_lstm, model_vggish, meta_learner, vggish_input_proc, vggish_extractor

##############################
# Batch Processing of Sliding Windows
##############################
def process_batch(batch_segments, mel_transform, db_transform,
                  model_crnn, model_lstm, model_vggish, 
                  meta_learner, vggish_input_proc, vggish_extractor):
    """
    Processes a batch of segments.
    Each segment is a tensor of shape (1, num_samples) with num_samples = window_length * sr.
    Returns a list of tuples: (meta_prob, predicted_label, prob_crnn, prob_lstm, prob_vggish)
    for each segment.
    """
    batch_tensor = torch.cat(batch_segments, dim=0)  # (B, num_samples)
    mel_specs = mel_transform(batch_tensor)            # (B, n_mels, T)
    mel_db = db_transform(mel_specs)                     # (B, n_mels, T)
    mel_input = mel_db.unsqueeze(1)                      # (B, 1, n_mels, T)
    
    with torch.no_grad():
        logits_crnn = model_crnn(mel_input)              # (B, 2)
        logits_lstm = model_lstm(mel_input)              # (B, 2)
    prob_crnn = logits_to_positive_prob(logits_crnn)      # (B,)
    prob_lstm = logits_to_positive_prob(logits_lstm)      # (B,)
    
    # Process VGGish branch individually
    vggish_inputs = []
    for seg in batch_segments:
        seg_mono = seg.squeeze(0)  # (num_samples,)
        processed = vggish_input_proc(seg_mono.cpu())  # Expected shape: (T, 1, 96, 64), e.g., T=62
        vggish_inputs.append(processed)
    vggish_inputs = torch.stack(vggish_inputs, dim=0).to(device)  # (B, T, 1, 96, 64)
    
    B, T, C, H, W = vggish_inputs.shape
    vggish_inputs = vggish_inputs.view(B * T, C, H, W)
    with torch.no_grad():
        embeddings = vggish_extractor(vggish_inputs)  # (B*T, D)
    D = embeddings.shape[1]
    embeddings = embeddings.view(B, T, D)
    # Flatten time dimension to match training (e.g., 62*128)
    agg_embeddings = embeddings.view(B, T * D)
    
    with torch.no_grad():
        logits_vggish = model_vggish(agg_embeddings)   # (B, 2)
    prob_vggish = logits_to_positive_prob(logits_vggish)  # (B,)
    
    ensemble_input = torch.stack([prob_crnn, prob_lstm, prob_vggish], dim=1)  # (B, 3)
    with torch.no_grad():
        meta_prob = meta_learner(ensemble_input)  # (B, 1)
    meta_prob = meta_prob.squeeze(1)  # (B,)
    pred_labels = (meta_prob >= 0.5).int()
    
    results = []
    for i in range(B):
        results.append((meta_prob[i].item(), pred_labels[i].item(),
                        prob_crnn[i].item(), prob_lstm[i].item(), prob_vggish[i].item()))
    return results

##############################
# Sliding Window Prediction Function
##############################
def predict_sliding_windows(audio_path, output_json="sliding_window_predictions.json",
                            window_length=60, step=10, batch_size=16):
    """
    Slides a 60-second window over the entire audio file with the specified step (in seconds),
    processes windows in batches using the full ensemble, and returns a list of predictions.
    
    Each prediction dictionary contains:
      "start": window start time,
      "end": window end time,
      "meta_probability": ensemble probability,
      "predicted_label": final predicted label (1 or 0),
      "crnn_probability", "lstm_probability", "vggish_probability"
    """
    waveform, file_sr = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    if file_sr != sr:
        waveform = Resample(orig_freq=file_sr, new_freq=sr)(waveform)
    waveform = waveform.to(device)
    total_duration = waveform.shape[1] / sr
    
    mel_transform = MelSpectrogram(sample_rate=sr, n_fft=1024, hop_length=512, n_mels=128).to(device)
    db_transform = AmplitudeToDB().to(device)
    
    model_crnn, model_lstm, model_vggish, meta_learner, vggish_input_proc, vggish_extractor = load_all_models(device)
    
    num_windows = int((total_duration - window_length) // step) + 1
    predictions = []
    batch_segments = []
    batch_times = []
    
    for i in range(num_windows):
        start_time = i * step
        end_time = start_time + window_length
        start_idx = int(start_time * sr)
        end_idx = int(end_time * sr)
        segment = waveform[:, start_idx:end_idx]  # (1, window_length * sr)
        batch_segments.append(segment)
        batch_times.append({"start": start_time, "end": end_time})
        
        if len(batch_segments) == batch_size or i == num_windows - 1:
            results = process_batch(batch_segments, mel_transform, db_transform,
                                    model_crnn, model_lstm, model_vggish,
                                    meta_learner, vggish_input_proc, vggish_extractor)
            for j, res in enumerate(results):
                meta_prob, pred_label, prob_crnn, prob_lstm, prob_vggish = res
                predictions.append({
                    "start": batch_times[j]["start"],
                    "end": batch_times[j]["end"],
                    "meta_probability": meta_prob,
                    "predicted_label": pred_label,
                    "crnn_probability": prob_crnn,
                    "lstm_probability": prob_lstm,
                    "vggish_probability": prob_vggish
                })
            batch_segments = []
            batch_times = []
            if (i+1) % 50 == 0:
                print(f"Processed {i+1}/{num_windows} windows")
    
    with open(output_json, "w") as f:
        json.dump(predictions, f, indent=4)
    print(f"Sliding window predictions saved to {output_json}")
    return predictions

def run_inference(audio_path):
    """
    Runs the entire sliding-window inference pipeline on the specified audio file.
    Performs sliding-window prediction, applies majority voting to obtain a per-second prediction vector,
    counts contiguous OSA events, computes events per hour, and classifies OSA severity.
    
    Returns a dictionary with:
       - events_per_hour
       - osa_severity
       - total_events
    """
    # Run sliding-window prediction
    predictions = predict_sliding_windows(audio_path, window_length=60, step=10, batch_size=16)
    
    # Load audio to compute total duration
    waveform, file_sr = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    if file_sr != sr:
        waveform = Resample(orig_freq=file_sr, new_freq=sr)(waveform)
    total_duration = waveform.shape[1] / sr
    
    # Majority voting: aggregate overlapping window predictions into per-second labels
    total_seconds = int(math.ceil(total_duration))
    vote_sum = np.zeros(total_seconds, dtype=np.int32)
    vote_count = np.zeros(total_seconds, dtype=np.int32)
    for pred in predictions:
        start = int(pred["start"])
        end = int(math.ceil(pred["end"]))
        vote_sum[start:end] += pred["predicted_label"]
        vote_count[start:end] += 1
    vote_count[vote_count == 0] = 1
    final_vector = (vote_sum / vote_count) >= 0.5
    final_vector = final_vector.astype(np.uint8)
    
    # Count events: a contiguous block of ones is considered one event
    padded = np.concatenate(([0], final_vector, [0]))
    diff = np.diff(padded)
    total_events = int(np.sum(diff == 1))
    
    events_per_hour = total_events / (total_duration / 3600.0)
    
    # Classification based on events per hour (if events_per_hour <= 5 then "Normal")
    if events_per_hour <= 5:
        severity = "Normal"
    elif events_per_hour < 15:
        severity = "Mild OSA"
    elif events_per_hour < 30:
        severity = "Moderate OSA"
    else:
        severity = "Severe OSA"
    
    result = {
        "events_per_hour": events_per_hour,
        "osa_severity": severity,
        "total_events": total_events
    }
    return result

if __name__ == "__main__":
    # For local testing:
    result = run_inference("1502.wav")
    print("Inference Result:")
    print(result)
