import librosa
import librosa.display
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

wav_file = '../mic_cleaned/00001329-100507.wav'
label_file = '../respiratory/00001329-100507.json'

audio, sr = librosa.load(wav_file, sr=16000)
with open(label_file) as f:
    annotations = json.load(f)

start1, end1 = 3200, 3500
start2, end2 = 6900, 7200

colors = {
    'CentralApnea': 'blue',
    'ObstructiveApnea': 'red',
    'MixedApnea': 'green',
    'Hypopnea': 'orange'
}

def plot_figure(start, end, filename):
    chunk = audio[start*sr:end*sr]

    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=chunk, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Create time array for the waveform
    time_wave = np.linspace(0, len(chunk) / sr, num=len(chunk))

    # Plot the waveform
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, width_ratios=[20, 1])

    ax1 = fig.add_subplot(gs[0, 0])  # First row, first column
    time_wave = np.linspace(0, len(chunk) / sr, num=len(chunk))
    ax1.plot(time_wave, chunk, zorder=1)
    ax1.set_xlim(0, len(chunk) / sr)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Signal")

    # Get current y-axis limits
    y_min, y_max = ax1.get_ylim()

    added_labels = {}

    # Overlay apnea annotations on waveform
    for annotation in annotations:
        event_start, event_end = annotation['start'], annotation['start'] + annotation['duration']
        if event_start >= start and event_start < end:
            event_end = min(end, event_end)
            label = annotation['type']
            color = colors[label]
        
            rect = plt.Rectangle(
                (event_start - start, y_min),  # x, y
                event_end - event_start,  # width
                y_max - y_min,  # height (full frequency range)
                linewidth=3,
                edgecolor=color,
                fill=False,
                label=label,
                zorder=2
            )
            ax1.add_patch(rect)
            if label not in added_labels:
                added_labels[label] = rect

    # Create legend with one entry per apnea type
    handles = [rect for label, rect in added_labels.items()]
    labels = [label for label in added_labels.keys()]
    
    # Add a subplot for the legend at gs[0, 1]
    leg_ax = fig.add_subplot(gs[0, 1])
    leg_ax.axis('off') 
    leg_ax.legend(handles, labels, loc='center')
    
    # Plot the mel spectrogram
    ax2 = fig.add_subplot(gs[1, 0])
    librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='log', ax=ax2)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Frequency (Hz)")

    # Add colorbar for the spectrogram
    cbar_ax = fig.add_subplot(gs[1, 1])  # Second row, second column (colorbar)
    plt.colorbar(librosa.display.specshow(mel_spec_db, sr=sr, ax=ax2, y_axis='mel'), cax=cbar_ax, format='%+2.0f dB')

    # Overlay apnea annotations on mel spectrogram

    plt.tight_layout()
    plt.savefig(filename)

plot_figure(start1, end1, 'temp1.png')
plot_figure(start2, end2, 'temp2.png')