import os
import librosa
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras import layers
import tensorflow_hub as hub
import xml.etree.ElementTree as ET

def load_rml_annotations(rml_file_path):
    """
    Parses an RML file to extract event annotations.
    """
    tree = ET.parse(rml_file_path)
    root = tree.getroot()
    annotations = []

    for event in root.findall('.//{http://www.respironics.com/PatientStudy.xsd}Event'):
        annotations.append({
            'family': event.attrib.get('Family', ''),
            'type': event.attrib.get('Type', ''),
            'start': float(event.attrib.get('Start', '0')),
            'end': float(event.attrib.get('Start', '0')) + float(event.attrib.get('Duration', '0'))
        })

    return annotations

def load_vggish_model():
    """
    Loads the pre-trained VGGish model from TensorFlow Hub.
    """
    return hub.load('https://tfhub.dev/google/vggish/1')

def process_and_train_vggish_model(wav_folder, rml_folder):
    """
    Processes audio files, extracts VGGish features, and trains a classifier.
    """
    all_embeddings, all_labels = [], []
    label_map = {'ObstructiveApnea': 0, 'Hypopnea': 1, 'CentralApnea': 2, 'MixedApnea': 3, 'Snore': 4}
    vggish_model = load_vggish_model()  # Load the VGGish model

    wav_files = [f for f in os.listdir(wav_folder) if f.endswith('.wav')]
    rml_files = {f.split('.')[0]: os.path.join(rml_folder, f) for f in os.listdir(rml_folder) if f.endswith('.rml')}

    for wav_file in wav_files:
        wav_file_path = os.path.join(wav_folder, wav_file)
        base_name = wav_file.split('.')[0]
        rml_file_path = rml_files.get(base_name)

        if not rml_file_path:
            print(f'Missing RML file for {wav_file}. Skipping.')
            continue

        annotations = load_rml_annotations(rml_file_path)
        labels, intervals = [], []
        for annotation in annotations:
            if annotation['type'] in ['ObstructiveApnea', 'Hypopnea', 'CentralApnea', 'MixedApnea', 'Snore']:
                intervals.append((annotation['start'], annotation['end']))
                labels.append(annotation['type'])

        if not labels:
            print(f'No valid labels for {wav_file}. Skipping.')
            continue

        audio, _ = librosa.load(wav_file_path, sr=16000)
        for interval, label in zip(intervals, labels):
            start_frame, end_frame = int(interval[0] * 16000), int(interval[1] * 16000)
            snippet = audio[start_frame:end_frame]
            embeddings = vggish_model(snippet).numpy()
            all_embeddings.append(embeddings)
            all_labels.extend([label_map[label]] * embeddings.shape[0])

    if not all_embeddings:
        print('No data available for training.')
        return

    # Prepare data for training
    all_embeddings = np.vstack(all_embeddings)
    all_labels = np.array(all_labels)
    print(f'[INFO] Embeddings Shape: {all_embeddings.shape}')
    print(f'[INFO] Labels Shape: {all_labels.shape}')

    X_train, X_test, y_train, y_test = train_test_split(all_embeddings, all_labels, test_size=0.2, random_state=42)

    # Define the classification model
    model = tf.keras.Sequential([
        layers.Flatten(input_shape=X_train.shape[1:]),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(len(label_map), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32)

    # Evaluate and display classification report
    y_pred = np.argmax(model.predict(X_test), axis=1)
    print(classification_report(y_test, y_pred, labels=np.arange(5), target_names=list(label_map.keys())))

# Specify paths and execute
wav_folder = '../mic_cleaned'
rml_folder = '../data'
process_and_train_vggish_model(wav_folder, rml_folder)