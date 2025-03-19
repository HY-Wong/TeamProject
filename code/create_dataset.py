import os
import json
import numpy as np
import wave


ONLY_OSA = True

audio_path = '../mic_cleaned'
label_path = '../respiratory'
apnea_path = '../apnea'
train_path = os.path.join(apnea_path, 'train')
val_path = os.path.join(apnea_path, 'val')
os.makedirs(apnea_path, exist_ok=True)
os.makedirs(train_path, exist_ok=True)
os.makedirs(val_path, exist_ok=True)

durations = []
global_apnea_count = 0
global_non_apnea_count = 0

sum_recording = 0

for filename in sorted(os.listdir(label_path)):
    if filename.endswith('.json'):
        print(f'[INFO] Processing {filename}')

        # Load label files
        patient_id = filename.split('.')[0]
        with open(os.path.join(label_path, filename)) as f:
            annotations = json.load(f)

        with wave.open(os.path.join(audio_path, f'{patient_id}.wav'), 'rb') as f:
            frame_rate = f.getframerate()
            n_frames = f.getnframes()
            recording_time = n_frames / frame_rate
            sum_recording += recording_time
        
        apneas, non_apneas, intervals = [], [], []
        for annotation in annotations:
            start = annotation['start']
            duration = annotation['duration']
            end = start + duration
            
            apnea_list = ['ObstructiveApnea'] if ONLY_OSA else ['ObstructiveApnea', 'Hypopnea']
            if annotation['type'] in apnea_list:
                # event-centered windows: center chunks around apnea events 
                # (e.g., 30 seconds before and after the start of an event)
                if start - 30 >= 0 and start + 30 <= recording_time:
                    apneas.append((start - 30, start + 30))
                    global_apnea_count += 1
                durations.append(annotation['duration'])
            
            # context-aware labeling: create non-apnea chunks from periods that are 
            # sufficiently distant from apnea events (e.g., at least 30 seconds away)
            intervals.append((start - 30, end + 30))
                
        # merge overlapped intervals
        intervals = sorted(intervals)
        merged_intervals = []
        for start, end in intervals:
            if not merged_intervals or start > merged_intervals[-1][1]:
                merged_intervals.append((start, end))
            else:
                merged_intervals[-1] = (merged_intervals[-1][0], max(merged_intervals[-1][1], end))

        # find non-overlapping regions (non-apnea regions)
        non_apnea_intervals = []
        last_end = 0
        for start, end in merged_intervals:
            if last_end < start:
                non_apnea_intervals.append((last_end, start))
            last_end = end
        if last_end < recording_time:
            non_apnea_intervals.append((last_end, recording_time))

        # divide non-apnea intervals into 60-second chunks
        for start, end in non_apnea_intervals:
            current = start
            while current + 60 <= end:
                non_apneas.append((current, current + 60))
                current += 60
                global_non_apnea_count += 1

        # split apnea and non-apnea events into train and val sets
        np.random.shuffle(apneas)
        np.random.shuffle(non_apneas)

        split_ratio = 0.8
        train_apneas = apneas[:int(len(apneas) * split_ratio)]
        val_apneas = apneas[int(len(apneas) * split_ratio):]
        train_non_apneas = non_apneas[:int(len(non_apneas) * split_ratio)]
        val_non_apneas = non_apneas[int(len(non_apneas) * split_ratio):]

        # combine and sort by start time
        train_events = [(start, end, 1) for start, end in train_apneas] + [(start, end, 0) for start, end in train_non_apneas]
        val_events = [(start, end, 1) for start, end in val_apneas] + [(start, end, 0) for start, end in val_non_apneas]

        train_events.sort(key=lambda x: x[0])
        val_events.sort(key=lambda x: x[0])

        # save to JSON files
        train_file_path = os.path.join(train_path, f'{patient_id}.json')
        val_file_path = os.path.join(val_path, f'{patient_id}.json')

        with open(train_file_path, 'w') as f:
            json.dump([{'start': start, 'end': end, 'label': label} for start, end, label in train_events], f, indent=4)

        with open(val_file_path, 'w') as f:
            json.dump([{'start': start, 'end': end, 'label': label} for start, end, label in val_events], f, indent=4)

print(f'[INFO] Max duration: {np.max(durations)}')
print(f'[INFO] Mean duration: {np.mean(durations)}')
print(f'[INFO] Percentage of incomplete apnea event: {np.sum(np.array(durations) > 30)/len(durations):.4f}')

print(f'[INFO] Global apnea event count: {global_apnea_count}')
print(f'[INFO] Global non-apnea event count: {global_non_apnea_count}')