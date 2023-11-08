import os
import random
import numpy as np
from collections import defaultdict

def write_wav_paths_to_file(directory):
    speaker_dict = {}
    speaker_id = 0
    with open('data_lists/all.scp', 'w', encoding='utf-8') as f:
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                if filename.endswith('.wav'):
                    full_path = os.path.join(dirpath, filename)
                    full_path = full_path.replace('./', '').replace('\\', '/')
                    speaker_name = os.path.basename(os.path.dirname(full_path))
                    if speaker_name not in speaker_dict:
                        speaker_dict[speaker_name] = speaker_id
                        speaker_id += 1
                    f.write(f"{full_path}:{speaker_dict[speaker_name]}\n")

    with open('./data_lists/speaker_id.scp', 'w', encoding='utf-8') as f:
        for speaker, id in speaker_dict.items():
            f.write(f"{id}:{speaker}\n")

def split_file(input_file, test_file, train_file, test_ratio=0.2): # 0.2 лушчее пока
    speaker_lines = defaultdict(list)
    with open(input_file, 'r', encoding='utf-8') as file:
        for line in file:
            speaker = line.split('/')[1]
            speaker_lines[speaker].append(line)
    test_lines = []
    train_lines = []
    for lines in speaker_lines.values():
        random.shuffle(lines)
        split_index = int(len(lines) * test_ratio)
        test_lines.extend(lines[:split_index])
        train_lines.extend(lines[split_index:])
    with open(test_file, 'w', encoding='utf-8') as file:
        file.writelines(test_lines)
    with open(train_file, 'w', encoding='utf-8') as file:
        file.writelines(train_lines)

def create_labels_dict(file_path):
    labels_dict = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            path, filename = os.path.split(line.strip())
            _, speaker_id = os.path.split(path)
            labels_dict[line.strip()] = speaker_id
    return labels_dict

write_wav_paths_to_file('./train/data_set/')
split_file('./data_lists/all.scp', './data_lists/test.scp', './data_lists/train.scp')
labels_dict = create_labels_dict('./data_lists/all.scp')
np.save('./data_lists/labels.npy', labels_dict)

# Checking the content of npy
labels_dict = np.load('./data_lists/labels.npy', allow_pickle=True).item()

# Outputting the first N key-value pairs
# for i, (key, value) in enumerate(labels_dict.items()):
#     if i >= 5:
#         break
#     print(f'{key}: {value}')
