from concurrent.futures import ThreadPoolExecutor
import math
import statistics
import os
import numpy as np
import soundfile as sf
import librosa
import shutil
from scipy.signal import lfilter, butter
from pydub import AudioSegment
import json
from tqdm import tqdm
import wave
import noisereduce as nr
from scipy.signal import butter, lfilter, stft, istft
from scipy.fftpack import fftshift
import tensorflow_io as tfio
from torchaudio.transforms import Resample
import torch
from IPython.display import Audio
import warnings
import itertools

warnings.filterwarnings("ignore", ".*prim::profile_ivalue.*")

input_dir = './train/data_set/'
WORK_THREADS = 10
SAMPLING_RATE = 16000
THRESHOLD = 2000 # продолжительность чанков, 1000 = 1 секунда, для сплита файлов
AUDIO_TIME = 4 # сколько в каждой папке будет аудио часов

def convert_to_wav(file_path):
    try:
        if not file_path.endswith('.wav'):
            audio = AudioSegment.from_file(file_path)
            wav_path = os.path.splitext(file_path)[0] + '.wav'
            audio.export(wav_path, format='wav')
            os.remove(file_path)

            # Check the sample rate of the current file
            with wave.open(wav_path, 'rb') as wave_file:
                current_sample_rate = wave_file.getframerate()
            
            # If the current sample rate is not the target sample rate, resample the audio
            if current_sample_rate != SAMPLING_RATE:
                audio = AudioSegment.from_wav(wav_path)
                resampled_audio = audio.set_frame_rate(SAMPLING_RATE)
                resampled_audio.export(wav_path, format='wav')
            return wav_path
    except Exception as e:
        print(f"Произошла ошибка при конвертации файла {file_path}")
        print(f"Тип ошибки: {type(e).__name__}, Описание ошибки: {str(e)}")
    return file_path

def stereo_to_mono(audio_file_path):
    sound = AudioSegment.from_wav(audio_file_path)
    if sound.channels > 1:
        sound = sound.set_channels(1)
        sound.export(audio_file_path, format="wav")

def match_amplitude(wav_path, target_dBFS=-30.0):
    if wav_path.endswith('.wav'):
        audio = AudioSegment.from_file(wav_path)
        normalized_audio = audio.apply_gain(target_dBFS - audio.dBFS)
        normalized_audio.export(wav_path, format='wav')
    return wav_path

def remove_silence(file_path, silence_threshold=0.001, chunk_size=200):
    try:
        # Загрузите аудиофайл
        audio, sr = librosa.load(file_path, sr=None)

        # Разделите аудио на части и вычислите амплитуду каждого
        chunks = [audio[i:i+chunk_size] for i in range(0, len(audio), chunk_size)]
        amplitudes = [np.mean(np.abs(chunk)) for chunk in chunks]

        # Определите части тишины и не тишины
        silence = [amp < silence_threshold for amp in amplitudes]
        non_silence = [not s for s in silence]

        # Удалите тишину и сохраните аудио
        audio_non_silence = np.concatenate([chunks[i] for i in range(len(chunks)) if non_silence[i]])
        sf.write(file_path, audio_non_silence, sr)
    except Exception as e:
        with open('./vad_error.txt', 'a', encoding='utf-8') as f:
            f.write(f"{file_path}\n{e}\n\n")
        print(f"Произошла ошибка с файлом {file_path}: {e}")
        if os.path.isfile(file_path):
            os.remove(file_path)

def remove_silence_all_files(input_dir):
    print("Запуск remove_silence_all_files...")
    file_paths = []
    for root, dirs, files in os.walk(input_dir):
        dirs[:] = [d for d in dirs if not d.startswith('_')]
        file_paths.extend(os.path.join(root, file) for file in files if file.endswith(".wav"))

    with ThreadPoolExecutor(max_workers=WORK_THREADS) as executor:
        with tqdm(total=len(file_paths)) as pbar:
            for _ in executor.map(remove_silence, file_paths):
                pbar.update()

def do_remove_null_files_and_subdir(file_path, check_duration):
    try:
        data, samplerate = sf.read(file_path)
        duration = len(data) / samplerate

        if duration < check_duration:
            os.remove(file_path)

        parent_dir = os.path.dirname(file_path)
        if not os.listdir(parent_dir):
            os.rmdir(parent_dir)
    except:
        print(f"Error processing file: {file_path}")

def convert_now_wav(file_path):
    wav_path = convert_to_wav(file_path)
    match_amplitude(wav_path)
    stereo_to_mono(wav_path)

def get_all_files(input_dir):
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            yield os.path.join(root, file)

def convert_all_files(input_dir):
    print("Running convert_all_files...")
    with ThreadPoolExecutor(max_workers=WORK_THREADS) as executor:
        file_paths = list(get_all_files(input_dir))
        with tqdm(total=len(file_paths)) as pbar:
            for _ in executor.map(convert_now_wav, file_paths):
                pbar.update()

def remove_null_files_and_subdir(input_dir, check_duration):
    print("Running remove_null_files_and_subdir...")
    with ThreadPoolExecutor(max_workers=WORK_THREADS) as executor:
        file_paths = list(get_all_files(input_dir))
        with tqdm(total=len(file_paths)) as pbar:
            for _ in executor.map(do_remove_null_files_and_subdir, file_paths, itertools.repeat(check_duration)):
                pbar.update()
            
    for dirpath, dirnames, filenames in os.walk(input_dir, topdown=False):
        if not dirnames and not filenames:
            os.rmdir(dirpath)

def check_long(input_dir):
    print("Running check_long...")
    durations = {}
    for folder in os.listdir(input_dir):
        folder_path = os.path.join(input_dir, folder)
        if os.path.isdir(folder_path):
            total_duration = 0
            for file in os.listdir(folder_path):
                if file.endswith(".wav"):
                    audio = AudioSegment.from_wav(os.path.join(folder_path, file))
                    total_duration += len(audio)
            durations[folder] = total_duration / 1000  # возвращаем длительность в секундах
            # если общая продолжительность меньше 30 минут, удаляем подпапку
            if total_duration / 1000 < 3600:
                shutil.rmtree(folder_path)
    # сортируем словарь по длительности
    sorted_durations = dict(sorted(durations.items(), key=lambda item: item[1]))
    with open('audio_long.json', 'w', encoding='utf-8') as f:
        json.dump(sorted_durations, f, ensure_ascii=False, indent=4)

def count_audio_duration_in_subdirectories(directory):
    subdirectory_audio_durations = {}
    for path, dirs, files in os.walk(directory):
        for dir in dirs:
            dir_path = os.path.join(path, dir)
            files_in_dir = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f)) and f.endswith('.wav')]
            if files_in_dir:  # Skip if directory is empty
                total_duration = 0
                for file in files_in_dir:
                    file_path = os.path.join(dir_path, file)
                    signal, sr = librosa.load(file_path, sr=None)
                    duration = len(signal) / sr
                    total_duration += duration
                subdirectory_audio_durations[dir_path] = total_duration
    return subdirectory_audio_durations

def balance_audio_duration_in_subdirectories(directory):
    print("Запуск balance_audio_duration_in_subdirectories...")
    subdirectory_audio_durations = count_audio_duration_in_subdirectories(directory)

    target_audio_duration = AUDIO_TIME * 60 * 60  # N часов

    for dir_path, audio_duration in subdirectory_audio_durations.items():
        files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f)) and f.endswith('.wav')]
        if not files:  # If directory is empty, remove it and continue to the next one
            os.rmdir(dir_path)
            continue

        # Increase duration by copying files
        if audio_duration < target_audio_duration:
            i = 0
            while audio_duration < target_audio_duration:
                new_file_name = f"copy_{i}_{files[i % len(files)]}"
                shutil.copy2(os.path.join(dir_path, files[i % len(files)]), os.path.join(dir_path, new_file_name))
                signal, sr = librosa.load(os.path.join(dir_path, new_file_name), sr=None)
                duration = len(signal) / sr
                audio_duration += duration
                i += 1

            # If the total duration exceeds the target duration, trim the last audio file
            if audio_duration > target_audio_duration:
                last_file_path = os.path.join(dir_path, new_file_name)
                signal, sr = librosa.load(last_file_path, sr=None)
                duration_to_trim = audio_duration - target_audio_duration
                samples_to_trim = int(duration_to_trim * sr)
                trimmed_signal = signal[:-samples_to_trim]
                sf.write(last_file_path, trimmed_signal, sr)
                audio_duration = target_audio_duration

        # Decrease duration by deleting files
        elif audio_duration > target_audio_duration:
            files.sort(reverse=True)  # Start deleting from the newest files
            i = 0
            while audio_duration > target_audio_duration and i < len(files):
                file_path = os.path.join(dir_path, files[i])
                signal, sr = librosa.load(file_path, sr=None)
                duration = len(signal) / sr
                os.remove(file_path)
                audio_duration -= duration
                i += 1

            # If the total duration is less than the target duration after deleting files, copy and trim the last deleted file
            if audio_duration < target_audio_duration:
                last_file_name = f"copy_{i}_{files[i]}"
                shutil.copy2(os.path.join(dir_path, files[i]), os.path.join(dir_path, last_file_name))
                signal, sr = librosa.load(os.path.join(dir_path, last_file_name), sr=None)
                duration_to_add = target_audio_duration - audio_duration
                samples_to_add = int(duration_to_add * sr)
                trimmed_signal = signal[:samples_to_add]
                sf.write(os.path.join(dir_path, last_file_name), trimmed_signal, sr)

def split_audio(directory):
    print("Running split_audio...")
    subdirectory_audio_durations = {}
    files = [(subdir, file) for subdir, dirs, files in os.walk(directory) for file in files if file.endswith(".wav")]
    for subdir, file in tqdm(files, desc="Processing files"):
        filepath = subdir + os.sep + file
        audio = AudioSegment.from_wav(filepath)
        length_audio = len(audio)
        start = 0
        end = 0
        counter = 0
        while start < len(audio):
            end += THRESHOLD
            if end > len(audio):
                end = len(audio)
            chunk = audio[start:end]
            filename = os.path.splitext(filepath)[0] + f'_chunk{counter}.wav'
            chunk.export(filename, format="wav", bitrate="256k")
            counter += 1
            start += THRESHOLD
        os.remove(filepath)
        if subdir in subdirectory_audio_durations:
            subdirectory_audio_durations[subdir] += length_audio
        else:
            subdirectory_audio_durations[subdir] = length_audio

def normalize(input_dir):
    print("Running normalize...")
    # Read List file
    list_sig = []
    for path, subdirs, files in os.walk(input_dir):
        for name in files:
            if name.endswith('.wav'):
                list_sig.append(os.path.join(path, name))

    # Speech Data Normalization Loop
    for wav_file in tqdm(list_sig, desc="Normalizing", unit="file"): 
        # Open the wav file
        [signal, fs] = sf.read(wav_file)
        signal = signal.astype(np.float64)

        # Signal normalization
        signal = signal / np.max(np.abs(signal))

        # Save normalized speech
        try:
            sf.write(wav_file, signal, fs)
        except Exception as e:
            print(f"Error while saving file: {e}")

def main():
    convert_all_files(input_dir)
    normalize(input_dir)
    remove_silence_all_files(input_dir)
    remove_null_files_and_subdir(input_dir, check_duration=10)
    check_long(input_dir)
    split_audio(input_dir)
    remove_null_files_and_subdir(input_dir, check_duration=THRESHOLD/1000)
    balance_audio_duration_in_subdirectories(input_dir)
    # remove_null_files_and_subdir(input_dir, check_duration=THRESHOLD/1000)

if __name__ == '__main__':
    main()