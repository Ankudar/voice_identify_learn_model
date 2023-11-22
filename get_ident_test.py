import os
import wave
import tensorflow as tf
import librosa
import numpy as np 
from collections import Counter
from pydub import AudioSegment

spkr_id_file = './data_lists/speaker_id.scp'
new_audio_files = "./random_samples/"
model_random_audio = tf.keras.models.load_model("final-model")
SAMPLING_RATE = 16000

spkr_id = {}

with open(spkr_id_file, "r", encoding = 'utf-8') as file:
    for line in file:
        key, value = line.strip().split(":", 1)
        spkr_id[int(key)] = value

def test_data_gen(wav):
    wav_parts = [wav[i:i+32000] for i in range(0, len(wav), 32000) if len(wav[i:i+32000]) == 32000]
    wav_arrs = [np.array([part.tolist()]) for part in wav_parts]
    return wav_arrs

def convert_to_wav(file_path):
    try:
        if not file_path.endswith('.wav'):
            audio = AudioSegment.from_file(file_path)
            wav_path = os.path.splitext(file_path)[0] + '.wav'
            audio.export(wav_path, format='wav')
            os.remove(file_path)

            with wave.open(wav_path, 'rb') as wave_file:
                current_sample_rate = wave_file.getframerate()
            
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

for new_file in os.listdir(new_audio_files):
    new_file_loc = os.path.join(new_audio_files, new_file)
    new_file_loc = convert_to_wav(new_file_loc)
    stereo_to_mono(new_file_loc)
    new_file_loc = match_amplitude(new_file_loc)
    wav,_ = librosa.load(new_file_loc,sr=16000)
    new_file_arrs = test_data_gen(wav)
    spkrs = []
    for arr in new_file_arrs:
        pred_spk = model_random_audio.predict(arr, verbose=0)
        spkr = np.argmax(pred_spk)
        probability = np.max(pred_spk) * 100
        if probability >= 95:
            spkrs.append(spkr_id[spkr])
    if spkrs:
        most_common_spkrs = Counter(spkrs).most_common(3)  # Изменено здесь
        # Вывод трех наиболее вероятных спикеров
        print(f"File: {new_file} - Top 3 most common speakers: ", end="")
        for spkr, count in most_common_spkrs:
            print(f"{spkr} ({count} times)", end=", ")
        print()  # Добавляет перевод строки в конце вывода
    else:
        print(f"File: {new_file} - Detected speaker is: Unknown")


