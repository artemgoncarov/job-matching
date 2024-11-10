import os
import time
import pandas as pd
import librosa
import numpy as np
from moviepy.editor import VideoFileClip
from catboost import CatBoostRegressor
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True)

model_conscientiousness = CatBoostRegressor()
model_extraversion = CatBoostRegressor()
model_neuroticism = CatBoostRegressor()
model_openness = CatBoostRegressor()
model_agreeableness = CatBoostRegressor()

model_conscientiousness.load_model("conscientiousness_best_model.cbm")
model_extraversion.load_model("extraversion_best_model.cbm")
model_neuroticism.load_model("neuroticism_best_model.cbm")
model_openness.load_model("openness_best_model.cbm")
model_agreeableness.load_model("agreeableness_best_model.cbm")

def convert_video_to_wav(video_path, destination_folder):
    os.makedirs(destination_folder, exist_ok=True)
    
    filename = os.path.basename(video_path)
    audio_output_path = os.path.join(destination_folder, f"{os.path.splitext(filename)[0]}.wav")
    
    try:
        start_time = time.time()

        videoclip = VideoFileClip(video_path)
        audio = videoclip.audio

        audio.write_audiofile(audio_output_path, codec='pcm_s16le')

        audio.close()
        videoclip.close()
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"Converted {filename} to {audio_output_path} in {duration:.2f} seconds")
        return audio_output_path
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None

def extract_spectral_centroid_mean(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_centroid_mean = np.mean(spectral_centroid)
    return spectral_centroid_mean

def extract_spectral_bandwidth_mean(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_bandwidth_mean = np.mean(spectral_bandwidth)
    return spectral_bandwidth_mean

def extract_zero_crossing_rate_mean(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    zero_crossings = librosa.feature.zero_crossing_rate(y)
    zero_crossing_rate_mean = np.mean(zero_crossings)
    return zero_crossing_rate_mean

def process_video_for_prediction(VIDEO_FILE_PATH, AUDIO_DATA_DESTINATION='audio'):
    wav_file_path = convert_video_to_wav(VIDEO_FILE_PATH, AUDIO_DATA_DESTINATION)
    if not wav_file_path:
        print("Conversion failed.")
        return None
    
    df = pd.DataFrame(data={"audio_path": [wav_file_path]})

    df["spectral_centroid_mean"] = df["audio_path"].parallel_apply(extract_spectral_centroid_mean)
    df["spectral_bandwidth"] = df["audio_path"].parallel_apply(extract_spectral_bandwidth_mean)
    df["zero_crossing_rate"] = df["audio_path"].parallel_apply(extract_zero_crossing_rate_mean)

    data = df[['spectral_centroid_mean', 'spectral_bandwidth', 'zero_crossing_rate']]

    predict_model_agreeableness = model_agreeableness.predict(data)
    predict_model_extraversion = model_extraversion.predict(data)
    predict_model_openness = model_openness.predict(data)
    predict_model_conscientiousness = model_conscientiousness.predict(data)
    predict_model_neuroticism = model_neuroticism.predict(data)

    answers = {
        "Conscientiousness": list(predict_model_conscientiousness),
        "Extraversion": list(predict_model_extraversion),
        "Neuroticism": list(predict_model_neuroticism),
        "Openness": list(predict_model_openness),
        "Agreeableness": list(predict_model_agreeableness)
    }
    
    return answers