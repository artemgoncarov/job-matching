import os
import time
import pandas as pd
import librosa
import numpy as np
import tqdm
from moviepy.editor import VideoFileClip
from catboost import CatBoostRegressor
from pandarallel import pandarallel


pandarallel.initialize(progress_bar=True)

def process_video_for_prediction(VIDEO_FILE_PATH, AUDIO_DATA_DESTINATION):
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

    def process_emotion_predictions(data: pd.DataFrame) -> pd.DataFrame:
        data[['happy', 'neutral', 'fear', 'disgust', 'angry']] = None 
    
        label_map = {
            "LABEL_0": "happy",
            "LABEL_1": "neutral",
            "LABEL_2": "fear",
            "LABEL_3": "disgust",
            "LABEL_4": "angry",
            "LABEL_5": "surprise", 
        }
        for idx, row in tqdm(data.iterrows(), total=data.shape[0], desc="Processing Audio Files"):
    
            predictions = pipe(row['audio_path'])
            happiness = {label_map[f"LABEL_{i}"]: 0 for i in range(len(label_map))}
    
            # Map predictions to the happiness dictionary
            for pred in predictions:
                happiness[label_map[pred['label']]] = pred["score"]
    
            # Now, assign the predicted values to the respective columns
            data.at[idx, 'happy'] = happiness['happy']
            data.at[idx, 'neutral'] = happiness['neutral']
            data.at[idx, 'fear'] = happiness['fear']
            data.at[idx, 'disgust'] = happiness['disgust']
            data.at[idx, 'angry'] = happiness['angry']
    
        return data
        
    wav_file_path = convert_video_to_wav(VIDEO_FILE_PATH, AUDIO_DATA_DESTINATION)

    if not wav_file_path:
        print("Conversion failed.")
        return None
    df = pd.DataFrame(data={"audio_path": [wav_file_path]})

    df["spectral_centroid_mean"] = df["audio_path"].parallel_apply(extract_spectral_centroid_mean)
    df["spectral_bandwidth"] = df["audio_path"].parallel_apply(extract_spectral_bandwidth_mean)
    df["zero_crossing_rate"] = df["audio_path"].parallel_apply(extract_zero_crossing_rate_mean)
    df = process_emotion_predictions(df)

    data = df[["spectral_bandwidth", "zero_crossing_rate", "spectral_centroid_mean", "happy", "neutral", "fear", "disgust", "angry"]]

    model_conscientiousness = CatBoostRegressor()
    model_extraversion = CatBoostRegressor()
    model_neuroticism = CatBoostRegressor()
    model_openness = CatBoostRegressor()
    model_agreeableness = CatBoostRegressor()

    model_conscientiousness.load_model("conscientiousness_best_model_audio_features.cbm")
    model_extraversion.load_model("extraversion_best_model_audio_features.cbm")
    model_neuroticism.load_model("neuroticism_best_model_audio_features.cbm")
    model_openness.load_model("openness_best_model_audio_features.cbm")
    model_agreeableness.load_model("agreeableness_best_model_audio_features.cbm")
    predict_model_agreeableness = model_agreeableness.predict(data)
    predict_model_extraversion = model_extraversion.predict(data)
    predict_model_openness = model_openness.predict(data)
    predict_model_conscientiousness = model_conscientiousness.predict(data)
    predict_model_neuroticism = model_neuroticism.predict(data)

    answers = {
        "conscientiousness": predict_model_conscientiousness,
        "extraversion": predict_model_extraversion,
        "neuroticism": predict_model_neuroticism,
        "openness": predict_model_openness,
        "agreeableness": predict_model_agreeableness
    }

    return answers
    
results_df = pd.DataFrame(columns=["filename", "conscientiousness", "extraversion", "neuroticism", "openness", "agreeableness"])

VIDEO_FOLDER_PATH = r'D:\Projects\CP_megnar\test_dataset_vprod_encr_test\test_data\test000'
AUDIO_DATA_DESTINATION = r'D:\Projects\CP_megnar\job-matching\folder_folder'

for filename in os.listdir(VIDEO_FOLDER_PATH):
    file_path = os.path.join(VIDEO_FOLDER_PATH, filename)
    answers = process_video_for_prediction(file_path, AUDIO_DATA_DESTINATION)
    print(answers)
    result_row = {
                "filename": filename,
                "conscientiousness": answers.get("conscientiousness", [None])[0],
                "extraversion": answers.get("extraversion", [None])[0],
                "neuroticism": answers.get("neuroticism", [None])[0],
                "openness": answers.get("openness", [None])[0],
                "agreeableness": answers.get("agreeableness", [None])[0]
            }
    results_df = pd.concat([results_df, pd.DataFrame([result_row])], ignore_index=True)
