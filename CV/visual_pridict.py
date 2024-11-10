import os
import cv2
import torch
import pandas as pd
from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image
import torch.nn.functional as F

# Инициализация процессора и модели
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

def get_yes_probability(image):
    inputs = processor(images=image, text="Is there something in the frame?", return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    answer_logits = outputs.logits[0]
    yes_index = None
    for idx, label in model.config.id2label.items():
        if label.lower() == "yes":
            yes_index = idx
            break

    if yes_index is not None:
        probabilities = F.softmax(answer_logits, dim=0)
        return probabilities[yes_index].item()
    else:
        return None

def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_probs = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Преобразуем кадр из формата BGR (OpenCV) в RGB (PIL)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Получаем вероятность "yes" для каждого кадра
        yes_prob = get_yes_probability(pil_image)
        if yes_prob is not None:
            frame_probs.append(yes_prob)
    
    cap.release()
    
    # Возвращаем среднюю вероятность для видео
    if frame_probs:
        return sum(frame_probs) / len(frame_probs)
    else:
        return None

def analyze_videos_in_folder(root_folder):
    results = []
    for video_file in os.listdir(root_folder):
        print(video_file)
        video_path = os.path.join(root_folder, video_file)
        if os.path.isfile(video_path) and video_file.endswith(('.mp4', '.avi', '.mkv')):
            avg_yes_prob = analyze_video(video_path)
            if avg_yes_prob is not None:
                results.append({'video': video_file, 'avg_yes_probability': avg_yes_prob})
                print(f"Processed {video_file}: Average 'yes' probability = {avg_yes_prob:.4f}")
    return pd.DataFrame(results)

root_folder = "guivans_folder/train/audio_data_train/"  # Путь к папке с видео
results_df = analyze_videos_in_folder(root_folder)

# Сохраняем результаты в CSV
results_df.to_csv("video_yes_probabilities.csv", index=False)
print("Results saved to video_yes_probabilities.csv")
