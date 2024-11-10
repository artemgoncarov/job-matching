
import stable_whisper
import pandas as pd
from moviepy.editor import VideoFileClip
import csv
import torch


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = stable_whisper.load_model('large').to(device)

def extract_audio(video_file, output_audio_file):
    video_clip = VideoFileClip(video_file)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(output_audio_file)
    audio_clip.close()


def get_text(video_path):
    extract_audio(video_path, 'audio.mp3')

    result = model.transcribe("audio.mp3")
    result.to_tsv("audio.tsv")

    text = []
    tsv_file = open("audio.tsv", 'r+', encoding='utf-8')
    read_tsv = csv.reader(tsv_file, delimiter="\t")
    for row in read_tsv:
        if len(row) > 0:
            text.append(row[-1])

    return ' '.join(text)