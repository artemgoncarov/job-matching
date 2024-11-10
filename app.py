from flask import Flask, render_template, request, Response, jsonify, send_file
import os
from pathlib import Path
import zipfile
import rarfile
from werkzeug.utils import secure_filename
from nlp_model import predict_personality_traits
from get_text import get_text
from ocean2mbti import ocean_to_mbti
from careers import mbti_careers
import pandas as pd
import cv2

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
UPLOAD_FOLDER = 'uploads'  # Папка для загрузки архивов
VIDEO_FOLDER = 'videos'   # Папка для сохранения видео
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'flv'}  # Разрешенные расширения видео

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists('static/photos'):
    os.mkdir('static/photos')

if not os.path.exists('static/uploads'):
    os.mkdir('static/uploads')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def safe_filename(filename):
    return secure_filename(filename)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/admin')
def admin():
    return render_template('admin.html')


@app.route('/selectPeople')
def selectPeople():
    return render_template('selectPeople.html')

@app.route('/showAllVideos')
def showAllVideos():
    return render_template('showAllVideos.html')

@app.route('/instruction')
def instruction():
    return render_template('instruction.html')


@app.route('/getAllVideos', methods=['GET'])
def getAllVideos():
    data = pd.read_csv('data.csv')

    res = []
    ids = data.index.tolist()

    for i, row in data.iterrows():
        res.append([ids[i], row.Openness, row.Conscientiousness, row.Extraversion, row.Agreeableness, row.Neuroticism, row.MBTI, row.careers, row.photo])
    
    return jsonify(res)

@app.route('/getOcean', methods=['GET'])
def getOcean():
    video = request.args.get('video')

    text = get_text(video)
    preds = predict_personality_traits(text)
    preds['MBTI'] = ocean_to_mbti(preds)
    preds['careers'] = mbti_careers[ocean_to_mbti(preds)]
        
    a = {
        list(preds.keys())[0]: [float(preds[list(preds.keys())[0]])],
        list(preds.keys())[1]: [float(preds[list(preds.keys())[1]])],
        list(preds.keys())[2]: [float(preds[list(preds.keys())[2]])],
        list(preds.keys())[3]: [float(preds[list(preds.keys())[3]])],
        list(preds.keys())[4]: [float(preds[list(preds.keys())[4]])],
        list(preds.keys())[5]: [str(preds[list(preds.keys())[5]])],
        "careers": ", ".join(mbti_careers[str(preds[list(preds.keys())[5]])])  # преобразуем список профессий в строку
    }

    return jsonify(a)


@app.route('/upload', methods=['POST'])
def upload_video():
    file = request.files['video']
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        return jsonify({'video_url': file_path})
    return jsonify({'error': 'No file uploaded'}), 400

@app.route('/upload_zip', methods=['POST'])
def upload_zip():
    file = request.files['zip']
    if file:
        filename = file.filename
        archive_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(archive_path)

        if filename.endswith('.zip'):
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(UPLOAD_FOLDER)
        elif filename.endswith('.rar'):
            with rarfile.RarFile(archive_path, 'r') as rar_ref:
                rar_ref.extractall(UPLOAD_FOLDER)

        extracted_files = os.listdir(UPLOAD_FOLDER)
        saved_videos = []
        predictions = []

        for file_name in extracted_files:
            file_path = os.path.join(UPLOAD_FOLDER, file_name)

            if '__MACOSX' in file_name or file_name.startswith('.'):
                continue

            if os.path.isfile(file_path) and allowed_file(file_name):
                video_filename = safe_filename(file_name)

                video_path = os.path.join(VIDEO_FOLDER, video_filename)
                os.rename(file_path, video_path)
                saved_videos.append(video_path)
                cap = cv2.VideoCapture(video_path)

                while 1:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    cv2.imwrite('static/photos/' + ('.'.join(video_path.split('.')[:-1])[7:]) + '.jpg', frame)
                    break
            else:
                os.remove(file_path)

        if os.path.exists(archive_path):
            os.remove(archive_path)
        
        for video_path in saved_videos:
            text = get_text(video_path)
            preds = predict_personality_traits(text)
            preds['MBTI'] = ocean_to_mbti(preds)
            predictions.append(preds)

        res = []
        
        data = pd.read_csv('data.csv')
        for i, ocean in enumerate(predictions):
            a = {
                list(ocean.keys())[0]: [round(float(ocean[list(ocean.keys())[0]]), 3)],
                list(ocean.keys())[1]: [round(float(ocean[list(ocean.keys())[1]]), 3)],
                list(ocean.keys())[2]: [round(float(ocean[list(ocean.keys())[2]]), 3)],
                list(ocean.keys())[3]: [round(float(ocean[list(ocean.keys())[3]]), 3)],
                list(ocean.keys())[4]: [round(float(ocean[list(ocean.keys())[4]]), 3)],
                list(ocean.keys())[5]: [str(ocean[list(ocean.keys())[5]])],
                "careers": ", ".join(mbti_careers[str(ocean[list(ocean.keys())[5]])])  # преобразуем список профессий в строку
            }
            b = a.copy()
            b['video'] = saved_videos[i]
            b['photo'] = 'static/photos/' + ('.'.join(saved_videos[i].split('.')[:-1])[7:]) + '.jpg'
            data = pd.concat([data, pd.DataFrame(b)])
            res.append(a)
        
        data = data.to_csv('data.csv', index=False)

        print(res)
        return res
        
    return jsonify({'error': 'No file uploaded'}), 400

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(VIDEO_FOLDER, exist_ok=True)
    app.run(port=1488, debug=True, host='0.0.0.0')