import os
import sqlite3
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, render_template, request
from PIL import Image

# Directory paths
dataset_dir = "dataset"
trainer_dir = "trainer"
attendance_dir = "attendance"
unknown_attempts_dir = "unknown_attempts"

# Create directories if they don't exist
os.makedirs(dataset_dir, exist_ok=True)
os.makedirs(trainer_dir, exist_ok=True)
os.makedirs(attendance_dir, exist_ok=True)
os.makedirs(unknown_attempts_dir, exist_ok=True)

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Dictionary to hold user ID and names
user_details = {}

def init_db():
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    # Create the attendance table if it doesn't exist
    c.execute('''CREATE TABLE IF NOT EXISTS attendance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    user_name TEXT NOT NULL,
                    date TEXT NOT NULL
                )''')
    conn.commit()
    conn.close()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/attendance')
def view_attendance():
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute("SELECT * FROM attendance")
    attendance_data = c.fetchall()
    conn.close()
    return render_template('attendance_table.html', attendance_data=attendance_data)

@app.route('/train', methods=['GET', 'POST'])
def train():
    if request.method == 'POST':
        user_id = request.form.get('user_id')
        user_name = request.form.get('user_name')
        if user_id and user_name:
            capture_images_and_train_model(user_id, user_name)
            return "Model trained successfully"
        else:
            return "Please provide both user ID and name."
    else:
        return render_template('train.html')

def capture_images_and_train_model(user_id, user_name):
    user_details[int(user_id)] = user_name

    cam = cv2.VideoCapture(0)
    sample_num = 0

    while True:
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            sample_num += 1
            cv2.imwrite(f"{dataset_dir}/User.{user_id}.{sample_num}.jpg", gray[y:y+h, x:x+w])
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.waitKey(100)

        cv2.imshow('frame', frame)
        cv2.waitKey(1)
        if sample_num >= 60:
            break

    cam.release()
    cv2.destroyAllWindows()
    train_model()

def get_images_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    face_samples = []
    ids = []

    for image_path in image_paths:
        gray_img = Image.open(image_path).convert('L')
        img_numpy = np.array(gray_img, 'uint8')
        id = int(os.path.split(image_path)[-1].split(".")[1])
        faces = face_cascade.detectMultiScale(img_numpy)

        for (x, y, w, h) in faces:
            face_samples.append(img_numpy[y:y+h, x:x+w])
            ids.append(id)

    return face_samples, ids

def train_model():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces, ids = get_images_and_labels(dataset_dir)
    recognizer.train(faces, np.array(ids))
    recognizer.write(f"{trainer_dir}/trainer.yml")

@app.route('/mark_attendance', methods=['GET', 'POST'])
def mark_attendance():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(f"{trainer_dir}/trainer.yml")
    cam = cv2.VideoCapture(0)
    
    recognized_users = set()  # Set to keep track of recognized users in the session

    while True:
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
            if confidence < 100:
                user_name = user_details.get(id)
                if user_name and id not in recognized_users:
                    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    conn = sqlite3.connect('attendance.db')
                    c = conn.cursor()
                    c.execute("INSERT INTO attendance (user_id, user_name, date) VALUES (?, ?, ?)", (id, user_name, date))
                    conn.commit()
                    conn.close()
                    recognized_users.add(id)
                    feedback_message = f"Attendance marked for {user_name}"
                else:
                    feedback_message = "User already marked"
            else:
                feedback_message = "Unknown user"

            cv2.putText(frame, feedback_message, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
    return "Attendance marking complete"

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
