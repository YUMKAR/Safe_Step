#Config.py

import os

# --- 카메라 및 YOLO 설정 ---
CAM_WIDTH = 360
CAM_HEIGHT = 256
YOLO_IMG_SIZE = 256
YOLO_MODEL_PATH = "yolov8n-human.pt"

# --- 얼굴 인식 임계값 ---
CONF_THRESHOLD = 0.6
FACE_TRACK_THRESHOLD = 30
MIN_FACE_SIZE = 150

MOVEMENT_TRIGGER_THRESHOLD = 20

# --- 이동 감지 및 MP3 설정 ---
MOVEMENT_THRESHOLD = 100
MP3_PATH_IN = os.path.join("static", "in.wav")
MP3_PATH_OUT = os.path.join("static", "out.wav")
MP3_PATH_HOT = os.path.join("static", "hot.wav")
MP3_PATH_RAIN = os.path.join("static", "rain.wav")

# --- 시리얼 설정 ---
SERIAL_PORT = '/dev/rfcomm0'
BAUD_RATE = 9600

# --- GUI 설정 ---
FPS_UPDATE_INTERVAL_MS = 50
