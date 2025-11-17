import os
import cv2
import threading
import time
from ultralytics import YOLO
import pygame
import platform
import subprocess
from Config import *

# 오디오 드라이버 설정
if platform.system() == "Linux":
    os.environ["SDL_AUDIODRIVER"] = "alsa"
elif platform.system() == "Windows":
    os.environ["SDL_AUDIODRIVER"] = "directsound"
else:
    os.environ["SDL_AUDIODRIVER"] = "coreaudio"


class YoloDetector:
    def __init__(self, lock, latest_frame_var, face_info_var, mp3_played_var, fps_queue, sensor_data_var=None):
        self.lock = lock
        self.latest_annotated_frame = latest_frame_var
        self.face_info = face_info_var
        self.mp3_played_for_movement = mp3_played_var
        self.fps_queue = fps_queue

        # 방향 설정
        self.direction_mode = "left_out"

        # 센서 값 초기화
        self.sensor_data = sensor_data_var if sensor_data_var else {
            "humidity": 0.0,
            "temperature": 0.0,
            "light": 0,
            "rain": 900
        }

        self.cap = None
        self.model = None
        self.first_face_box_coords = None
        self.previous_face_center_x = None

        self._init_resources()

    # ------------------------ 초기 설정 ------------------------
    def _init_resources(self):
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=2048)

            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

            self.model = YOLO(YOLO_MODEL_PATH)
            print(f"[YOLO] 모델 로드 완료: {YOLO_MODEL_PATH}")

        except Exception as e:
            print(f"[YOLO 초기화 오류] {e}")

    def set_direction_mode(self, mode):
        self.direction_mode = mode
        print(f"[YoloDetector] 방향 모드 변경됨 → {mode}")

    # ------------------------ MP3 재생 ------------------------
    def _ensure_wav(self, path):
        if path.lower().endswith(".mp3"):
            wav_path = os.path.splitext(path)[0] + "_auto.wav"
            subprocess.run(
                ["ffmpeg", "-y", "-i", path, "-ar", "44100", "-ac", "2", wav_path],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            return wav_path
        return path

    def _play_mp3(self, mp3_file_path):
        try:
            fixed_path = self._ensure_wav(mp3_file_path)
            if not os.path.exists(fixed_path):
                print(f"[MP3] 파일 없음: {fixed_path}")
                return

            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=2048)

            pygame.mixer.music.load(fixed_path)
            pygame.mixer.music.set_volume(1.0)
            pygame.mixer.music.play()

            while pygame.mixer.music.get_busy():
                time.sleep(0.05)

        finally:
            with self.lock:
                self.mp3_played_for_movement["value"] = False

    # ------------------------ 얼굴 추적 ------------------------
    def _box_center(self, box):
        if box.xyxy.shape[0] > 0:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            return (x1 + x2) // 2, (y1 + y2) // 2
        return -1, -1

    def _is_same_face(self, box1_coords, box2):
        x1, y1, x2, y2 = box1_coords
        cx1, cy1 = (x1 + x2) // 2, (y1 + y2) // 2
        cx2, cy2 = self._box_center(box2)
        if cx2 == -1:
            return False
        return abs(cx1 - cx2) <= FACE_TRACK_THRESHOLD and abs(cy1 - cy2) <= FACE_TRACK_THRESHOLD

    def _set_first_face(self, box):
        if box.xyxy.shape[0] > 0:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            self.first_face_box_coords = (x1, y1, x2, y2)
            self.face_info["x"] = x1
            self.face_info["y"] = y1
            self.face_info["width"] = x2 - x1
            self.face_info["height"] = y2 - y1

    # ------------------------ 조건별 음성 --------------------------
    def _trigger_sensor_audio(self):
        temp = self.sensor_data.get("temperature", 0.0)
        hum = self.sensor_data.get("humidity", 0.0)
        rain_val = self.sensor_data.get("rain", 900)

        width = int(self.face_info.get("width", 0))
        height = int(self.face_info.get("height", 0))

        # 박스 크기 필터
        if width * height / 2 < 150:
            print(f"[Trigger] 박스 작음 (w:{width}, h:{height}) → 음성 X")
            return

        print(f"[Trigger] 센서 체크: T={temp}, H={hum}, R={rain_val}")

        if temp >= 27:
            threading.Thread(target=self._play_mp3, args=(MP3_PATH_HOT,), daemon=True).start()
        elif hum >= 80 or rain_val < 890:
            threading.Thread(target=self._play_mp3, args=(MP3_PATH_RAIN,), daemon=True).start()
        else:
            threading.Thread(target=self._play_mp3, args=(MP3_PATH_OUT,), daemon=True).start()

    # ------------------------ 메인 루프 ------------------------
    def run(self):
        if not self.cap or not self.cap.isOpened() or not self.model:
            print("[YOLO Thread] 초기화 실패")
            return

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            start_time = time.perf_counter()
            results = self.model(frame, imgsz=YOLO_IMG_SIZE, conf=CONF_THRESHOLD, verbose=False)
            end_time = time.perf_counter()

            boxes = results[0].boxes
            annotated = frame.copy()

            with self.lock:
                if boxes and boxes.xyxy.shape[0] > 0:

                    valid_boxes = [
                        box for box in boxes
                        if (box.xyxy[0][2] - box.xyxy[0][0]) >= MIN_FACE_SIZE and
                           (box.xyxy[0][3] - box.xyxy[0][1]) >= MIN_FACE_SIZE
                    ]

                    if valid_boxes:
                        matched = False

                        if self.first_face_box_coords:
                            for box in valid_boxes:
                                if self._is_same_face(self.first_face_box_coords, box):
                                    self._set_first_face(box)
                                    matched = True
                                    break

                        if not matched:
                            self._set_first_face(valid_boxes[0])
                            self.previous_face_center_x = None

                        if self.first_face_box_coords:
                            x1, y1, x2, y2 = self.first_face_box_coords
                            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

                            center_x = (x1 + x2) // 2

                            # --- 이동 감지 ---
                            if self.previous_face_center_x is not None:
                                movement_diff = center_x - self.previous_face_center_x

                                if abs(movement_diff) > 20 and not self.mp3_played_for_movement["value"]:
                                    self.mp3_played_for_movement["value"] = True

                                    # --- 방향 모드에 따른 퇴장 판정 ---
                                    if self.direction_mode == "left_out":
                                        is_exit = (movement_diff < 0)
                                    else:  # right_out
                                        is_exit = (movement_diff > 0)

                                    # --- 입장 ---
                                    if not is_exit:
                                        print(f"[Movement] 입장 → in.wav")
                                        threading.Thread(target=self._play_mp3, args=(MP3_PATH_IN,), daemon=True).start()

                                    # --- 퇴장 ---
                                    else:
                                        print(f"[Movement] 퇴장 → 센서 기반 음성")
                                        self._trigger_sensor_audio()

                            self.previous_face_center_x = center_x

                else:
                    self._reset_tracking()

                # FPS
                fps = 1.0 / (end_time - start_time) if end_time > start_time else 0
                self.fps_queue.append(fps)

                self.latest_annotated_frame["frame"] = annotated

            time.sleep(0.001)

    def _reset_tracking(self):
        self.first_face_box_coords = None
        self.face_info["width"] = 0
        self.face_info["height"] = 0
        self.previous_face_center_x = None
        self.mp3_played_for_movement["value"] = False

    def cleanup(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
        print("[YoloDetector] 종료 완료")
