import os
import cv2
import threading
import time
from ultralytics import YOLO
import pygame
import platform
import subprocess
from Config import *

if platform.system() == "Linux":
    os.environ["SDL_AUDIODRIVER"] = "alsa"
elif platform.system() == "Windows":
    os.environ["SDL_AUDIODRIVER"] = "directsound"
else:
    os.environ["SDL_AUDIODRIVER"] = "coreaudio"


class YoloDetector:
    def __init__(self, lock, latest_frame_var, face_info_var, mp3_played_var, fps_queue):
        self.lock = lock
        self.latest_annotated_frame = latest_frame_var
        self.face_info = face_info_var
        self.mp3_played_for_movement = mp3_played_var
        self.fps_queue = fps_queue
        self.direction_mode = "left_out"

        self.cap = None
        self.model = None
        self.first_face_box_coords = None
        self.previous_face_center_x = None

        self._init_resources()

    def set_direction_mode(self, mode):
        self.direction_mode = mode
        print(f"[YoloDetector] 방향 모드 변경됨: {mode}")

    def _init_resources(self):
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=2048)
                print("[Pygame] 믹서 초기화 성공 ✅")
        except pygame.error as e:
            print(f"[Pygame 오류] {e}")

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

        try:
            self.model = YOLO(YOLO_MODEL_PATH)
            print(f"[YOLO] 모델 로드 완료: {YOLO_MODEL_PATH}")
        except Exception as e:
            print(f"[YOLO 오류] 모델 로드 실패: {e}")

    def _ensure_wav(self, path):
        if path.lower().endswith(".mp3"):
            wav_path = os.path.splitext(path)[0] + "_auto.wav"
            subprocess.run(
                ["ffmpeg", "-y", "-i", path, "-ar", "44100", "-ac", "2", wav_path],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
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
            print(f"[MP3] 재생 시작: {fixed_path}")

            time.sleep(0.5)
            while pygame.mixer.music.get_busy():
                time.sleep(0.05)

            print("[MP3] 재생 완료 ✅")

        except Exception as e:
            print(f"[MP3 오류] {e}")

        finally:
            with self.lock:
                self.mp3_played_for_movement["value"] = False

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

    def run(self):
        if not self.cap or not self.cap.isOpened() or not self.model:
            print("[YOLO Thread] 초기화 실패. 종료.")
            return

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("[YOLO Thread] 카메라 오류.")
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
                        if (box.xyxy[0][2] - box.xyxy[0][0]) >= MIN_FACE_SIZE
                        and (box.xyxy[0][3] - box.xyxy[0][1]) >= MIN_FACE_SIZE
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

                            if self.previous_face_center_x is not None:
                                movement_diff = center_x - self.previous_face_center_x

                                if abs(movement_diff) > 3:
                                    if not self.mp3_played_for_movement["value"]:
                                        self.mp3_played_for_movement["value"] = True

                                        # 🔽 방향 모드 기반으로 MP3 결정
                                        if self.direction_mode == "left_out":
                                            if movement_diff < 0:
                                                mp3 = MP3_PATH_OUT
                                                print(f"[← 왼쪽 이동] OUT 방향")
                                            else:
                                                mp3 = MP3_PATH_IN
                                                print(f"[→ 오른쪽 이동] IN 방향")
                                        else:  # right_out
                                            if movement_diff > 0:
                                                mp3 = MP3_PATH_OUT
                                                print(f"[→ 오른쪽 이동] OUT 방향")
                                            else:
                                                mp3 = MP3_PATH_IN
                                                print(f"[← 왼쪽 이동] IN 방향")

                                        threading.Thread(
                                            target=self._play_mp3, args=(mp3,), daemon=True
                                        ).start()

                            self.previous_face_center_x = center_x
                    else:
                        self._reset_tracking()
                else:
                    self._reset_tracking()

                self.latest_annotated_frame["frame"] = annotated
                fps = 1.0 / (end_time - start_time) if (end_time - start_time) > 0 else 0
                self.fps_queue.append(fps)

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
            print("[YoloDetector] 카메라 해제 완료.")
        print("[YoloDetector] 종료 처리 완료.")
