import sys
import traceback
import threading
import cv2
import tkinter as tk
from PIL import Image, ImageTk
from collections import deque
import time
from bottle import Bottle, run, response
import json


from Config import *                 # 기존 Config 파일 사용
from SerialMonitor import SerialMonitor
from YoloDetector import YoloDetector


app_server = Bottle()

# -------------------- CORS ENABLE --------------------
@app_server.hook('after_request')
def enable_cors():
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Origin, Accept, Content-Type, X-Requested-With'

@app_server.route('/', method=['OPTIONS'])
def options_handler():
    return {}
# ------------------------------------------------------


shared_sensor_data = {"humidity":0.0, "temperature":0.0, "rain":900, "light":0, "lock": threading.Lock()}

@app_server.route('/', method='GET')
def send_sensor():
    with threading.Lock():
        data = {
            "temperature": shared_sensor_data.get("temperature", 0.0),
            "humidity": shared_sensor_data.get("humidity", 0.0),
            "rain": shared_sensor_data.get("rain", 900),
            "light": shared_sensor_data.get("light", 0),
            "timestamp": int(time.time())
        }
    response.content_type = 'application/json'
    return json.dumps(data)



def run_server():
    run(app_server, host='0.0.0.0', port=8000, debug=True)


class MainApp:
    # 영상 출력 크기
    FIXED_DISPLAY_WIDTH = 720
    FIXED_DISPLAY_HEIGHT = 540

    # UI 컬러셋
    BG_DARK = "#1E1E1E"
    CARD_BG = "#2B2B2B"
    TEXT_MAIN = "#E8E8E8"
    TEXT_SUB = "#A0A0A0"
    ACCENT_BLUE = "#4DA3FF"
    ACCENT_GREEN = "#77DD77"
    ACCENT_ORANGE = "#FFB347"
    ACCENT_RED = "#FF6961"

    INFO_PANEL_WIDTH = 300

    def __init__(self, master):
        self.master = master
        master.title("Smart Monitoring System")
        master.configure(bg=self.BG_DARK)

        # 공유 데이터 (스레드 안전을 위해 lock 사용)
        self.lock = threading.Lock()
        self.latest_annotated_frame = {'frame': None}
        # sensor_data 기본값: rain 기본은 900(맑음)
        self.sensor_data = {"humidity": 0.0, "temperature": 0.0, "light": 0, "rain": 900}
        self.face_info = {"x": 0, "y": 0, "width": 0, "height": 0}
        self.mp3_played_for_movement = {'value': False}
        self.fps_queue = deque(maxlen=10)
        self.direction_mode = tk.StringVar(value="left_out")

        # 컴포넌트 변수
        self.yolo_detector = None
        self.serial_monitor = None
        self.yolo_thread = None
        self.serial_thread = None

        # GUI 구성
        self._setup_gui()
        master.protocol("WM_DELETE_WINDOW", self.on_closing)

        # 백그라운드 컴포넌트 시작 (안전하게 예외 처리)
        try:
            self.serial_monitor = SerialMonitor(self.sensor_data)
            # run()가 블로킹이므로 스레드로 실행
            self.serial_thread = threading.Thread(target=self.serial_monitor.run, daemon=True)
            self.serial_thread.start()
        except Exception:
            print("[Error] SerialMonitor start failed:")
            traceback.print_exc()

        try:
            self.yolo_detector = YoloDetector(
                self.lock,
                self.latest_annotated_frame,
                self.face_info,
                self.mp3_played_for_movement,
                self.fps_queue,
                self.sensor_data
            )
            self.yolo_thread = threading.Thread(target=self.yolo_detector.run, daemon=True)
            self.yolo_thread.start()
        except Exception:
            print("[Error] YoloDetector start failed:")
            traceback.print_exc()

        # 초기 버튼 상태 반영
        self.update_direction_buttons()

        # 주기적으로 GUI 업데이트 시작
        self.update_gui()

    # ---------------- GUI 구성 ----------------
    def _setup_gui(self):
        # Video frame (left)
        self.video_frame = tk.Frame(self.master, bg=self.BG_DARK)
        self.video_frame.grid(row=0, column=0, padx=12, pady=12)

        self.video_label = tk.Label(
            self.video_frame,
            bg="#000000",
            width=self.FIXED_DISPLAY_WIDTH,
            height=self.FIXED_DISPLAY_HEIGHT
        )
        self.video_label.pack()

        # Info panel (right)
        self.info_panel = tk.Frame(self.master, bg=self.BG_DARK)
        self.info_panel.grid(row=0, column=1, sticky="ns", padx=10, pady=12)

        # System card
        self.system_card = self._create_card(self.info_panel, "📊 System Performance")
        self.fps_var = tk.StringVar(value="N/A")
        self.face_size_var = tk.StringVar(value="N/A")
        self._add_kv_row(self.system_card, "FPS", self.fps_var, self.ACCENT_GREEN)
        self._add_kv_row(self.system_card, "Face Size", self.face_size_var, self.ACCENT_ORANGE)

        # Sensor card
        self.sensor_card = self._create_card(self.info_panel, "🌡️ Sensor Data")
        self.hum_var = tk.StringVar(value="0.0%")
        self.temp_var = tk.StringVar(value="0.0°C")
        self.light_var = tk.StringVar(value="0 Lux")
        self.rain_var = tk.StringVar(value="Clear")
        self._add_kv_row(self.sensor_card, "Humidity", self.hum_var, self.TEXT_MAIN)
        self._add_kv_row(self.sensor_card, "Temperature", self.temp_var, self.ACCENT_RED)
        self._add_kv_row(self.sensor_card, "Light", self.light_var, self.ACCENT_ORANGE)
        self._add_kv_row(self.sensor_card, "Rain", self.rain_var, self.ACCENT_BLUE)

        # Direction card
        self.direction_card = self._create_card(self.info_panel, "🔄 Direction Setting")
        # 버튼 인스턴스를 저장해서 상태에 따라 스타일을 바꿀 수 있게 함
        self.btn_left = tk.Button(self.direction_card, text="Left = OUT", command=lambda: self.set_outside("left"),
                             bg=self.ACCENT_BLUE, fg="white", relief="flat", font=("Helvetica", 12, "bold"))
        self.btn_left.pack(fill="x", padx=10, pady=6)
        self.btn_right = tk.Button(self.direction_card, text="Right = OUT", command=lambda: self.set_outside("right"),
                              bg=self.CARD_BG, fg=self.TEXT_MAIN, relief="flat", font=("Helvetica", 12))
        self.btn_right.pack(fill="x", padx=10, pady=6)

    def _create_card(self, parent, title):
        frame = tk.Frame(parent, bg=self.CARD_BG, bd=1, relief="flat", highlightthickness=0)
        frame.pack(fill="x", pady=10)
        tk.Label(frame, text=title, bg=self.CARD_BG, fg=self.ACCENT_BLUE,
                 font=("Helvetica", 14, "bold"), anchor="w", padx=10, pady=6).pack(fill="x")
        return frame

    def _add_kv_row(self, parent, key, var, color):
        row = tk.Frame(parent, bg=self.CARD_BG)
        row.pack(fill="x", padx=10, pady=4)
        tk.Label(row, text=f"{key}:", bg=self.CARD_BG, fg=self.TEXT_SUB, font=("Helvetica", 11)).pack(side="left")
        tk.Label(row, textvariable=var, bg=self.CARD_BG, fg=color, font=("Helvetica", 11, "bold")).pack(side="right")

    def set_outside(self, side):
        if side == "left":
            new_mode = "left_out"
        else:
            new_mode = "right_out"
        self.direction_mode.set(new_mode)
        # 디텍터에도 모드 전달
        if self.yolo_detector:
            try:
                self.yolo_detector.set_direction_mode(new_mode)
            except Exception:
                traceback.print_exc()
        # 버튼 스타일 갱신
        self.update_direction_buttons()

    def update_direction_buttons(self):
        # 현재 direction_mode에 맞춰 버튼 스타일을 바꿈
        mode = self.direction_mode.get()
        if mode == "left_out":
            # 왼쪽이 OUT이면 왼쪽 버튼 하이라이트
            self.btn_left.config(bg=self.ACCENT_GREEN, fg="white", font=("Helvetica", 12, "bold"))
            self.btn_right.config(bg=self.CARD_BG, fg=self.TEXT_MAIN, font=("Helvetica", 12, "normal"))
        else:
            # 오른쪽이 OUT이면 오른쪽 버튼 하이라이트
            self.btn_right.config(bg=self.ACCENT_GREEN, fg="white", font=("Helvetica", 12, "bold"))
            self.btn_left.config(bg=self.CARD_BG, fg=self.TEXT_MAIN, font=("Helvetica", 12, "normal"))

    # ---------------- GUI 업데이트 ----------------
    def update_gui(self):
        try:
            with self.lock:
                frame = self.latest_annotated_frame.get('frame')
                if frame is not None:
                    try:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        img = Image.fromarray(rgb)
                        ratio = min(self.FIXED_DISPLAY_WIDTH / img.width, self.FIXED_DISPLAY_HEIGHT / img.height)
                        img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.Resampling.LANCZOS)
                        imgtk = ImageTk.PhotoImage(img)
                        self.video_label.imgtk = imgtk
                        self.video_label.config(image=imgtk)
                    except Exception:
                        # 이미지 변환 중 오류가 나면 무시하고 다음 루프로
                        traceback.print_exc()

                # FPS
                avg_fps = sum(self.fps_queue) / len(self.fps_queue) if self.fps_queue else 0
                self.fps_var.set(f"{avg_fps:.1f}" if avg_fps > 0 else "N/A")

                # Face size
                if self.face_info.get('width', 0) > 0:
                    self.face_size_var.set(f"{self.face_info['width']}x{self.face_info['height']}")
                else:
                    self.face_size_var.set("N/A")

                # Sensor values (safe formatting)
                try:
                    self.hum_var.set(f"{self.sensor_data.get('humidity', 0.0):.1f}%")
                    self.temp_var.set(f"{self.sensor_data.get('temperature', 0.0):.1f}°C")
                    self.light_var.set(f"{int(self.sensor_data.get('light', 0))} Lux")
                    rain_val = self.sensor_data.get('rain', 900)
                    # rain 판정: rain < 800 -> 비로 판단
                    self.rain_var.set("Rain" if rain_val < 890 else "Clear")
                except Exception:
                    traceback.print_exc()

        except Exception:
            traceback.print_exc()
        finally:
            # schedule next update
            self.master.after(FPS_UPDATE_INTERVAL_MS, self.update_gui)

    # ---------------- 종료 ----------------
    def on_closing(self):
        print("[MainApp] Shutting down...")
        try:
            if self.yolo_detector:
                self.yolo_detector.cleanup()
        except Exception:
            traceback.print_exc()
        try:
            if self.serial_monitor:
                self.serial_monitor.cleanup()
        except Exception:
            traceback.print_exc()
        self.master.destroy()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    root = tk.Tk()
    # 윈도우 크기 설정: 영상 + 우측 패널 + 여백
    initial_width = MainApp.FIXED_DISPLAY_WIDTH + MainApp.INFO_PANEL_WIDTH + 60
    initial_height = MainApp.FIXED_DISPLAY_HEIGHT + 40
    root.geometry(f"{initial_width}x{initial_height}")
    root.resizable(False, False)
    app = MainApp(root)

    shared_sensor_data = app.sensor_data

    root.mainloop()
