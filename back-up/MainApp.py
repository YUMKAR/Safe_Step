
import cv2
import threading
import time
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from collections import deque

# Config.py, YoloDetector.py, SerialMonitor.py에서 필요한 클래스 임포트
# 실제 실행을 위해서는 이 파일들이 필요합니다.
from Config import * 
from YoloDetector import YoloDetector
from SerialMonitor import SerialMonitor


class MainApp:
    # 비디오 패널에 사용할 고정 크기를 정의합니다.
    FIXED_DISPLAY_WIDTH = 720
    FIXED_DISPLAY_HEIGHT = 540
    
    # GUI 스타일 상수
    BG_DARK = "#212F3D"      # 전체 배경
    FRAME_BG = "#2C3E50"     # 프레임 배경 (약간 밝게)
    FG_WHITE = "#ECF0F1"     # 일반 텍스트
    ACCENT_BLUE = "#1E90FF"  # 액센트 색상
    INFO_PANEL_WIDTH = 280   # 정보 패널 고정 폭

    def __init__(self, master):
        """MainApp GUI 및 시스템 초기화."""
        self.master = master
        master.title("스마트 모니터링 시스템")
        master.configure(bg=self.BG_DARK)
        
        # 1. 스레드 공유 변수 초기화
        self.lock = threading.Lock()
        self.latest_annotated_frame = {'frame': None}
        self.sensor_data = {"humidity": 0.0, "temperature": 0.0, "light": 0}
        self.face_info = {"x": 0, "y": 0, "width": 0, "height": 0}
        self.mp3_played_for_movement = {'value': False}
        self.fps_queue = deque(maxlen=10)
        self.direction_mode = tk.StringVar(value="left_out") # 방향 모드 변수
        
        # 2. 모니터링 클래스 인스턴스화
        self.yolo_detector = YoloDetector(
            self.lock, self.latest_annotated_frame, self.face_info, 
            self.mp3_played_for_movement, self.fps_queue
        )
        self.serial_monitor = SerialMonitor(self.sensor_data)
        
        # 3. GUI 위젯 초기화
        self._setup_gui()

        # 4. 종료 핸들러 설정
        master.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # 5. 스레드 시작
        threading.Thread(target=self.yolo_detector.run, daemon=True).start()
        threading.Thread(target=self.serial_monitor.run, daemon=True).start()

        # 6. GUI 업데이트 루프 시작
        self.update_gui()


    def _setup_gui(self):
        """Tkinter GUI 레이아웃을 고정 크기로 설정합니다 (grid 기반)."""
        
        FONT_STYLE = ("Helvetica", 11)
        HEADER_FONT = ("Helvetica", 14, "bold")
        DATA_FONT = ("Helvetica", 13, "bold") 
        
        # --- 0. 고정 크기 레이아웃 설정 (가중치 0) ---
        self.master.grid_columnconfigure(0, weight=0) # 비디오 패널
        self.master.grid_columnconfigure(1, weight=0) # 정보 패널
        self.master.grid_rowconfigure(0, weight=0)
        
        # --- 1. 비디오 패널 (0열) ---
        # 고정된 width와 height를 설정하여 크기를 명확히 지정
        self.video_label = tk.Label(self.master, bg="#000000", padx=5, pady=5,
                                    width=self.FIXED_DISPLAY_WIDTH, 
                                    height=self.FIXED_DISPLAY_HEIGHT)
        self.video_label.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        # --- 2. 정보 프레임 (1열) ---
        self.info_frame = tk.Frame(self.master, bg=self.BG_DARK, 
                                   width=self.INFO_PANEL_WIDTH, padx=5, pady=5)
        self.info_frame.grid(row=0, column=1, sticky="ns", padx=10, pady=10)
        # 프레임이 내용물 크기에 따라 줄어드는 것을 방지
        self.info_frame.pack_propagate(False) 
        self.info_frame.grid_columnconfigure(0, weight=1) # 내부 위젯은 확장 가능
        
        # --- 2.1. 시스템 정보 LabelFrame ---
        performance_frame = tk.LabelFrame(self.info_frame, text="💻 System Performance", font=HEADER_FONT,
                                            fg="#A9CCE3", bg=self.FRAME_BG, bd=0, relief=tk.FLAT, padx=10, pady=10)
        performance_frame.pack(fill="x", pady=(0, 20)) 
        performance_frame.grid_columnconfigure(1, weight=1)

        tk.Label(performance_frame, text="FPS:", font=FONT_STYLE, fg=self.FG_WHITE, bg=self.FRAME_BG).grid(row=0, column=0, sticky="w", padx=5, pady=4)
        self.fps_text = tk.StringVar(value="N/A")
        tk.Label(performance_frame, textvariable=self.fps_text, font=DATA_FONT, fg="#2ECC71", bg=self.FRAME_BG, anchor='e').grid(row=0, column=1, sticky="e", padx=5, pady=4)

        tk.Label(performance_frame, text="Face Size:", font=FONT_STYLE, fg=self.FG_WHITE, bg=self.FRAME_BG).grid(row=1, column=0, sticky="w", padx=5, pady=4)
        self.face_info_text = tk.StringVar(value="N/A")
        tk.Label(performance_frame, textvariable=self.face_info_text, font=DATA_FONT, fg="#F39C12", bg=self.FRAME_BG, anchor='e').grid(row=1, column=1, sticky="e", padx=5, pady=4)

        # --- 2.2. 센서 데이터 LabelFrame ---
        sensor_frame = tk.LabelFrame(self.info_frame, text="🌡️ Environment Sensor Data", font=HEADER_FONT,
                                            fg="#A9CCE3", bg=self.FRAME_BG, bd=0, relief=tk.FLAT, padx=10, pady=10)
        sensor_frame.pack(fill="x", pady=0)
        sensor_frame.grid_columnconfigure(1, weight=1)

        tk.Label(sensor_frame, text="Humidity:", font=FONT_STYLE, fg=self.FG_WHITE, bg=self.FRAME_BG).grid(row=0, column=0, sticky="w", padx=5, pady=4)
        self.humidity_text = tk.StringVar(value="0.0%")
        tk.Label(sensor_frame, textvariable=self.humidity_text, font=DATA_FONT, fg="#3498DB", bg=self.FRAME_BG, anchor='e').grid(row=0, column=1, sticky="e", padx=5, pady=4)

        tk.Label(sensor_frame, text="Temperature:", font=FONT_STYLE, fg=self.FG_WHITE, bg=self.FRAME_BG).grid(row=1, column=0, sticky="w", padx=5, pady=4)
        self.temperature_text = tk.StringVar(value="0.0°C")
        tk.Label(sensor_frame, textvariable=self.temperature_text, font=DATA_FONT, fg="#E74C3C", bg=self.FRAME_BG, anchor='e').grid(row=1, column=1, sticky="e", padx=5, pady=4)

        tk.Label(sensor_frame, text="Light:", font=FONT_STYLE, fg=self.FG_WHITE, bg=self.FRAME_BG).grid(row=2, column=0, sticky="w", padx=5, pady=4)
        self.light_text = tk.StringVar(value="0 Lux")
        tk.Label(sensor_frame, textvariable=self.light_text, font=DATA_FONT, fg="#F1C40F", bg=self.FRAME_BG, anchor='e').grid(row=2, column=1, sticky="e", padx=5, pady=4)
        
        # --- 2.3. 방향 설정 버튼 LabelFrame ---
        self.toggle_frame = tk.LabelFrame(self.info_frame, text="Direction Setting", font=HEADER_FONT,
                                          fg="#A9CCE3", bg=self.FRAME_BG, bd=0, relief=tk.FLAT, padx=10, pady=10)
        self.toggle_frame.pack(fill="x", pady=(20, 0)) 
        self.toggle_frame.grid_columnconfigure((0, 1), weight=1)
        
        # 버튼 생성
        self.left_toggle = tk.Button(
            self.toggle_frame, text="LEFT: OUT (기본)",
            command=lambda: self.set_outside("left"), width=15,
            relief=tk.FLAT, font=FONT_STYLE
        )
        self.left_toggle.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        self.right_toggle = tk.Button(
            self.toggle_frame, text="RIGHT: OUT",
            command=lambda: self.set_outside("right"), width=15,
            relief=tk.FLAT, font=FONT_STYLE
        )
        self.right_toggle.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        
        # 초기 스타일 및 모드 설정
        self.set_outside("left")


    def set_outside(self, side):
        """방향 설정 모드를 토글하고 YoloDetector에 업데이트를 알립니다."""
        
        # 버튼 스타일 재설정 함수
        def set_button_style(btn, is_active):
            btn.config(bg=self.ACCENT_BLUE if is_active else self.FRAME_BG, 
                      fg="white" if is_active else self.FG_WHITE)
            
        if side == "left":
            self.direction_mode.set("left_out")
            set_button_style(self.left_toggle, True)
            set_button_style(self.right_toggle, False)
            print("[MODE] 왼쪽이 OUT, 오른쪽이 IN 모드로 설정됨.")
        else:
            self.direction_mode.set("right_out")
            set_button_style(self.right_toggle, True)
            set_button_style(self.left_toggle, False)
            print("[MODE] 오른쪽이 OUT, 왼쪽이 IN 모드로 설정됨.")

        # YoloDetector에 변경된 모드 알림 (YoloDetector에 set_direction_mode 메서드가 있어야 함)
        self.yolo_detector.set_direction_mode(self.direction_mode.get())
        

    def update_gui(self):
        """주기적으로 GUI를 업데이트합니다."""
        
        with self.lock:
            # --- 비디오 업데이트 (화면 표시 핵심 코드) ---
            frame = self.latest_annotated_frame.get('frame')
            if frame is not None:
                # BGR을 RGB로 변환
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                
                # 고정된 디스플레이 크기에 맞춰 비율을 유지하며 리사이징
                display_width = self.FIXED_DISPLAY_WIDTH
                display_height = self.FIXED_DISPLAY_HEIGHT
                
                if img.size[0] > 0 and img.size[1] > 0:
                    img_width, img_height = img.size
                    # 비율 계산: 가로 및 세로 중 더 작은 비율을 선택하여 잘리지 않게 조정
                    ratio = min(display_width / img_width, display_height / img_height)
                    new_width = int(img_width * ratio)
                    new_height = int(img_height * ratio)
                    
                    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    imgtk = ImageTk.PhotoImage(image=img)
                    
                    self.video_label.imgtk = imgtk # 참조 유지
                    self.video_label.config(image=imgtk, 
                                            width=display_width, 
                                            height=display_height) # 라벨 크기를 고정 크기로 설정

            # --- 정보 텍스트 업데이트 ---
            avg_fps = sum(self.fps_queue) / len(self.fps_queue) if self.fps_queue else 0
            self.fps_text.set(f"{avg_fps:.1f}" if avg_fps > 0 else "N/A")
            
            if self.face_info['width'] > 0:
                self.face_info_text.set(f"{self.face_info['width']}x{self.face_info['height']} Pixels")
            else:
                self.face_info_text.set("N/A")
            
            self.humidity_text.set(f"{self.sensor_data['humidity']:.1f}%")
            self.temperature_text.set(f"{self.sensor_data['temperature']:.1f}°C")
            self.light_text.set(f"{self.sensor_data['light']} Lux") 

        # 다음 GUI 업데이트를 예약합니다.
        self.master.after(FPS_UPDATE_INTERVAL_MS, self.update_gui)

    def on_closing(self):
        """창이 닫힐 때 호출되며, 모든 자원을 해제합니다."""
        print("\n[MainApp] Application termination initiated.")
        
        self.yolo_detector.cleanup()
        self.serial_monitor.cleanup()

        self.master.destroy()
        cv2.destroyAllWindows()
        print("[MainApp] Program termination complete.")


if __name__ == "__main__":
    # 필수 전역 변수가 정의되지 않았을 경우를 대비하여 더미 클래스 및 변수 정의
    try:
        # Config 파일에서 FPS_UPDATE_INTERVAL_MS 가져오기
        _ = FPS_UPDATE_INTERVAL_MS
    except NameError:
        # Config.py가 로드되지 않은 경우 더미 값 설정
        FPS_UPDATE_INTERVAL_MS = 200
        
    try:
        # YoloDetector 및 SerialMonitor가 정의되지 않은 경우 더미 클래스 사용
        _ = YoloDetector
    except NameError:
        class DummyYoloDetector:
            def __init__(self, *args): pass
            def set_direction_mode(self, mode): print(f"Dummy Yolo Mode: {mode}")
            def run(self): print("Dummy Yolo Detector running...")
            def cleanup(self): print("Dummy Yolo Detector cleanup.")

        class DummySerialMonitor:
            def __init__(self, *args): pass
            def run(self): print("Dummy Serial Monitor running...")
            def cleanup(self): print("Dummy Serial Monitor cleanup.")

        YoloDetector = DummyYoloDetector
        SerialMonitor = DummySerialMonitor
        print("WARNING: YoloDetector or SerialMonitor not found. Using dummy components.")


    root = tk.Tk()
    
    # 고정된 비디오 크기 (720x540)와 정보 패널 크기 (280)를 고려하여 창 크기 설정
    initial_width = MainApp.FIXED_DISPLAY_WIDTH + MainApp.INFO_PANEL_WIDTH + 40 # 720 + 280 + 패딩 40
    initial_height = MainApp.FIXED_DISPLAY_HEIGHT + 20 # 540 + 패딩 20
    root.geometry(f"{initial_width}x{initial_height}")
    root.resizable(False, False) # 고정형이므로 창 크기 조절 방지

    app = MainApp(root)
    root.mainloop()
