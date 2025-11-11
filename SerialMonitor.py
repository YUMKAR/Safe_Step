# SerialMonitor.py

import threading
import time
import serial
import re
from Config import *

class SerialMonitor:
    def __init__(self, sensor_data_var):
        """SerialMonitor 클래스 초기화."""
        self.sensor_data = sensor_data_var
        self.ser = None

    def run(self):
        """시리얼 데이터 수신 메인 루프."""
        try:
            self.ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1) 
            time.sleep(2)
            print(f"[Serial] {SERIAL_PORT} 연결 성공. 데이터 수신 시작...\n")

            # 데이터 파싱 패턴: H: % , T: C , L: Lux
            pattern = re.compile(r"H:\s*(\d+\.?\d*)%\s*,\s*T:\s*(\d+\.?\d*)C\s*,\s*L:\s*(\d+)")

            while True:
                line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                if not line:
                    time.sleep(0.1) 
                    continue

                match = pattern.search(line)
                if match:
                    try:
                        self.sensor_data['humidity'] = float(match.group(1))
                        self.sensor_data['temperature'] = float(match.group(2))
                        self.sensor_data['light'] = int(match.group(3))
                    except ValueError as ve:
                        print(f"[Serial 파싱 오류] 숫자 변환 실패: {ve}, 데이터: {line}")
                
        except serial.SerialException as se:
            print(f"[Serial 연결 오류] 포트 {SERIAL_PORT} 열기 실패. 오류: {se}")
        except Exception as e:
            print(f"[Serial 기타 오류] {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        """종료 시 시리얼 포트를 닫습니다."""
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("[SerialMonitor] 포트 닫힘.")