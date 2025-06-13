import serial
import time
import pandas as pd
import os

# 설정 변수
"""수집 전 매번 offset 확인 필수"""
port = "COM4"  # 아두이노가 연결된 포트 (리눅스/Mac의 경우 "/dev/ttyUSB0" 등)
baud_rate = 115200  # 아두이노 시리얼 통신 속도
experiment_count = 24  # 총 실험 반복 횟수
experiment_duration = 3  # 실험 지속 시간 (초)
interval = 0.005  # 데이터 수집 간격 (초)

late_day = 0 #달걀상온경과일. 실험마다 수정 필요.
exp_num = 5 #실험(달걀) 번호. 실험마다 수정 필요.


csv_filename = (f"data/train/exp_{exp_num}_DAY{late_day}/DAY{late_day}_exp")  # 저장할 파일 이름 (실험 번호가 추가됨)

# 폴더가 존재하지 않으면 생성
os.makedirs(os.path.dirname(csv_filename), exist_ok=True)


iteration = int(experiment_duration / interval)  # 각 실험에서 수집할 데이터 개수

def clearInput():
    while( ser.in_waiting > 0 ):
        treshline = ser.readline()
    

# 시리얼 포트 열기
ser = serial.Serial(port, baud_rate, timeout=1)
time.sleep(3)  # 아두이노 초기화 대기

for exp in range(1, experiment_count + 1):
    
    time.sleep(1)  # 실험 간 대기 시간 (필요에 따라 조정)
    clearInput()
    
    
    print(f"exp {exp} start...")
    
    ser.write(b'8')  # 아두이노에서 "END" 신호를 인식하도록 설정
    # print("vibration end")
    
    clearInput()
    data_list = []
    start_time = time.time()

    for i in range(iteration):
        if ser.in_waiting > 0:
            try:
                line = ser.readline().decode("utf-8", errors="ignore").strip()
            except UnicodeDecodeError:
                print("decoding error appeared. ignore it.")
                continue

            values = line.split(",")
            if len(values) == 3:  # 예상한 데이터 개수와 일치하는 경우만 저장
                try:
                    data_list.append([float(v) for v in values])
                except ValueError:
                    pass  # 변환 불가능한 데이터 무시
        time.sleep(interval)  # 다음 데이터 수집까지 대기

    # 데이터 저장
    df = pd.DataFrame(data_list, columns=["gyro_x", "gyro_y", "gyro_z"])
    filename = f"{csv_filename}{exp}.csv"
    df.to_csv(filename, index=False)
    print(f"exp {exp} data successfully collected: {filename}")

    

# 시리얼 포트 닫기
ser.close()
print("all experiments completed.")
