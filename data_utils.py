import glob
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from model import InceptionNetwork
import matplotlib.pyplot  as plt
import os




def log_loss_to_csv(log_path, epoch, loss_value):
    # 첫 실행 시 헤더 포함하여 저장
    if not os.path.exists(log_path):
        df = pd.DataFrame(columns=["epoch", "loss"])
        df.to_csv(log_path, index=False)
    new_row = {"epoch": epoch,"loss": loss_value}
    df = pd.DataFrame([new_row])
    df.to_csv(log_path, mode='a', header=False, index=False)


def show_origin_data(data_paths, show_graph = True):
    for data_path in data_paths:
        print("data path: ",data_path)
        cnt = 0
        file_list = sorted(glob.glob(data_path + "/*.csv"))

        # 그래프 그리기
        for file in file_list:
            
            df = pd.read_csv(file)
            cnt += 1
            
            if(show_graph):
                plt.figure(figsize=(20, 6))
                
                # 각 실험 파일의 데이터 플로팅
                plt.plot(df.index, df["gyro_x"], label=f"{file.split('/')[-1]} - gyro_x", alpha=0.6)
                plt.plot(df.index, df["gyro_y"], label=f"{file.split('/')[-1]} - gyro_y", alpha=0.6)
                plt.plot(df.index, df["gyro_z"], label=f"{file.split('/')[-1]} - gyro_z", alpha=0.6)
                

                plt.xlabel("Time Step")
                plt.ylabel("Value")
                plt.title("Experimental Data Visualization")
                plt.legend()
                plt.grid(True)
                plt.show()
        print("files : ", cnt)
        
        
def compute_global_min_max(folder_list):
    
    all_data = []

    for folder_path in folder_list:
        file_list = glob.glob(folder_path + "/*.csv")
        for file in file_list:
            df = pd.read_csv(file)
            all_data.append(torch.tensor(df.values, dtype=torch.float32))

    concatenated = torch.cat(all_data, dim=0)  # (전체 시계열 합, 채널 수)
    global_min = concatenated.min(dim=0).values  # 채널별 최소
    global_max = concatenated.max(dim=0).values  # 채널별 최대
    return global_min, global_max       
        
class PaddedTimeSeriesDataset(Dataset):
    def __init__(self, folder_list,folder2label, global_min, global_max, regression = False):
        self.data = []
        self.labels = []
        self.global_min = global_min
        self.global_max = global_max
        self.folder2label = folder2label
        
        
        for folder_path in folder_list:
            file_list = sorted(glob.glob(folder_path + "/*.csv"))
            label = self.folder2label[folder_path]


            for file in file_list:
                df = pd.read_csv(file)
                tensor_data = torch.tensor(df.values, dtype=torch.float32)  # (시계열 길이, 채널)

                min_vals = self.global_min
                max_vals = self.global_max
                
            
                tensor_data = (tensor_data - min_vals) / (max_vals - min_vals + 1e-8)  # 분모 0 방지
                
                self.data.append(tensor_data)
                if regression:
                    self.labels.append(torch.tensor(label, dtype=torch.float32))
                else:
                    self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
    
    
    
def custom_pad_sequence(sequences):
    """ 각 시퀀스의 마지막 값을 패딩 값으로 사용하여 패딩 수행 """
    max_len = max(seq.shape[0] for seq in sequences)  # 가장 긴 시퀀스 길이

    padded_sequences = []
    for seq in sequences:
        mean_value = seq[-1:, :]  # (1, 채널 수) - 마지막 행(시간축 기준)
        pad_len = max_len - seq.shape[0]  # 필요한 패딩 길이
        
        if pad_len > 0:
            pad_tensor = mean_value.repeat(pad_len, 1)  # (pad_len, 채널) 형태로 패딩 생성
            seq = torch.cat([seq, pad_tensor], dim=0)  # 패딩 추가
        padded_sequences.append(seq)

    return torch.stack(padded_sequences)  # (batch_size, max_seq_len, 3)




def collate_fn(batch):
    """ (batch_size, channels, max_sequence_len) 형태로 변환 """
    sequences, labels = zip(*batch)  # 데이터와 레이블 분리
    sequences = custom_pad_sequence(sequences)
    sequences = sequences.permute(0, 2, 1)  # (batch_size, 3, max_seq_len)로 변경
    
    labels = torch.tensor(labels, dtype=torch.long)  # 레이블을 텐서로 변환
    
    return sequences, labels

def collate_fn_regression(batch):
    """ (batch_size, channels, max_sequence_len) 형태로 변환 """
    sequences, labels = zip(*batch)  # 데이터와 레이블 분리
    sequences = custom_pad_sequence(sequences)
    sequences = sequences.permute(0, 2, 1)  # (batch_size, 3, max_seq_len)로 변경
    
    labels = torch.stack(labels).float()  # 회귀용으로 float32 stack
   
    return sequences, labels




def show_normal_data(dataloader,label2name,show_graph = True):
    """정규화 데이터 확인"""
    cnt = 0

    for X_batch, y_batch in dataloader:
        X_train, y_train = X_batch, y_batch
        for i in range(4):
            if show_graph:
                plt.figure(figsize=(10,6))
                plt.plot(X_train[i][0].numpy(), label="gyro_x")
                plt.plot(X_train[i][1].numpy(), label="gyro_y")
                plt.plot(X_train[i][2].numpy(), label="gyro_z")
                plt.legend()
                plt.xlabel("Time Step")
                plt.ylabel("Value")
                plt.title("Experimental Nomalized Data")
                plt.grid()
                
                plt.show()
            print("label : ", label2name[int(y_train[i])])
            cnt += 1
        # print(X_train[0].shape)
        # break  # 첫 번째 배치만 실행하고 종료
        print("files : ",cnt)