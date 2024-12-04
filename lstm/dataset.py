import numpy as np
import pandas as pd
from datetime import datetime
import torch
from torch.utils.data import Dataset

train = pd.read_csv('/home/s312657018/TBrain/data-preprocess/AvgDATA.csv')
mean = train['Power(mW)'].mean()
std = train['Power(mW)'].std()

class WeatherDataset(Dataset):
    def __init__(self, features, labels=None):
        # 保存特征均值和标准差，用于反标准化
        self.feature_mean = features.mean(axis=0)  # 每列特征的均值
        self.feature_std = features.std(axis=0)   # 每列特征的标准差

        # 特征标准化
        self.features = (features - self.feature_mean) / self.feature_std
        self.labels = labels
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        if self.labels is not None:
            return self.features[idx], self.labels[idx]
        else:
            return self.features[idx]
    
    def inverse_transform_labels(self, predictions, target_feature_index=0):
        return predictions * std + mean
    
def create_lstm_dataset(X, y, look_back, device):
    X_lstm, y_lstm = [], []
    for i in range(len(X)):
        if i < look_back:
            # 数据不足 look_back，使用前几笔数据填充
            padded_X = X.iloc[:i+1].values
            padded_X = np.pad(padded_X, ((look_back - len(padded_X), 0), (0, 0)), mode='edge')  # 用最前面的值填充
        else:
            # 正常获取窗口
            padded_X = X.iloc[i-look_back+1:i+1].values

        X_lstm.append(padded_X)
        if i < len(y):
            y_lstm.append(y[i])    
    
    X_lstm = torch.tensor(np.array(X_lstm), dtype=torch.float32).to(device)
    y_lstm = torch.tensor(np.array(y_lstm), dtype=torch.float32).to(device)

    return X_lstm, y_lstm

def create_lstm_dataset_test(X, look_back, device):
    X_lstm  = []
    for i in range(len(X)):
        if i < look_back:
            # 数据不足 look_back，使用前几笔数据填充
            padded_X = X.iloc[:i+1].values
            padded_X = np.pad(padded_X, ((look_back - len(padded_X), 0), (0, 0)), mode='edge')  # 用最前面的值填充
        else:
            # 正常获取窗口
            padded_X = X.iloc[i-look_back+1:i+1].values

        X_lstm.append(padded_X)  
    
    X_lstm = torch.tensor(np.array(X_lstm), dtype=torch.float32).to(device)

    return X_lstm

def process_serial(df, serial_column_name):
    # Convert serial to datetime format
    df['Serial'] = df[serial_column_name]
    df['Datetime'] = df[serial_column_name].astype(str).apply(lambda x: datetime.strptime(x[:12], '%Y%m%d%H%M'))
    df['Year'] = df['Datetime'].dt.year
    df['Month'] = df['Datetime'].dt.month
    df['Day'] = df['Datetime'].dt.day
    df['Hour'] = df['Datetime'].dt.hour
    df['Minute'] = df['Datetime'].dt.minute
    df['Station_ID'] = df[serial_column_name].astype(str).str[-2:].astype(int)  # 提取後兩碼作為序列代號
    # df['Timestamp'] = df['Timestamp'].astype(int) / 10**9  # Convert to Unix timestamp for numeric input
    
    if 'WindSpeed(m/s)' in df.columns and 'Sunlight(Lux)' in df.columns:
        df['WindSpeed(m/s)_x_Sunlight(Lux)'] = df['WindSpeed(m/s)'] * df['Sunlight(Lux)']
    if 'Temperature(°C)' in df.columns and 'Sunlight(Lux)' in df.columns:
        df['Temperature(°C)_x_Sunlight(Lux)'] = df['Temperature(°C)'] * df['Sunlight(Lux)']
 
    return df
