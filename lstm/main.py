import sys
sys.path.append('code')
import pandas as pd
from tqdm import tqdm
import glob

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from model import LSTMRegressor, StackedLSTM, StackedGRU, LSTMRegressor_4layers
from Configs import Config, seed_everything
from dataset import WeatherDataset, create_lstm_dataset
import importlib
import dataset
import train
importlib.reload(train)
from train import run_one_epoch, plotCurve, testResults

data = pd.read_csv('data-preprocess/AvgDATA.csv')
features = ['WindSpeed(m/s)', 'Pressure(hpa)', 'Temperature(°C)', 'Humidity(%)', 'Sunlight(Lux)', 'Month', 'Day', 'Hour', 'Minute', 'Station_ID']
# features = ['WindSpeed(m/s)', 'Pressure(hpa)', 'Temperature(°C)', 'Humidity(%)', 'Month', 'Day', 'Hour', 'Minute', 'Station_ID']
# features = ['WindSpeed(m/s)', 'Temperature(°C)', 'Sunlight(Lux)',
#         'WindSpeed(m/s)_x_Sunlight(Lux)', 'Temperature(°C)_x_Sunlight(Lux)',
#        'Month', 'Day', 'Hour', 'Minute', 'Station_ID']
X = data[features]
y = data['Power(mW)']

LookBackNum = 12  # LSTM 往前看的筆數
ForecastNum = 48  # 預測筆數

X_lstm, y_lstm = create_lstm_dataset(X, y, look_back=LookBackNum, device=Config.device)
X_train, X_valid, y_train, y_valid = train_test_split(X_lstm, y_lstm, test_size=0.2, random_state=Config.seed)
print(f"X_train shape: {X_train.shape}, X_valid shape: {X_valid.shape}")

train_dataset = WeatherDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=Config.batch_size)
valid_dataset = WeatherDataset(X_valid, y_valid)
valid_loader = DataLoader(valid_dataset, batch_size=Config.batch_size)

# model = LSTMRegressor(input_size=10,  hidden_size1=512, hidden_size2=64, num_layers=2, output_size=1, dropout=0.2).to(Config.device)
# model = StackedLSTM(input_size=10, hidden_size=128, num_layers=2, output_size=1, dropout=0.2).to(Config.device)
model = StackedGRU(input_size=len(features), hidden_size=128, num_layers=2, output_size=1, dropout=0.2).to(Config.device)
# model = LSTMRegressor_4layers(input_size=len(features), hidden_sizes=[50, 60, 80, 120], dropout_rates = [0.2, 0.3, 0.4, 0.5]).to(Config.device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=Config.lr,
        epochs=Config.epochs,
        steps_per_epoch=train_loader.__len__()
    )

def main():
    seed_everything(Config.seed)
    
    best_val_loss = float('inf')
    history = {'train':{'mse': [], 'mae': [], 'r2': []},
                'valid': {'mse': [], 'mae': [], 'r2': []}}
    
    for epoch in tqdm(range(Config.epochs)):
        print('Training...')
        train_loss, train_mae, train_r2, train_preds, train_gts = run_one_epoch(model, train_loader, optimizer, 
                                                                                scheduler, criterion, Config.device, phase='train')
        print('-' * 50)
        print('Validation...')
        valid_loss, valid_mae, valid_r2, valid_preds, valid_gts = run_one_epoch(model, valid_loader, optimizer, 
                                                                                scheduler, criterion, Config.device, phase='valid')
        print('-' * 50)

        history['train']['mse'].append(train_loss)
        history['train']['mae'].append(train_mae)
        history['train']['r2'].append(train_r2)
        history['valid']['mse'].append(valid_loss)
        history['valid']['mae'].append(valid_mae)
        history['valid']['r2'].append(valid_r2)
        
        print(f'Epoch[{epoch+1}/{Config.epochs}]')
        print(f'Train-MSELoss: {train_loss:.4f}, MAE: {train_mae:.4f}, Train r2: {train_r2:.4f}')
        print(f'Valid-MSELoss: {valid_loss:.4f}, MAE: {valid_mae:.4f}, Valid r2: {valid_r2:.4f}') 
        print(f'LR: {optimizer.state_dict()["param_groups"][0]["lr"]:.6f}')
        
        if valid_loss < best_val_loss:
            best_val_loss = valid_loss

            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch
            }, f'LSTM_checkpoint.pth')
            print(f"New best model saved with valid loss: {valid_loss}")
            
        print("*"*50)
        
    #     wandb.log({
    #         "epoch": epoch + 1,
    #         "train_loss": train_loss,
    #         "train_mae": train_mae,
    #         "train_r2": train_r2,
    #         "valid_loss": valid_loss,
    #         "valid_mae": valid_mae,
    #         "valid_r2": valid_r2
    #     })
    
    # # 完成後結束 W&B 會話
    # wandb.finish()
    best_model = model.to(Config.device).float()
    checkpoint = torch.load(f'LSTM_checkpoint.pth')
    best_model.load_state_dict(checkpoint['model_state_dict'])
    
    plotCurve(history, Config.epochs, output_path='figure/test.png')
    
if __name__ == '__main__':
    # main()
    testResults(X, '/home/s312657018/TBrain/示範程式/LSTM(比賽用)/ExampleTestData/upload.csv', features, 
                model, 'LSTM_checkpoint.pth', optimizer, scheduler, criterion)
