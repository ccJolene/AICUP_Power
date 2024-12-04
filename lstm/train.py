import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import wandb
from Configs import Config
from dataset import WeatherDataset, create_lstm_dataset, create_lstm_dataset_test, process_serial
from torch.utils.data import DataLoader
from sklearn.ensemble import RandomForestRegressor
import os
from tqdm import tqdm

def run_one_epoch(model, data_loader, optimizer, scheduler, criterion, device, phase='train'):
    if phase == 'train':
        model.train()
    else:
        model.eval()
        
    epoch_loss = .0
    predictions, ground_truths = [], []
    
    with torch.set_grad_enabled(phase == 'train'):
        for batch in data_loader:
            # 判断 batch 是否有 labels
            if isinstance(batch, (tuple, list)) and len(batch) == 2:
                features, labels = batch
                labels = labels.to(device=device, dtype=torch.float)
            else:
                features = batch
                labels = None
            
            features = features.to(device=device, dtype=torch.float)
            preds = model(features)
            preds = preds.view(-1)  # 将 preds 改变形状
            
            if labels is not None:
                loss = criterion(preds, labels)
                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                epoch_loss += loss.item()
                ground_truths.append(labels.detach().cpu().numpy())
            
            predictions.append(preds.detach().cpu().numpy())  
    
    predictions = np.concatenate(predictions, axis=0)
    if labels is not None:
        ground_truths = np.concatenate(ground_truths, axis=0)
        mae = mean_absolute_error(ground_truths, predictions)
        r2 = r2_score(ground_truths, predictions)
        print(f'Mean Absolute Error (MAE): {mae:.4f}')
        print(f'R² Score: {r2:.4f}')
        return epoch_loss, mae, r2, predictions, ground_truths
    else:
        return predictions
    

def plotCurve(history, epochs, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = ["mse", "mae", "r2"]
    titles = ["Loss (MSE)", "MAE", "R²"]

    for idx, ax in enumerate(axes):
        metric = metrics[idx]

        ax.plot(range(epochs), history['train'][metric], label='Training', color='blue')
        ax.plot(range(epochs), history['valid'][metric], label='Validation', color='orange', linestyle='--')

        ax.set_title(titles[idx])
        ax.set_xlabel('Epochs')
        ax.set_ylabel(metric.upper())
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, format='png')
    print(f"Plot saved to {output_path}")
    
    plt.show()

def testResults(X, test_data_path, features, model, pth_path, optimizer, scheduler, criterion, forecast_num=48, LookBackNum=12):
    checkpoint = torch.load(pth_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(Config.device)  
    model.eval()
    
    # 加载测试数据
    test_data = pd.read_csv(test_data_path)
    test_data = process_serial(test_data, '序號')
    
    # 补全测试数据的缺失特征
    time_features = ['Month', 'Day', 'Hour', 'Minute', 'Station_ID']
    target_columns = [col for col in features if col not in time_features]
    train_features = X[time_features]  # 假设 X 是您训练时使用的数据
    models = {}
    for target in target_columns:
        rf_model = RandomForestRegressor(random_state=42, n_estimators=200)
        rf_model.fit(train_features, X[target])
        models[target] = rf_model
        test_data[target] = rf_model.predict(test_data[time_features])
    
    # 确保特征与训练时一致
    test_features = test_data[features]
    X_test = create_lstm_dataset_test(test_features, look_back=LookBackNum, device=Config.device)
    test_dataset = WeatherDataset(X_test)
    test_loader = DataLoader(test_dataset, batch_size=Config.batch_size)
    
    # 运行一轮测试并接收预测结果
    test_preds = run_one_epoch(model, test_loader, optimizer, 
                                           scheduler, criterion, Config.device, phase='test')
    
    # 反标准化预测值
    test_preds_original = test_dataset.inverse_transform_labels(test_preds)
    test_preds_original = np.round(test_preds_original, 2)
    
    # 保存结果
    result_df = pd.DataFrame({
        '序號': test_data.iloc[:, 0],
        '答案': test_preds_original.flatten()
    })
    result_df.to_csv('test_results.csv', index=False)
    print("Test results saved to 'test_results.csv'")
    
    score = abs(test_data['答案'] - test_preds_original.flatten()).sum()
    print(f'Score: {score}')
    return test_preds


# def testResults(X, test_data_path, features, model, pth_path, optimizer, scheduler, criterion, forecast_num=48, LookBackNum=12):
#     """
#     测试函数，使用转换公式对测试数据进行补植，并运行预测。
    
#     Parameters:
#         - X: 训练数据
#         - test_data_path: 测试数据文件路径
#         - features: 特征列表
#         - model: LSTM 模型
#         - pth_path: 模型权重路径
#         - transformation_params: 转换参数 DataFrame
#         - optimizer, scheduler, criterion: PyTorch 优化器、学习率调度器和损失函数
#         - forecast_num: 预测步长
#         - LookBackNum: 回看步长
#     """

#     # 加载模型检查点
#     checkpoint = torch.load(pth_path)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     model.to(Config.device)
#     model.eval()
    
#     # 加载测试数据
#     test_data = pd.read_csv(test_data_path)
#     test_data = process_serial(test_data, '序號')
#     weather_data = pd.read_csv('/home/s312657018/TBrain/data-preprocess/weather_data.csv')
#     weather_data['Datetime'] = pd.to_datetime(weather_data['Datetime'])
#     transformation_params = pd.read_csv('/home/s312657018/TBrain/data-preprocess/transformation_params.csv')

#     # 补全测试数据的缺失特征
#     merged_data = pd.merge_asof(
#         test_data.sort_values('Datetime'),
#         weather_data.sort_values('Datetime'),
#         on='Datetime',
#         direction='nearest'
#     )

#     # 转换函数
#     def apply_transformation(row, params):
#         """
#         根据 Station_ID 和特征名应用线性转换。
#         """
#         station_id = row['Station_ID']
#         transformed_values = {}
#         for feature in ['Pressure(hpa)', 'Temperature(°C)', 'Humidity(%)']:
#             station_params = params[(params['Station_ID'] == station_id) & (params['Feature'] == feature)]
#             if not station_params.empty:
#                 coef = station_params['Coefficient'].values[0]
#                 intercept = station_params['Intercept'].values[0]
#                 transformed_values[feature] = row[feature] * coef + intercept
#                 # print(feature, station_id, coef, intercept)
#             else:
#                 transformed_values[feature] = np.nan  # 如果没有对应参数，则填充 NaN
#         return pd.Series(transformed_values)

#     transformed_features = merged_data.apply(lambda row: apply_transformation(row, transformation_params), axis=1)

#     # 将转换后的特征合并回测试数据
#     for feature in ['Pressure(hpa)', 'Temperature(°C)', 'Humidity(%)']:
#         test_data[feature] = transformed_features[feature]
    
#     time_features = ['Month', 'Day', 'Hour', 'Minute', 'Station_ID', 'Pressure(hpa)', 'Temperature(°C)', 'Humidity(%)']
#     target_columns = [col for col in features if col not in time_features]
#     train_features = X[time_features]  # 假设 X 是您训练时使用的数据
#     models = {}
#     for target in target_columns:
#         rf_model = RandomForestRegressor(random_state=42, n_estimators=200)
#         rf_model.fit(train_features, X[target])
#         models[target] = rf_model
#         test_data[target] = rf_model.predict(test_data[time_features])
    
#     print(test_data)
#     # 确保特征与训练时一致
#     test_features = test_data[features]
#     X_test = create_lstm_dataset_test(test_features, look_back=LookBackNum, device=Config.device)
#     test_dataset = WeatherDataset(X_test)
#     test_loader = DataLoader(test_dataset, batch_size=Config.batch_size)
    
#     # 运行一轮测试并接收预测结果
#     test_preds = run_one_epoch(model, test_loader, optimizer, 
#                                            scheduler, criterion, Config.device, phase='test')
    
#     # 反标准化预测值
#     test_preds_original = test_dataset.inverse_transform_labels(test_preds)
#     test_preds_original = np.round(test_preds_original, 2)
    
#     # 保存结果
#     result_df = pd.DataFrame({
#         '序號': test_data.iloc[:, 0],
#         '答案': test_preds_original.flatten()
#     })
#     result_df.to_csv('test_results.csv', index=False)
#     print("Test results saved to 'test_results.csv'")
    
#     score = abs(test_data['答案'] - test_preds_original.flatten()).sum()
#     print(f'Score: {score}')
    
#     return test_preds
