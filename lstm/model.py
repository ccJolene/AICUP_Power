import torch
import torch.nn as nn

class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_layers, output_size, dropout=0.0):
        super(LSTMRegressor, self).__init__()
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size1, num_layers=num_layers, 
                             batch_first=True, dropout=dropout)
        self.lstm2 = nn.LSTM(input_size=hidden_size1, hidden_size=hidden_size2, num_layers=num_layers, 
                             batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.fc(x[:, -1, :])  # 取最後一個時間步的輸出
        return x
    
class StackedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        super(StackedLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0  
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.lstm(x)  
        x = self.fc(x[:, -1, :])  
        return x
    
class StackedGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        super(StackedGRU, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0  # 在多於 1 層時啟用 dropout
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.gru(x)  # GRU 層的輸出
        x = self.fc(x[:, -1, :])  # 使用最後一個時間步的輸出
        return x
    

class LSTMRegressor_4layers(nn.Module):
    def __init__(self, input_size, hidden_sizes, dropout_rates):
        super(LSTMRegressor_4layers, self).__init__()

        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_sizes[0], batch_first=True)
        self.dropout1 = nn.Dropout(dropout_rates[0])
        
        self.lstm2 = nn.LSTM(input_size=hidden_sizes[0], hidden_size=hidden_sizes[1], batch_first=True)
        self.dropout2 = nn.Dropout(dropout_rates[1])
        
        self.lstm3 = nn.LSTM(input_size=hidden_sizes[1], hidden_size=hidden_sizes[2], batch_first=True)
        self.dropout3 = nn.Dropout(dropout_rates[2])
        
        self.lstm4 = nn.LSTM(input_size=hidden_sizes[2], hidden_size=hidden_sizes[3], batch_first=True)
        self.dropout4 = nn.Dropout(dropout_rates[3])

        self.fc = nn.Linear(hidden_sizes[3], 1)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = torch.relu(x)
        x = self.dropout1(x)
        
        x, _ = self.lstm2(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        
        x, _ = self.lstm3(x)
        x = torch.relu(x)
        x = self.dropout3(x)
        
        x, _ = self.lstm4(x)
        x = torch.relu(x)
        x = self.dropout4(x)
        
        x = self.fc(x[:, -1, :])  # shape (batch_size, 1)
        return x