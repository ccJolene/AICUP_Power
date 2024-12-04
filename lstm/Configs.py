import torch
import random
import numpy as np

class Config:
    seed = 42
    epochs = 50
    batch_size = 32
    lr = 5e-4
    weight_decay = 1e-4
    # hidden_size1 = 128  # 第一層 LSTM 的隱藏單元數
    # hidden_size2 = 64   # 第二層 LSTM 的隱藏單元數
    # num_layers = 2      # LSTM 總層數
    # dropout = 0.3       # Dropout 比例
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

def seed_everything(seed):
    random.seed(seed)

    np.random.seed(seed)
    
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True # 某些操作變得確定性，相同的輸入將產生相同的結果
    torch.backends.cudnn.benchmark = False    # 更新時禁用自動算法優化＆引入隨機性