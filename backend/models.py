import torch
import torch.nn as nn
import xgboost as xgb


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class HybridModel(nn.Module):
    """LSTM followed by GRU"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.2):
        super(HybridModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # First layer LSTM
        self.lstm = nn.LSTM(input_dim, hidden_dim, 1, batch_first=True)
        # Second layer GRU
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers - 1, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out, _ = self.gru(out)
        out = self.fc(out[:, -1, :])
        return out

class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, nhead=4, num_layers=2, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.input_embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.input_embedding(x)
        out = self.transformer_encoder(x)
        out = self.fc(out[:, -1, :])
        return out

class XGBoostPredictor:
    def __init__(self, model_params=None):
        self.model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
        if model_params:
            self.model.set_params(**model_params)

    def fit(self, X, y):
        # X shape: (samples, seq_len, features) -> flatten to (samples, seq_len*features)
        X_flat = X.reshape(X.shape[0], -1)
        self.model.fit(X_flat, y)

    def predict(self, X):
        X_flat = X.reshape(X.shape[0], -1)
        return self.model.predict(X_flat)
    
    def save_model(self, path):
        self.model.save_model(path)

    def load_model(self, path):
        # Safeguard for XGBoost sklearn wrapper issue where _estimator_type might be undefined
        if not hasattr(self.model, '_estimator_type'):
            self.model._estimator_type = "regressor"
        self.model.load_model(path)
