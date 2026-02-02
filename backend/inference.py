import torch
import joblib
import pandas as pd
import numpy as np
import os
import xgboost as xgb
from backend.models import LSTMModel, HybridModel, TransformerModel, XGBoostPredictor
from backend.features import add_features

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class InferenceEngine:
    def __init__(self, model_dir='backend/saved_models'):
        self.model_dir = model_dir
        self.scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
        self.target_scaler = joblib.load(os.path.join(model_dir, 'target_scaler.pkl'))
        self.models = {}
        self._load_models()
        
    def _load_models(self):
        # Infer input dim from scaler
        input_dim = self.scaler.n_features_in_
        
        # LSTM
        self.models['LSTM'] = LSTMModel(input_dim, 64, 1)
        self.models['LSTM'].load_state_dict(torch.load(os.path.join(self.model_dir, 'LSTM.pth'), map_location=DEVICE))
        self.models['LSTM'].to(DEVICE)
        self.models['LSTM'].eval()
        
        # Hybrid
        self.models['Hybrid'] = HybridModel(input_dim, 64, 1)
        self.models['Hybrid'].load_state_dict(torch.load(os.path.join(self.model_dir, 'Hybrid.pth'), map_location=DEVICE))
        self.models['Hybrid'].to(DEVICE)
        self.models['Hybrid'].eval()
        
        # Transformer
        self.models['Transformer'] = TransformerModel(input_dim, 64, 1)
        self.models['Transformer'].load_state_dict(torch.load(os.path.join(self.model_dir, 'Transformer.pth'), map_location=DEVICE))
        self.models['Transformer'].to(DEVICE)
        self.models['Transformer'].eval()
        
        # XGBoost
        self.models['XGBoost'] = XGBoostPredictor()
        self.models['XGBoost'].load_model(os.path.join(self.model_dir, 'XGBoost.json'))
        
    def predict_next_12h(self, df):
        """
        Predicts the 12-hour Log Return and implies 12h Price.
        """
        # Ensure we have features
        df = add_features(df)
        
        # Need at least LOOKBACK=48 rows
        if len(df) < 48:
            raise ValueError("Not enough data. Need at least 48 hours.")
            
        recent_data = df.iloc[-48:].copy() # LOOKBACK=48
        
        # Scale Features
        # Note: df has features but scaler.transform expects exact cols match.
        # scaler was fit on data from add_features, so columns should match order if consistent.
        # But add_features might have fewer cols than train.py's df if train.py dropped some?
        # train.py dropped 'Target' col. add_features doesn't have it.
        # So we should be good.
        
        scaled_data = self.scaler.transform(recent_data)
        
        input_seq = torch.FloatTensor(scaled_data).unsqueeze(0).to(DEVICE) # (1, 48, F)
        
        preds_scaled = {}
        
        with torch.no_grad():
            preds_scaled['LSTM'] = self.models['LSTM'](input_seq).item()
            preds_scaled['Hybrid'] = self.models['Hybrid'](input_seq).item()
            preds_scaled['Transformer'] = self.models['Transformer'](input_seq).item()
            
        preds_scaled['XGBoost'] = self.models['XGBoost'].predict(np.array([scaled_data]))[0]
        
        final_preds_price = {}
        final_preds_log_ret = {}
        
        raw_preds_scaled = []
        last_close = df['Close'].iloc[-1]
        
        for name, p_scaled in preds_scaled.items():
            # Inverse Transform Target
            # p_scaled is float. reshape to (1,1)
            inv_log_ret = self.target_scaler.inverse_transform(np.array([[p_scaled]]))[0, 0]
            
            # Reconstruct Price
            # Price[t+12] = Price[t] * exp(log_ret)
            pred_price = last_close * np.exp(inv_log_ret)
            
            final_preds_price[name] = float(pred_price)
            final_preds_log_ret[name] = float(inv_log_ret)
            raw_preds_scaled.append(p_scaled)
            
        # Ensemble Scaled
        ens_scaled = np.mean(raw_preds_scaled)
        ens_log_ret = self.target_scaler.inverse_transform(np.array([[ens_scaled]]))[0, 0]
        ens_price = last_close * np.exp(ens_log_ret)
        
        final_preds_price['Ensemble'] = float(ens_price)
        final_preds_log_ret['Ensemble'] = float(ens_log_ret)
        
        return {
            'price': final_preds_price,
            'log_return': final_preds_log_ret,
            'Ensemble': float(ens_price),
            'horizon': '12h'
        }

    # Deprecating predict_future recursive for now as 12h horizon makes hourly stepping impossible
    # Replacing with a simple "Next 12h" repeater or just removed. 
    # For compatibility, we return just the 12h prediction as a single point "forecast".
    def predict_future(self, df, days=1):
        # We only support 1 step (12h) ahead reliably with this model.
        pred = self.predict_next_12h(df)
        
        last_date = df.index[-1]
        future_date = last_date + pd.Timedelta(hours=12)
        
        return [{"date": future_date.strftime("%Y-%m-%d %H:%M"), "predicted_price": pred['Ensemble']}]

if __name__ == "__main__":
    # Test inference
    from backend.data_loader import fetch_data
    df = fetch_data()
    engine = InferenceEngine()
    p = engine.predict_next_12h(df)
    print("Predictions:", p)
