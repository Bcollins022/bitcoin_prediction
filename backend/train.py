import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import os
import joblib

from backend.data_loader import fetch_data
from backend.features import add_features
from backend.models import LSTMModel, HybridModel, TransformerModel, XGBoostPredictor

# Set device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_sequences(data, seq_length, target_col_idx):
    """
    Creates sequences for time series forecasting.
    """
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length, target_col_idx]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Overloaded for separate target capability
def create_sequences_separate_target(data, seq_length, target_data):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = target_data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def train_model_torch(model, X_train, y_train, X_val, y_val, epochs=5, batch_size=32, lr=0.001):
    model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train).unsqueeze(1))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # if (epoch + 1) % 1 == 0:
        #     print(f"Epoch [{epoch+1}/{epochs}], Loss: {train_loss/len(train_loader):.4f}")
            
    # Validation
    model.eval()
    with torch.no_grad():
        val_inputs = torch.FloatTensor(X_val).to(DEVICE)
        val_preds = model(val_inputs).cpu().numpy()
        
    return val_preds

def train_pipeline(test_months=3):
    # 1. Load Data
    # [QUANT STANDARDS] Hourly data, last 730 days, incomplete last candle dropped in loader
    df = fetch_data(interval="1h", period="730d") 
    df = add_features(df)
    
    # [QUANT STANDARDS] Target Definition
    # Predict 12-hour log returns: log(Close[t+12] / Close[t])
    HORIZON = 12
    target_col = 'Target_12h_LogRet'
    
    # We compute the target by shifting Close backwards by Horizon (future data)
    # Then take log return. 
    # Target[t] = log(Close[t+HORIZON]) - log(Close[t])
    df[target_col] = np.log(df['Close'].shift(-HORIZON) / df['Close'])
    
    # Drops rows where Target is NaN (the last 12 hours of data)
    # This prevents training on samples where we don't know the outcome yet
    df_labeled = df.dropna(subset=[target_col]).copy()
    
    # 2. Setup Walk-Forward Validation
    # We will use an expanding window approach.
    # Total samples
    n_samples = len(df_labeled)
    
    # Configuration
    LOOKBACK = 48 # 48 hours input sequence
    n_folds = 5
    # Reserve last portion for validation folds
    # Let's say we want to validate over the last 30% of data using 5 folds
    validation_size = int(n_samples * 0.3)
    fold_size = int(validation_size / n_folds)
    
    # Start of validation period
    start_val_idx = n_samples - validation_size
    
    print(f"Total Labeled Samples: {n_samples}")
    print(f"Walk-Forward Validation: {n_folds} folds, size {fold_size}")
    
    # Global Lists to store OOS predictions (for overall metrics)
    oos_preds = {'LSTM': [], 'Hybrid': [], 'Transformer': [], 'XGBoost': [], 'Ensemble': []}
    oos_targets = []
    
    # Metrics accumulator
    fold_metrics = []
    
    input_dim = 0 # will be set dynamically
    
    # Save the feature columns to exclude Target from features
    feature_cols = [c for c in df_labeled.columns if c != target_col]
    
    # 3. Walk-Forward Loop
    for fold in range(n_folds):
        current_train_end = start_val_idx + (fold * fold_size)
        current_test_end = current_train_end + fold_size
        
        # [QUANT STANDARDS] Train / Test Boundaries
        # Train: [0 : current_train_end]
        # Test:  [current_train_end : current_test_end]
        
        train_df = df_labeled.iloc[:current_train_end]
        test_df = df_labeled.iloc[current_train_end:current_test_end]
        
        # [QUANT STANDARDS] Scaling
        # Fit ONLY on training data
        scaler = MinMaxScaler()
        scaler.fit(train_df[feature_cols])
        
        # Transform both
        train_scaled = scaler.transform(train_df[feature_cols])
        test_scaled = scaler.transform(test_df[feature_cols])
        
        # Targets (already essentially scaled as log returns are small, but we leave them raw for regression or scale them?)
        # Standard practice: Scale targets too if using MSELoss to keep gradients stable.
        # Let's fit a target scaler.
        target_scaler = MinMaxScaler()
        target_scaler.fit(train_df[[target_col]])
        
        y_train_scaled = target_scaler.transform(train_df[[target_col]]).flatten()
        y_test_scaled = target_scaler.transform(test_df[[target_col]]).flatten()
        
        # Create Sequences
        # Note: Sequence creation usually chops off the first LOOKBACK samples.
        X_train, y_train_seq = create_sequences_separate_target(train_scaled, LOOKBACK, y_train_scaled)
        X_test, y_test_seq = create_sequences_separate_target(test_scaled, LOOKBACK, y_test_scaled)
        
        if len(X_test) == 0:
            print(f"Fold {fold}: Not enough test data. Skipping.")
            continue
            
        input_dim = X_train.shape[2]
        
        # Re-initialize models for each fold to strictly prevent leakage
        models = {
            'LSTM': LSTMModel(input_dim, 64, 1),
            'Hybrid': HybridModel(input_dim, 64, 1),
            'Transformer': TransformerModel(input_dim, 64, 1),
            'XGBoost': XGBoostPredictor()
        }
        
        print(f"--- Fold {fold+1}/{n_folds} ---")
        print(f"Train: {len(X_train)} samples | Test: {len(X_test)} samples")
        
        fold_predictions_scaled = {}
        
        for name, model in models.items():
            if name == 'XGBoost':
                model.fit(X_train, y_train_seq)
                p = model.predict(X_test)
            else:
                p = train_model_torch(model, X_train, y_train_seq, X_test, y_test_seq, epochs=3) # reduced epochs for speed
                p = p.flatten()
            
            fold_predictions_scaled[name] = p
            
        # Ensemble Scaled
        ens_scaled = np.mean([fold_predictions_scaled[m] for m in models], axis=0)
        fold_predictions_scaled['Ensemble'] = ens_scaled
        
        # [QUANT STANDARDS] Inverse Transform Targets for Evaluation
        # We want to evaluate meaningful log returns
        actual_log_ret = target_scaler.inverse_transform(y_test_seq.reshape(-1, 1)).flatten()
        oos_targets.extend(actual_log_ret)
        
        for name, p_scaled in fold_predictions_scaled.items():
            inv_p = target_scaler.inverse_transform(p_scaled.reshape(-1, 1)).flatten()
            oos_preds[name].extend(inv_p)
            
    # 4. Overall OOS Evaluation
    print("\n=== Aggregate Out-of-Sample Performance ===")
    final_results = {}
    
    oos_targets = np.array(oos_targets)
    
    for name, preds in oos_preds.items():
        preds = np.array(preds)
        mse = mean_squared_error(oos_targets, preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(oos_targets, preds)
        
        # Directional Accuracy on Log Returns
        # If predicted log ret > 0, we predict Up. 
        da = np.mean(np.sign(preds) == np.sign(oos_targets)) * 100
        
        print(f"{name}: RMSE={rmse:.6f}, MAE={mae:.6f}, DA={da:.2f}%")
        
        final_results[name] = {
            'RMSE': rmse, 
            'MAE': mae, 
            'DA': da, 
            # We save the LAST fold's models/scalers later for live inference
        }
        
    # 5. Final Model Training (For Deployment)
    # Train on ALL labeled data to maximize information for live trading
    print("\nTraining Final Production Models on Full Dataset...")
    feature_scaler = MinMaxScaler()
    feature_scaler.fit(df_labeled[feature_cols])
    
    target_scaler = MinMaxScaler()
    target_scaler.fit(df_labeled[[target_col]])
    
    X_full = feature_scaler.transform(df_labeled[feature_cols])
    y_full = target_scaler.transform(df_labeled[[target_col]]).flatten()
    
    X_seq, y_seq = create_sequences_separate_target(X_full, LOOKBACK, y_full)
    
    # Save artifacts
    os.makedirs('backend/saved_models', exist_ok=True)
    joblib.dump(feature_scaler, 'backend/saved_models/scaler.pkl')
    joblib.dump(target_scaler, 'backend/saved_models/target_scaler.pkl') # Need to save target scaler too
    
    live_results = {}
    
    for name, model in models.items():
        # Re-init
        if name == 'LSTM': model = LSTMModel(input_dim, 64, 1)
        elif name == 'Hybrid': model = HybridModel(input_dim, 64, 1)
        elif name == 'Transformer': model = TransformerModel(input_dim, 64, 1)
        elif name == 'XGBoost': model = XGBoostPredictor()
        
        if name == 'XGBoost':
            model.fit(X_seq, y_seq)
            model.save_model(f'backend/saved_models/{name}.json')
        else:
            # For torch models, we just do a simple train loop as "validation" is now just convergence check
            # We can use a small portion of end of train as pseudo-val or just train for fixed epochs
            train_model_torch(model, X_seq, y_seq, X_seq[-100:], y_seq[-100:], epochs=5)
            torch.save(model.state_dict(), f'backend/saved_models/{name}.pth')
            
    # Save Results for report (using OOS metrics for honesty)
    # We define 'Preds' and 'True' as the OOS concatenated arrays for the report generator
    for name in final_results:
        final_results[name]['Preds'] = np.array(oos_preds[name])
        final_results[name]['True'] = oos_targets
        
    joblib.dump(final_results, 'backend/saved_models/results.pkl')
    print("Done. Models and OOS results valid.")
    return final_results

# Helper for seq creation to handle separate X and y
def create_sequences(data, seq_length, target_data):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = target_data[i + seq_length] # Alignment: Input [0..47] -> Target [48]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

if __name__ == "__main__":
    train_pipeline()
