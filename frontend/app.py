"""
Bitcoin Price Prediction Dashboard.
This Streamlit application serves as the frontend for the Bitcoin Price Prediction System.
It provides visualizations, live predictions, future forecasts, and deep analysis tools.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import os
import sys

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backend.inference import InferenceEngine
from backend.data_loader import fetch_data

st.set_page_config(page_title="Bitcoin Price Prediction", layout="wide")

st.title("Bitcoin Price Prediction System 🚀")

@st.cache_resource
def load_inference_engine():
    if not os.path.exists('backend/saved_models/LSTM.pth'):
        return None
    return InferenceEngine()

@st.cache_data
def load_results():
    if not os.path.exists('backend/saved_models/results.pkl'):
        return None
    return joblib.load('backend/saved_models/results.pkl')

engine = load_inference_engine()
results = load_results()


tabs = st.tabs(["Dashboard", "Future Forecasts", "Model Performance", "Deep Analysis", "Raw Data", "Report"])

with tabs[0]:
    st.header("12-Hour Forecast")
    
    if engine:
        # Fetch latest data
        with st.spinner("Fetching latest market data..."):
            df = fetch_data() # Defaults to today
            
        if not df.empty:
            st.subheader("Recent Price Action (Hourly)")
            # Make charts zoomable
            fig_recent = go.Figure()
            fig_recent.add_trace(go.Scatter(x=df.index[-90:], y=df['Close'].iloc[-90:], mode='lines', name='Close Price'))
            fig_recent.update_layout(dragmode='pan', xaxis=dict(fixedrange=False), yaxis=dict(fixedrange=False))
            st.plotly_chart(fig_recent, use_container_width=True, config={'scrollZoom': True})
            
            if st.button("Predict Next 12 Hours"):
                try:
                    preds_data = engine.predict_next_12h(df)
                    
                    # Extract Data
                    price_preds = preds_data['price']
                    log_ret_preds = preds_data['log_return']
                    ensemble_price = price_preds['Ensemble']
                    ensemble_log_ret = log_ret_preds['Ensemble']
                    
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("Ensemble Prediction (12h)", f"${ensemble_price:,.2f}")
                    with c2:
                        st.metric("Last Close", f"${df['Close'].iloc[-1]:,.2f}")
                    with c3:
                        diff = ensemble_price - df['Close'].iloc[-1]
                        pct = (diff / df['Close'].iloc[-1]) * 100
                        st.metric("Predicted Move (12h)", f"{pct:.2f}%", delta=f"{diff:,.2f}")
                    
                    # Predicted Log Return
                    st.metric("Predicted 12h Log Return", f"{ensemble_log_ret:.6f}")

                    # Buy/Sell/Hold Recommendation
                    st.write("### AI Recommendation")
                    rec_col1, rec_col2 = st.columns([1, 3])
                    recommendation = "HOLD"
                    color = "blue"
                    
                    if pct > 1.0:
                        recommendation = "BUY"
                        color = "green"
                    elif pct < -1.0:
                        recommendation = "SELL"
                        color = "red"
                        
                    with rec_col1:
                        st.markdown(f"<h2 style='color: {color}; text-align: center; border: 2px solid {color}; border-radius: 10px; padding: 10px;'>{recommendation}</h2>", unsafe_allow_html=True)
                    with rec_col2:
                        st.info(f"Model predicts a move of {pct:.2f}% over the next 12 hours. Recommendation based on >1% threshold.")

                    st.write("### Individual Model Predictions (Price)")
                    st.json(price_preds)
                    
                    st.write("### Individual Model Predictions (Log Returns)")
                    st.json(log_ret_preds)
                    
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    
            # Full History Graph
            st.subheader("All-Time Bitcoin Price History")
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price', line=dict(color='orange')))
            fig_hist.update_layout(
                title="Bitcoin Price (Inception - Present)",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                dragmode='pan',
                hovermode='x unified'
            )
            st.plotly_chart(fig_hist, use_container_width=True, config={'scrollZoom': True})
            
        else:
            st.error("Could not fetch data.")
    else:
        st.warning("Models not found. Please train the models first.")

with tabs[1]:
    st.header("Forecast Details")
    st.markdown("""
    **Note**: This model predicts the **12-Hour Log Return**. 
    The chart below shows the projected price point 12 hours from now.
    """)
    
    if engine:
        if st.button("Generate 12h Projection"):
            with st.spinner("Calculating projection..."):
                try:
                    df_future = fetch_data()
                    future_preds = engine.predict_future(df_future, days=1) # days arg ignored/deprecated, returns 12h point
                    
                    # Parse response
                    # msg = [{"date": ..., "predicted_price": ...}]
                    pred_date = pd.to_datetime(future_preds[0]['date'])
                    pred_val = future_preds[0]['predicted_price']
                    
                    # Plot
                    fig_future = go.Figure()
                    
                    # Historical (last 48h)
                    hist_data = df_future.iloc[-48:]
                    fig_future.add_trace(go.Scatter(x=hist_data.index, y=hist_data['Close'], name='Historical', line=dict(color='gray')))
                    
                    # Future Point
                    fig_future.add_trace(go.Scatter(x=[pred_date], y=[pred_val], mode='markers+text', name='12h Forecast', 
                                                    text=[f"${pred_val:,.0f}"], textposition="top center",
                                                    marker=dict(color='orange', size=12)))
                    
                    fig_future.update_layout(dragmode='pan', hovermode='x unified', title="12-Hour Price Projection")
                    st.plotly_chart(fig_future, use_container_width=True, config={'scrollZoom': True})
                    
                except Exception as e:
                    st.error(f"Forecasting failed: {e}")
    else:
        st.warning("Models not loaded.")

with tabs[2]:
    st.header("Model Evaluation (Walk-Forward Testing)")
    st.write("Metrics evaluated on Out-of-Sample (OOS) data from Walk-Forward Validation.")
    st.write("**Target**: 12-Hour Log Return")
    
    if results:
        # Metrics Table
        metrics_data = []
        for name, res in results.items():
            metrics_data.append({
                "Model": name,
                "RMSE": res['RMSE'],
                "MAE": res['MAE'],
                "DA (%)": res['DA']
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        st.table(metrics_df.style.format({
            "RMSE": "{:.6f}", 
            "MAE": "{:.6f}",
            "DA (%)": "{:.2f}"
        }))
        
        # Charts
        st.subheader("Prediction vs Actual (Log Returns)")
        
        selected_models = st.multiselect("Select Models to Compare", list(results.keys()), default=["Ensemble"])
        
        if selected_models:
            fig = go.Figure()
            # Plot Actual (from first selected model's true values)
            first = selected_models[0]
            true_vals = results[first]['True']
            fig.add_trace(go.Scatter(y=true_vals, name="Actual Log Return", line=dict(color='green', width=1)))
            
            for m in selected_models:
                pred_vals = results[m]['Preds']
                fig.add_trace(go.Scatter(y=pred_vals, name=f"{m} Log Ret"))
                
            fig.update_layout(
                dragmode='pan', 
                hovermode='x unified',
                yaxis_title="Log Return",
                xaxis_title="Test Samples (Concatenated Folds)"
            )
            st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})
            
    else:
        st.info("No training results found. Run backend/train.py")

with tabs[3]:
    st.header("Deep Analysis")
    st.write("Interactive investigation of model performance (Log Returns).")
    
    if results:
        # Multiselect for models
        da_models = st.multiselect("Select Models to Analyze", list(results.keys()), default=["Ensemble"])
        
        if da_models:
            from plotly.subplots import make_subplots
            
            # Create subplots: Row 1 = Log Return, Row 2 = Difference
            fig_da = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                   vertical_spacing=0.1, 
                                   subplot_titles=("Actual vs Predicted Log Returns", "Prediction Error"),
                                   row_heights=[0.7, 0.3])
            
            # Use the first selected model to get the 'True' values and length
            base_model = da_models[0]
            true_vals = results[base_model]['True']
            # Generate a generic index (0 to N) or use dates if available in results. 
            # results object currently has numpy arrays, no index. We can imply steps.
            steps = np.arange(len(true_vals))
            
            # Add Actual Price Trace
            fig_da.add_trace(go.Scatter(x=steps, y=true_vals, name="Actual", line=dict(color='black', width=1)), row=1, col=1)
            
            for m in da_models:
                preds = results[m]['Preds']
                diff = preds - true_vals
                
                # Price Trace
                fig_da.add_trace(go.Scatter(x=steps, y=preds, name=f"{m} Pred", opacity=0.8), row=1, col=1)
                
                # Difference Trace
                fig_da.add_trace(go.Scatter(x=steps, y=diff, name=f"{m} Error"), row=2, col=1)
            
            fig_da.update_layout(
                height=700,
                hovermode='x unified',
                dragmode='pan'
            )
            fig_da.update_xaxes(title_text="Time Steps (OOS)", row=2, col=1)
            fig_da.update_yaxes(title_text="Log Return", row=1, col=1)
            fig_da.update_yaxes(title_text="Error", row=2, col=1)
            
            st.plotly_chart(fig_da, use_container_width=True, config={'scrollZoom': True})
            
            # Stats for selection
            st.write("### Error Statistics for Selection")
            stats_list = []
            for m in da_models:
                diff = results[m]['Preds'] - results[m]['True']
                stats_list.append({
                    "Model": m,
                    "Mean Error": np.mean(diff),
                    "Max Positive Error (Overshoot)": np.max(diff),
                    "Max Negative Error (Undershoot)": np.min(diff),
                    "Std Dev Error": np.std(diff)
                })
            st.table(pd.DataFrame(stats_list).style.format({
                "Mean Error": "{:.2f}",
                "Max Positive Error (Overshoot)": "{:.2f}",
                "Max Negative Error (Undershoot)": "{:.2f}",
                "Std Dev Error": "{:.2f}"
            }))
            
    else:
        st.info("No results available for analysis.")

with tabs[4]:
    st.header("Raw Data")
    df_raw = fetch_data()
    st.dataframe(df_raw.tail(100))

with tabs[5]:
    st.header("Report Generation")
    st.write("Generate a professional PDF report of the findings.")
    
    if results:
        report_models = st.multiselect("Select Models for Report", list(results.keys()), default=list(results.keys()))
    else:
        report_models = []
    if st.button("Generate PDF Report"):
        from presentation.report_generator import generate_report
        if results:
            try:
                # Filter results based on selection
                if not report_models:
                    st.error("Please select at least one model.")
                else:
                    filtered_results = {k: v for k, v in results.items() if k in report_models}
                    msg = generate_report(filtered_results)
                    st.success(msg)
                    
                    with open("Bitcoin_Price_Prediction_Report.pdf", "rb") as pdf:
                        st.download_button("Download Report", pdf, "Bitcoin_Price_Prediction_Report.pdf")
            except Exception as e:
                st.error(f"Failed to generate report: {e}")
        else:
            st.error("No results to report on.")
