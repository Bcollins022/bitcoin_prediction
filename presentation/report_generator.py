from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import matplotlib.pyplot as plt
import io
import os
import numpy as np

def generate_report(results):
    """
    Generates a PDF report based on training results.
    Args:
        results (dict): The results dictionary from train.py
    """
    filename = "Bitcoin_Price_Prediction_Report.pdf"
    doc = SimpleDocTemplate(filename, pagesize=letter)
    local_styles = getSampleStyleSheet()
    story = []

    # Title
    title_style = local_styles['Title']
    story.append(Paragraph("Bitcoin Price Prediction System Report", title_style))
    story.append(Spacer(1, 12))

    # Overview
    body_style = local_styles['BodyText']
    story.append(Paragraph("<b>1. Project Overview</b>", local_styles['Heading2']))
    text = ("This report summarizes the performance of various machine learning models trained "
            "to predict Bitcoin (BTC-USD) **12-hour log returns** using hourly market data. "
            "The system employs a **Walk-Forward Validation** (rolling window) strategy to ensure "
            "robust, out-of-sample performance evaluation without look-ahead bias. "
            "The models evaluated include LSTM, Hybrid (LSTM+GRU), Transformer, and XGBoost, along with an Ensemble model.")
    story.append(Paragraph(text, body_style))
    story.append(Spacer(1, 12))

    # Performance Metrics
    story.append(Paragraph("<b>2. Model Performance Evaluation</b>", local_styles['Heading2']))
    story.append(Paragraph("The following table shows the aggregate out-of-sample (OOS) performance metrics across all validation folds. "
                           "The target variable is the 12-hour log return.", body_style))
    story.append(Spacer(1, 12))

    # Table Data
    data = [['Model', 'RMSE', 'MAE', 'Dir Acc (%)']]
    for name, res in results.items():
        row = [
            name,
            f"{res['RMSE']:.6f}",
            f"{res['MAE']:.6f}",
            f"{res['DA']:.2f}"
        ]
        data.append(row)

    t = Table(data)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(t)
    story.append(Spacer(1, 24))

    # Visualization
    story.append(Paragraph("<b>3. Prediction Visualization (OOS)</b>", local_styles['Heading2']))
    
    # Create plot
    plt.figure(figsize=(8, 4))
    
    # Plot Actual
    # Using the first model's True values as reference
    first_key = list(results.keys())[0]
    true_vals = results[first_key]['True']
    plt.plot(true_vals, label='Actual Log Return', color='green', linewidth=1.0, alpha=0.6)
    
    # Plot Ensemble (to keep it clean, maybe just plot Ensemble or top selections)
    if 'Ensemble' in results:
        plt.plot(results['Ensemble']['Preds'], label='Ensemble Prediction', color='orange', linewidth=1.5)
    
    plt.title("Bitcoin 12h Log Return: Actual vs Predicted (OOS)")
    plt.xlabel("Test Samples (Hourly Steps)")
    plt.ylabel("Log Return")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=300)
    plt.close()
    img_buffer.seek(0)
    
    im = Image(img_buffer, width=450, height=225)
    story.append(im)
    story.append(Spacer(1, 12))

    # Conclusion
    story.append(Paragraph("<b>4. Conclusion & Strategy Review</b>", local_styles['Heading2']))
    conclusion = ("The use of log returns and walk-forward validation provides a more statistically sound "
                  "framework for quantitative trading than simple price prediction. "
                  "Directional accuracy above 50% suggests potential alpha, though execution costs and "
                  "market spread must be considered in a live strategy. "
                  "Risk Warning: Cryptocurrency trading involves extreme volatility. Past performance does not guarantee future results.")
    story.append(Paragraph(conclusion, body_style))

    doc.build(story)
    return f"Report generated successfully: {filename}"

if __name__ == "__main__":
    # Test generation with dummy data
    dummy_results = {
        'LSTM': {'RMSE': 1000, 'MAE': 800, 'MAPE': 0.05, 'DA': 55, 'Preds': np.random.rand(100)*10000, 'True': np.random.rand(100)*10000},
        'Ensemble': {'RMSE': 900, 'MAE': 750, 'MAPE': 0.04, 'DA': 60, 'Preds': np.random.rand(100)*10000, 'True': np.random.rand(100)*10000}
    }
    print(generate_report(dummy_results))
