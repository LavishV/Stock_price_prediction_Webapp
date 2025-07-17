import os
import pandas as pd
from prediction.utils.predict import run_prediction

# Set your dates and ticker
ticker = 'AAPL'
start_date = '2023-01-01'
end_date = '2025-05-20'

# Run prediction once
df_future, df_results, image_path = run_prediction(ticker, start_date, end_date)

# Path to store demo files
demo_dir = os.path.join('prediction', 'static', 'demo_data')
os.makedirs(demo_dir, exist_ok=True)

# Save CSV files
df_results.to_csv(os.path.join(demo_dir, 'df_results.csv'), index=False)
df_future.to_csv(os.path.join(demo_dir, 'df_future.csv'), index=False)

# Move chart image to demo folder (if needed)
if image_path and os.path.exists(image_path):
    import shutil
    shutil.copy(image_path, os.path.join(demo_dir, 'prediction_chart.png'))

print("Demo data generated and saved in 'prediction/static/demo_data/'")
