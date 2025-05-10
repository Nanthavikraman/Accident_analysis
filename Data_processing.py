import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os
import time
from datetime import datetime
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def update_progress(progress, message=''):
    bar_length = 30
    block = int(round(bar_length * progress))
    text = f"\r[{('=' * block) + ('-' * (bar_length - block))}] {progress*100:.1f}% - {message}"
    print(text, end='', flush=True)

def load_data(file_path):
    print(f"\nStarting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    file_size = os.path.getsize(file_path)
    chunk_size = min(int(1e6), max(int(file_size / (100 * 1024 * 1024)), int(1e5)))
    chunks = []
    total_rows = sum(1 for _ in open(file_path)) - 1
    total_chunks = (total_rows - 1) // chunk_size + 1
    dtype_dict = {
        'Severity': 'int8',
        'Temperature(F)': 'float32',
        'Humidity(%)': 'float32',
        'Pressure(in)': 'float32',
        'Visibility(mi)': 'float32',
        'Wind_Speed(mph)': 'float32',
        'Precipitation(in)': 'float32'
    }
    for i, chunk in enumerate(pd.read_csv(file_path, 
                                        chunksize=chunk_size,
                                        dtype=dtype_dict,
                                        usecols=['Start_Time', 'Severity', 'Temperature(F)', 
                                                'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 
                                                'Wind_Speed(mph)', 'Precipitation(in)', 
                                                'Weather_Condition'])):
        update_progress((i + 1) / total_chunks, f"Loading chunk {i+1}/{total_chunks}")
        chunk['Start_Time'] = pd.to_datetime(chunk['Start_Time'], format='mixed', errors='coerce')
        chunk = chunk.dropna(subset=['Start_Time'])
        chunk['Hour'] = chunk['Start_Time'].dt.hour
        chunk['Day_of_Week'] = chunk['Start_Time'].dt.dayofweek
        chunk['Month'] = chunk['Start_Time'].dt.month
        chunk['Severe_Accident'] = (chunk['Severity'] > 2).astype('int8')
        chunk.drop(['Start_Time', 'Severity'], axis=1, inplace=True)
        chunks.append(chunk)
    df = pd.concat(chunks, copy=False)
    del chunks
    return df.copy()

def feature_engineering(df):
    df['Is_Rush_Hour'] = ((df['Hour'].between(7, 9)) | 
                         (df['Hour'].between(16, 18))).astype('int8')
    df['Is_Weekend'] = (df['Day_of_Week'] >= 5).astype('int8')
    weather_categories = {
        'Clear|Fair': 'Clear',
        'Cloudy|Overcast': 'Cloudy',
        'Rain|Heavy Rain': 'Rain',
        'Snow|Heavy Snow': 'Snow',
        'Fog|Haze': 'Fog'
    }
    df['Weather_Simple'] = 'Other'
    for pattern, category in weather_categories.items():
        mask = df['Weather_Condition'].str.contains(pattern, case=False, na=False)
        df.loc[mask, 'Weather_Simple'] = category
    features_to_check = ['Hour', 'Day_of_Week', 'Month', 'Temperature(F)',
                        'Humidity(%)', 'Pressure(in)', 'Visibility(mi)',
                        'Wind_Speed(mph)', 'Precipitation(in)', 'Weather_Simple',
                        'Is_Rush_Hour', 'Is_Weekend', 'Severe_Accident']
    df = df.dropna(subset=features_to_check)
    return df, features_to_check

try:
    start_time = time.time()
    df = load_data('/content/drive/MyDrive/US_Accidents_March23.csv')
    load_time = time.time() - start_time
    feature_start = time.time()
    df, features = feature_engineering(df)
    feature_time = time.time() - feature_start
    save_start = time.time()
    output_dir = '/content/drive/MyDrive/accident_prediction_processed'
    os.makedirs(output_dir, exist_ok=True)
    chunk_size = 500000
    n_chunks = len(df) // chunk_size + 1
    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(df))
        if i == 0:
            df.iloc[start_idx:end_idx].to_csv(f"{output_dir}/processed_data.csv", index=False, mode='w')
        else:
            df.iloc[start_idx:end_idx].to_csv(f"{output_dir}/processed_data.csv", index=False, mode='a', header=False)
        update_progress((i + 1) / n_chunks, f"Saving chunk {i+1}/{n_chunks}")
    save_time = time.time() - save_start
    total_time = time.time() - start_time
    print(f"\nProcessing Summary:")
    print(f"- Loading time: {load_time:.1f} seconds ({load_time/60:.1f} minutes)")
    print(f"- Feature engineering time: {feature_time:.1f} seconds ({feature_time/60:.1f} minutes)")
    print(f"- Saving time: {save_time:.1f} seconds ({save_time/60:.1f} minutes)")
    print(f"- Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"- Final dataset shape: {df.shape}")
    print(f"- Memory usage: {df.memory_usage().sum() / (1024**2):.1f} MB")
except Exception as e:
    print(f"\nError occurred: {str(e)}")
    raise
