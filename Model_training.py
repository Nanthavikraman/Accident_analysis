import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os
import time
from datetime import datetime

def update_progress(progress, message=''):
    bar_length = 30
    block = int(round(bar_length * progress))
    text = f"\r[{('=' * block) + ('-' * (bar_length - block))}] {progress*100:.1f}% - {message}"
    print(text, end='', flush=True)

try:
    start_time = time.time()

    print(f"\nStarting model pipeline at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    update_progress(0.05, "Loading dataset...")

    file_path = '/content/drive/MyDrive/accident_prediction_processed/processed_data.csv'
    data = pd.read_csv(file_path)
    data.dropna(inplace=True)
    update_progress(0.15, "Dataset loaded and cleaned")

    update_progress(0.20, "Sampling data (500,000 rows)...")
    data = data.sample(n=500000, random_state=42)

    update_progress(0.30, "Encoding categorical features...")
    encoder = LabelEncoder()
    data['Weather_Simple'] = encoder.fit_transform(data['Weather_Simple'])

    if 'Weather_Condition' in data.columns:
        data.drop(columns=['Weather_Condition'], inplace=True)

    for column in data.columns:
        if data[column].dtype == 'object':
            data.drop(columns=[column], inplace=True)

    update_progress(0.40, "Splitting data...")
    X = data.drop(columns=['Severe_Accident'])
    y = data['Severe_Accident']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    update_progress(0.50, "Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    update_progress(0.60, "Training model (Random Forest)...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    train_start = time.time()
    model.fit(X_train_scaled, y_train)
    train_end = time.time()
    update_progress(0.85, f"Model trained in {train_end - train_start:.1f}s")

    update_progress(0.90, "Evaluating model...")
    y_pred = model.predict(X_test_scaled)
    print("\n\n Evaluation Results:")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    update_progress(0.95, "Saving model and scaler...")
    output_dir = '/content/drive/MyDrive/accident_prediction_model'
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(model, f'{output_dir}/random_forest_model.pkl')
    joblib.dump(scaler, f'{output_dir}/scaler.pkl')
    joblib.dump(encoder, f'{output_dir}/weather_encoder.pkl')

    update_progress(1.0, "All done!")
    total_time = time.time() - start_time

    print(f"\n\nPipeline complete in {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"Artifacts saved in: {output_dir}")

except Exception as e:
    print(f"\n Error: {str(e)}")
    raise
