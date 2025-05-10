import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('/content/drive/MyDrive/accident_prediction_processed/processed_data.csv')

print(df.columns)

numerical_features = df.select_dtypes(include=['number']).columns
df_numerical = df[numerical_features]

correlation_matrix = df_numerical.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of Features')
plt.show()

target_corr = correlation_matrix['Severe_Accident'].sort_values(ascending=False)
print(target_corr)
