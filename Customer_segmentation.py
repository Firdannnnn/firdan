from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


file_path = '/content/drive/MyDrive/data.csv'
df = pd.read_csv(file_path, encoding='ISO-8859-1')

# membaca dan membersihkan clustering
df = df.dropna(subset=['CustomerID'])
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

# REGRESI LINEAR
X = df[['Quantity', 'UnitPrice']]
y = df['TotalPrice']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluasi Model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("=== EVALUASI MODEL REGRESI LINEAR ===")
print(f"MAE (Mean Absolute Error): {mae:.2f}")
print(f"MSE (Mean Squared Error): {mse:.2f}")

# K-MEANS CLUSTERING
customer_df = df.groupby('CustomerID').agg({'TotalPrice': 'sum'}).reset_index()
customer_df.columns = ['CustomerID', 'TotalPurchase']

scaler = StandardScaler()
customer_df['NormalizedPurchase'] = scaler.fit_transform(customer_df[['TotalPurchase']])

kmeans = KMeans(n_clusters=3, random_state=42)
customer_df['Cluster'] = kmeans.fit_predict(customer_df[['NormalizedPurchase']])

# Visualisasi Cluster
plt.figure(figsize=(10, 6))
plt.scatter(customer_df['CustomerID'], customer_df['TotalPurchase'],
            c=customer_df['Cluster'], cmap='viridis', s=60)
plt.title('K-Means Clustering: Segmentasi Pelanggan Berdasarkan Total Pembelian')
plt.xlabel('CustomerID')
plt.ylabel('TotalPurchase')
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()

print("\n=== CONTOH HASIL CLUSTERING ===")
print(customer_df.head())