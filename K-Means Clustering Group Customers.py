import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings

# Ignore warnings for a cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)

try:
    df = pd.read_csv(r"C:\Users\adils\OneDrive\Documents\customer_data.csv")# Replace with your file path
except FileNotFoundError:
    print("Error: 'customer_data.csv' not found.")
    print("Please make sure the file is in the same directory or run the data generation script.")
    exit()
# Use exact, case-sensitive names from the CSV file.
features = df[['Annual_Income_k', 'Spending_Score']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
#Taking 5 clusters for example
K = 5
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(scaled_features)
plt.figure(figsize=(10, 7))

plt.scatter(df['Annual_Income_k'], df['Spending_Score'], c=df['Cluster'], cmap='viridis', alpha=0.8, edgecolors='k')

# Adding titles and labels for understanding
plt.title('Customer Segments', fontsize=16)
plt.xlabel('Annual Income (k$)', fontsize=12)
plt.ylabel('Spending Score (1-100)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
print(f"\nSuccessfully segmented customers into {K} clusters.")
print("Here is a sample of the data with the assigned clusters:")
print(df.head())
