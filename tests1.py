import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load dataset
df = pd.read_csv("Task 3 and 4_Loan_Data.csv")

# Get FICO and default data
fico_scores = df['fico_score'].values.reshape(-1, 1)
defaults = df['default'].values

# -----------------------------
# Function to bucket FICO scores using KMeans
# -----------------------------
def quantize_fico_scores_mse(fico_scores, n_buckets=5):
    kmeans = KMeans(n_clusters=n_buckets, random_state=42, n_init=10)
    bucket_labels = kmeans.fit_predict(fico_scores)

    df_buckets = pd.DataFrame({
        'fico_score': fico_scores.flatten(),
        'bucket': bucket_labels
    })

    df_buckets['default'] = defaults
    bucket_default_rates = df_buckets.groupby('bucket')['default'].mean()
    sorted_buckets = bucket_default_rates.sort_values().index
    bucket_rating_map = {old: new for new, old in enumerate(sorted_buckets)}
    df_buckets['rating'] = df_buckets['bucket'].map(bucket_rating_map)

    return df_buckets[['fico_score', 'rating']]

# -----------------------------
# Apply bucketing
# -----------------------------
bucketed_scores = quantize_fico_scores_mse(fico_scores, n_buckets=5)
df = df.merge(bucketed_scores, on='fico_score')

# -----------------------------
# Group summary
# -----------------------------
bucket_summary = df.groupby('rating').agg({
    'fico_score': ['min', 'max'],
    'default': ['mean', 'count']
})
bucket_summary.columns = ['fico_min', 'fico_max', 'default_rate', 'count']
print("\n📊 Bucket Summary:\n", bucket_summary)

# -----------------------------
# Plotting
# -----------------------------
plt.figure(figsize=(8, 5))
plt.bar(bucket_summary.index, bucket_summary['default_rate'], color='orange')
plt.xticks(bucket_summary.index)
plt.xlabel("Rating Bucket (0 = Best FICO)")
plt.ylabel("Default Rate")
plt.title("Default Rate by FICO Rating Bucket")
plt.grid(True)
plt.tight_layout()
plt.show()
