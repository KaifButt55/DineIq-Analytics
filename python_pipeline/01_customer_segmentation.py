import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(BASE_DIR, 'raw_data')
OUT = os.path.join(BASE_DIR, 'python_pipeline')
os.makedirs(OUT, exist_ok=True)

print("⏳ Loading data...")
orders = pd.read_csv(f'{RAW}/orders.csv', parse_dates=['order_date'])
orders = orders[(orders['status']=='Completed') & (orders['total_amount']>0)]

# RFM
print("📊 RFM calculate...")
max_date = orders['order_date'].max() + pd.Timedelta(days=1)
rfm = orders.groupby('customer_id').agg({
    'order_date': lambda x: (max_date - x.max()).days,
    'order_id': 'count',
    'total_amount': 'sum'
}).rename(columns={'order_date':'recency','order_id':'frequency','total_amount':'monetary'})

print(f"✅ {len(rfm)} customers")

# K-Means
scaler = StandardScaler()
X = scaler.fit_transform(rfm[['recency','frequency','monetary']])
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
rfm['segment'] = kmeans.fit_predict(X)

# Labels
seg_order = rfm.groupby('segment')['monetary'].mean().sort_values(ascending=False).index
labels = ['High-Value Loyal','Frequent','Promotion-Driven','Occasional','At-Risk']
rfm['segment_name'] = rfm['segment'].map({seg: labels[i] for i, seg in enumerate(seg_order)})

rfm.to_csv(f'{OUT}/customer_segments_python.csv')
print("\n📊 Segment summary:")
print(rfm.groupby('segment_name')[['recency','frequency','monetary']].mean().round(2))
print(f"\n✅ Saved: python_pipeline/customer_segments_python.csv")