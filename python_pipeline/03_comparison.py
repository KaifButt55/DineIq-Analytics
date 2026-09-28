import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE_DIR, 'python_pipeline')
PARQUET = os.path.join(BASE_DIR, 'parquet_data')

print("⏳ Comparison report...")

# Spark results
spark_seg = pd.read_parquet(f'{PARQUET}/customer_segments/')[['customer_id','prediction']]
spark_seg.columns = ['customer_id','spark_segment']

# Python results
py_seg = pd.read_csv(f'{OUT}/customer_segments_python.csv')[['customer_id','segment_name']]
py_seg.columns = ['customer_id','python_segment']

# Merge
merged = spark_seg.merge(py_seg, on='customer_id', how='inner')
merged['match'] = (merged['spark_segment'].astype(str) == merged['python_segment'].astype(str))

total = len(merged)
matches = merged['match'].sum()
pct = (matches/total)*100

print(f"\n📊 COMPARISON RESULTS:")
print(f"   Total records: {total}")
print(f"   Matches: {matches}")
print(f"   Mismatches: {total - matches}")
print(f"   Agreement %: {pct:.2f}%")

merged.to_csv(f'{OUT}/comparison_report.csv', index=False)
print(f"\n✅ Saved: python_pipeline/comparison_report.csv")