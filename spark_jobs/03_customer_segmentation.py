import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, max, datediff, current_date, to_date, lit
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUET_PATH = os.path.join(BASE_DIR, 'parquet_data')

print("⏳ Spark Session start...")
spark = SparkSession.builder.appName("DineIQ_CustomerSeg").config("spark.driver.memory","4g").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
print("✅ Ready!")

# Data Load
print("\n⏳ Loading data...")
orders = spark.read.parquet(f'{PARQUET_PATH}/orders/').filter(col('status')=='Completed')

# RFM Calculate
print("📊 RFM calculate ho raha hai...")
rfm = orders.withColumn('order_date', to_date('order_date')) \
    .groupBy('customer_id').agg(
        datediff(max('order_date'), max('order_date')).alias('dummy'),  # placeholder
        max('order_date').alias('last_order'),
        sum('total_amount').alias('monetary'),
        sum(lit(1)).alias('frequency')
    )

# Recency calculate karein
max_date = orders.select(max(to_date('order_date'))).collect()[0][0]
rfm = orders.withColumn('order_date', to_date('order_date')) \
    .groupBy('customer_id').agg(
        datediff(lit(max_date), max('order_date')).alias('recency'),
        sum(lit(1)).alias('frequency'),
        sum('total_amount').alias('monetary')
    )

print(f"✅ RFM calculated: {rfm.count()} customers")
rfm.show(5)

# K-Means Clustering
print("\n⏳ K-Means clustering...")
assembler = VectorAssembler(inputCols=['recency','frequency','monetary'], outputCol='features_raw')
rfm_vec = assembler.transform(rfm)

scaler = StandardScaler(inputCol='features_raw', outputCol='features', withMean=True, withStd=True)
scaler_model = scaler.fit(rfm_vec)
rfm_scaled = scaler_model.transform(rfm_vec)

# Train K-Means (5 clusters)
kmeans = KMeans(featuresCol='features', k=5, seed=42)
model = kmeans.fit(rfm_scaled)
predictions = model.transform(rfm_scaled)

# Evaluate
evaluator = ClusteringEvaluator(featuresCol='features')
silhouette = evaluator.evaluate(predictions)
print(f"✅ Silhouette Score: {silhouette:.4f}")

# Segment Summary
print("\n📊 SEGMENT SUMMARY:")
predictions.groupBy('prediction').agg(
    {'recency':'avg','frequency':'avg','monetary':'avg'}
).orderBy('prediction').show()

# Save
output_path = os.path.join(PARQUET_PATH, 'customer_segments')
predictions.select('customer_id','recency','frequency','monetary','prediction').write.mode('overwrite').parquet(output_path)
print(f"\n✅ Saved: {output_path}")
print("🎉 CUSTOMER SEGMENTATION COMPLETE!")
spark.stop()