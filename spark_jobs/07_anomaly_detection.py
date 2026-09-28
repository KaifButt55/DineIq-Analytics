import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, stddev, sum as spark_sum, count, to_date, abs as spark_abs

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUET = os.path.join(BASE_DIR, 'parquet_data')

print("⏳ Spark Session...")
spark = SparkSession.builder.appName("DineIQ_Anomaly").config("spark.driver.memory","4g").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

orders = spark.read.parquet(f'{PARQUET}/orders/')
ratings = spark.read.parquet(f'{PARQUET}/ratings/')

# ==========================================================
# 1. SALES ANOMALY DETECTION (Z-score method)
# ==========================================================
print("\n📊 SALES ANOMALY DETECTION...")
daily_sales = orders.withColumn('order_date', to_date('order_date')) \
    .groupBy('order_date').agg(
        spark_sum('total_amount').alias('daily_revenue'),
        count('order_id').alias('order_count')
    )

stats = daily_sales.select(avg('daily_revenue').alias('mean'), stddev('daily_revenue').alias('std')).collect()[0]
mean_rev, std_rev = stats['mean'], stats['std']

anomalies = daily_sales.withColumn('z_score', 
    spark_abs(col('daily_revenue') - mean_rev) / std_rev
).filter(col('z_score') > 2.5)

print(f"✅ Mean Revenue: ${mean_rev:.2f}, StdDev: ${std_rev:.2f}")
print(f"✅ Detected {anomalies.count()} anomalous days")
anomalies.orderBy(col('z_score').desc()).show(10)

anomalies.write.mode('overwrite').parquet(f'{PARQUET}/sales_anomalies/')

# ==========================================================
# 2. RATING ANOMALY DETECTION
# ==========================================================
print("\n📊 RATING ANOMALY DETECTION...")
item_ratings = ratings.groupBy('item_id').agg(
    avg('rating').alias('avg_rating'),
    count('rating').alias('rating_count'),
    stddev('rating').alias('rating_std')
)

# Low std = suspicious (too many identical ratings)
suspicious = item_ratings.filter(
    (col('rating_count') > 100) & (col('rating_std') < 0.5)
)
print(f"✅ Suspicious items (identical ratings): {suspicious.count()}")
suspicious.show(5)

suspicious.write.mode('overwrite').parquet(f'{PARQUET}/rating_anomalies/')

print("\n🎉 ANOMALY DETECTION COMPLETE!")
spark.stop()