import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as spark_sum, avg, round as spark_round, when, count

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUET = os.path.join(BASE_DIR, 'parquet_data')

spark = SparkSession.builder.appName("DineIQ_Price").config("spark.driver.memory","4g").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

items = spark.read.parquet(f'{PARQUET}/order_items/')
pricing = spark.read.parquet(f'{PARQUET}/pricing_history/')

print("📊 PRICE SENSITIVITY ANALYSIS...")

# Item price variance
price_stats = pricing.groupBy('item_id').agg(
    avg('new_price').alias('avg_price'),
    count('price_id').alias('num_price_changes')
)

# Item sales
sales = items.groupBy('item_id').agg(
    spark_sum('quantity').alias('qty_sold'),
    avg('unit_price').alias('avg_selling_price')
)

df = sales.join(price_stats, on='item_id', how='left').fillna(0)

# Classify price sensitivity based on price variation vs sales
df = df.withColumn('price_variation_pct',
    when(col('avg_price') > 0, (col('avg_selling_price') - col('avg_price')) / col('avg_price') * 100).otherwise(0)
)

df = df.withColumn('sensitivity_class',
    when(col('price_variation_pct') > 10, 'Highly Price Sensitive')
    .when(col('price_variation_pct') > 5, 'Moderately Price Sensitive')
    .otherwise('Low Price Sensitivity')
)

print("\n📊 PRICE SENSITIVITY CLASSIFICATION:")
df.groupBy('sensitivity_class').count().show()

print("\n🏆 TOP 10 PRICE-SENSITIVE ITEMS:")
df.filter(col('sensitivity_class') == 'Highly Price Sensitive') \
    .orderBy(col('qty_sold').desc()).show(10)

df.write.mode('overwrite').parquet(f'{PARQUET}/price_sensitivity/')
print("\n🎉 PRICE SENSITIVITY COMPLETE!")
spark.stop()