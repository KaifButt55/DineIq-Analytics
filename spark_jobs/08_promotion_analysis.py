import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as spark_sum, avg, count, round as spark_round

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUET = os.path.join(BASE_DIR, 'parquet_data')

spark = SparkSession.builder.appName("DineIQ_Promotion").config("spark.driver.memory","4g").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

orders = spark.read.parquet(f'{PARQUET}/orders/')
items = spark.read.parquet(f'{PARQUET}/order_items/')
menu = spark.read.parquet(f'{PARQUET}/menu_items/')
promos = spark.read.parquet(f'{PARQUET}/promotions/')

print("📊 PROMOTION EFFECTIVENESS ANALYSIS...")

# Join promotions with orders/items
df = items.join(orders.select('order_id','order_date','status'), on='order_id') \
    .filter(col('status') == 'Completed') \
    .join(menu.select('item_id','cost'), on='item_id') \
    .withColumn('profit', col('line_total') - (col('quantity') * col('cost')))

# Promoted vs Non-promoted items
promo_items = promos.select('item_id').distinct()
promoted = df.join(promo_items, on='item_id', how='inner')
non_promoted = df.join(promo_items, on='item_id', how='left_anti')

print("\n📊 PROMOTED ITEMS PERFORMANCE:")
promo_perf = promoted.groupBy('item_id').agg(
    spark_sum('quantity').alias('qty_sold'),
    spark_round(spark_sum('line_total'), 2).alias('revenue'),
    spark_round(spark_sum('profit'), 2).alias('total_profit')
)
promo_perf.show(10)

# PROMO TRAP DETECTION: Sales up but profit down
print("\n🚨 PROMO TRAP DETECTION (High Sales, Low Profit):")
promo_traps = promo_perf.filter(
    (col('qty_sold') > 5000) & (col('total_profit') / col('revenue') < 0.4)
)
print(f"✅ Detected {promo_traps.count()} potential promo traps")
promo_traps.show(10)

promo_traps.write.mode('overwrite').parquet(f'{PARQUET}/promo_traps/')
print("\n🎉 PROMOTION ANALYSIS COMPLETE!")
spark.stop()