import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, collect_set, size, lit
from pyspark.ml.fpm import FPGrowth

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUET_PATH = os.path.join(BASE_DIR, 'parquet_data')

print("⏳ Spark Session start...")
spark = SparkSession.builder.appName("DineIQ_Basket").config("spark.driver.memory","4g").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# Load
print("⏳ Loading data...")
orders = spark.read.parquet(f'{PARQUET_PATH}/orders/').filter(col('status')=='Completed')
order_items = spark.read.parquet(f'{PARQUET_PATH}/order_items/')

# Basket banayein
print("📊 Basket analysis...")
baskets = order_items.join(orders.select('order_id','status'), on='order_id') \
    .groupBy('order_id').agg(collect_set('item_id').alias('items')) \
    .filter(size('items') >= 2)

print(f"✅ Baskets with 2+ items: {baskets.count()}")

# FP-Growth
print("\n⏳ FP-Growth algorithm...")
fp = FPGrowth(itemsCol='items', minSupport=0.01, minConfidence=0.3)
model = fp.fit(baskets)

print("\n🏆 FREQUENT ITEMSETS (Top 10):")
model.freqItemsets.orderBy(col('freq').desc()).show(10, truncate=False)

print("\n🔗 ASSOCIATION RULES (Top 10):")
model.associationRules.orderBy(col('lift').desc()).show(10, truncate=False)

# Save
output_path = os.path.join(PARQUET_PATH, 'basket_rules')
model.associationRules.write.mode('overwrite').parquet(output_path)
print(f"\n✅ Saved: {output_path}")
print("🎉 MARKET BASKET COMPLETE!")
spark.stop()
