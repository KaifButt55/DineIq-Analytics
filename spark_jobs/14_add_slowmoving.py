import os
import sys

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as spark_sum, count, asc

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUET = os.path.join(BASE_DIR, 'parquet_data')

print("🚀 Fixing Slow-Moving Dishes...")
spark = SparkSession.builder.appName("FixSlowMoving").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# Data Load
items = spark.read.parquet(f'{PARQUET}/order_items/').filter(col('quantity') > 0)
menu = spark.read.parquet(f'{PARQUET}/menu_items/')

# Sales calculate karein
item_sales = items.groupBy('item_id').agg(
    spark_sum('quantity').alias('total_qty'),
    count('order_id').alias('order_count')
)

# ❌ Purana logic (threshold) hata dein
# ✅ Naya logic: Bottom 20 items with lowest sales
slow_items = item_sales.orderBy(asc('total_qty')).limit(20) \
    .join(menu.select('item_id', 'item_name', 'category'), on='item_id') \
    .orderBy(asc('total_qty'))

# Save
slow_items.write.mode('overwrite').parquet(f'{PARQUET}/slow_moving_items/')
print(f"✅ Fixed! Found {slow_items.count()} slow-moving items")
print("🏆 Bottom 5:")
slow_items.select('item_name', 'total_qty').show(5)

spark.stop()
print("\n✅ Done!")