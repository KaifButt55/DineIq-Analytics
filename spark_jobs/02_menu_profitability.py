import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, avg, when, round as spark_round, count

# Path Setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUET_PATH = os.path.join(BASE_DIR, 'parquet_data')

print("⏳ Spark Session start ho raha hai...")
spark = SparkSession.builder \
    .appName("DineIQ_MenuProfitability") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
print("✅ Spark Session ready!")

# 1. Data Load Karein
print("\n⏳ Parquet data load ho raha hai...")
orders = spark.read.parquet(f'{PARQUET_PATH}/orders/')
order_items = spark.read.parquet(f'{PARQUET_PATH}/order_items/')
menu = spark.read.parquet(f'{PARQUET_PATH}/menu_items/')

print(f"✅ Orders loaded: {orders.count()}")
print(f"✅ Order Items loaded: {order_items.count()}")
print(f"✅ Menu loaded: {menu.count()}")

# 2. Filtering (Sirf Completed Orders aur Valid Quantities)
orders_clean = orders.filter((col('status') == 'Completed') & (col('total_amount') > 0))
items_clean = order_items.filter(col('quantity') > 0)

# 3. Join Karein
print("\n⏳ Data join ho raha hai...")
df = items_clean.join(orders_clean, on='order_id', how='inner') \
                .join(menu, on='item_id', how='inner')

print(f"✅ Joined data: {df.count()} records")

# 4. Menu Profitability Calculate Karein
print("\n📊 Menu Profitability calculate ho rahi hai...")

menu_features = df.groupBy('item_id', 'item_name', 'category', 'cost', 'base_price').agg(
    sum('quantity').alias('qty_sold'),
    spark_round(sum('line_total'), 2).alias('revenue'),
    count('order_id').alias('order_count')
)

# Cost aur Profit calculate karein
menu_features = menu_features \
    .withColumn('total_cost', spark_round(col('qty_sold') * col('cost'), 2)) \
    .withColumn('profit', spark_round(col('revenue') - col('total_cost'), 2)) \
    .withColumn('profit_pct', 
                spark_round(
                    when(col('revenue') > 0, (col('profit') / col('revenue')) * 100).otherwise(0), 
                    2
                )
    )

# 5. Classification (Profit Driver, Volume Driver, etc.)
print("\n⏳ Menu items classify ho rahe hain...")

# Average nikaalein thresholds ke liye
stats = menu_features.select(avg('qty_sold').alias('avg_qty'), avg('profit_pct').alias('avg_profit')).collect()[0]
avg_qty = stats['avg_qty']
avg_profit = stats['avg_profit']
print(f"📊 Average Qty Sold: {avg_qty:.2f}")
print(f"📊 Average Profit %: {avg_profit:.2f}%")

classified = menu_features.withColumn(
    'performance_class',
    when((col('qty_sold') >= avg_qty) & (col('profit_pct') >= avg_profit), 'Profit Driver')
    .when((col('qty_sold') >= avg_qty) & (col('profit_pct') < avg_profit), 'Volume Driver')
    .when((col('qty_sold') < avg_qty) & (col('profit_pct') >= avg_profit), 'Hidden Opportunity')
    .otherwise('Low Performer')
)

# 6. Results Dikhayein
print("\n🏆 TOP 10 PROFITABLE ITEMS:")
classified.orderBy(col('profit').desc()).select('item_name', 'qty_sold', 'revenue', 'profit', 'profit_pct', 'performance_class').show(10, truncate=False)

print("\n📊 PERFORMANCE CLASSIFICATION SUMMARY:")
classified.groupBy('performance_class').count().orderBy('count', ascending=False).show()

# 7. Save to Parquet
output_path = os.path.join(PARQUET_PATH, 'menu_classified')
classified.write.mode('overwrite').parquet(output_path)
print(f"\n✅ Menu classification saved to: {output_path}")

print("\n" + "="*60)
print("🎉 MENU PROFITABILITY COMPLETE!")
print("="*60)

spark.stop()