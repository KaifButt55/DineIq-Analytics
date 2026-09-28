import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as spark_sum, avg, round as spark_round, count

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUET_PATH = os.path.join(BASE_DIR, 'parquet_data')

print("⏳ Spark Session...")
spark = SparkSession.builder.appName("DineIQ_Wastage").config("spark.driver.memory","4g").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# Load
print("⏳ Loading...")
wastage = spark.read.parquet(f'{PARQUET_PATH}/wastage/')
menu = spark.read.parquet(f'{PARQUET_PATH}/menu_items/')
order_items = spark.read.parquet(f'{PARQUET_PATH}/order_items/')
orders = spark.read.parquet(f'{PARQUET_PATH}/orders/')

# Wastage by item
w = wastage.groupBy('item_id').agg(
    spark_sum('quantity').alias('total_wasted_qty'),
    spark_round(spark_sum('cost'), 2).alias('total_wastage_cost')
)

# Sales by item
s = order_items.join(orders.select('order_id','status'), on='order_id') \
    .filter(col('status')=='Completed') \
    .groupBy('item_id').agg(spark_sum('quantity').alias('total_sold'))

# Merge
df = menu.join(w, on='item_id', how='left').join(s, on='item_id', how='left').fillna(0)
df = df.withColumn('wastage_pct', 
    spark_round(
        col('total_wasted_qty') / (col('total_sold') + col('total_wasted_qty') + 1) * 100,
        2
    )
)

print("\n🏆 TOP 10 HIGH WASTAGE ITEMS:")
df.orderBy(col('total_wastage_cost').desc()).select('item_name','total_wasted_qty','total_wastage_cost','wastage_pct').show(10, truncate=False)

print("\n📊 WASTAGE BY LOCATION:")
wastage.groupBy('location_id').agg(
    spark_sum('quantity').alias('qty'),
    spark_round(spark_sum('cost'),2).alias('cost')
).orderBy(col('cost').desc()).show()

# Save
output = os.path.join(PARQUET_PATH, 'wastage_analysis')
df.write.mode('overwrite').parquet(output)
print(f"✅ Saved: {output}")
print("🎉 WASTAGE ANALYSIS COMPLETE!")
spark.stop()