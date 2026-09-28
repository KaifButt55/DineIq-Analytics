import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as spark_sum, avg, count, round as spark_round

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUET = os.path.join(BASE_DIR, 'parquet_data')

spark = SparkSession.builder.appName("DineIQ_Location").config("spark.driver.memory","4g").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

orders = spark.read.parquet(f'{PARQUET}/orders/').filter(col('status') == 'Completed')
items = spark.read.parquet(f'{PARQUET}/order_items/')
menu = spark.read.parquet(f'{PARQUET}/menu_items/')
wastage = spark.read.parquet(f'{PARQUET}/wastage/')

print("📊 LOCATION COMPARISON ANALYSIS...")

# Revenue per location
loc_perf = orders.groupBy('location_id').agg(
    spark_sum('total_amount').alias('revenue'),
    count('order_id').alias('total_orders'),
    avg('total_amount').alias('avg_order_value'),
    count('customer_id').alias('unique_customers')
)

# Profit per location
profit = items.join(orders.select('order_id','location_id'), on='order_id') \
    .join(menu.select('item_id','cost'), on='item_id') \
    .groupBy('location_id').agg(
        spark_round(spark_sum('line_total'), 2).alias('gross_revenue'),
        spark_round(spark_sum(col('quantity') * col('cost')), 2).alias('total_cost')
    ).withColumn('profit', col('gross_revenue') - col('total_cost'))

# Wastage per location
waste = wastage.groupBy('location_id').agg(
    spark_sum('quantity').alias('total_wasted_qty'),
    spark_round(spark_sum('cost'), 2).alias('wastage_cost')
)

# Final merge
final = loc_perf.join(profit, on='location_id', how='left') \
    .join(waste, on='location_id', how='left').fillna(0)

final = final.withColumn('profit_margin_pct', 
    spark_round((col('profit') / col('revenue')) * 100, 2)
)

print("\n🏆 TOP LOCATIONS BY REVENUE:")
final.orderBy(col('revenue').desc()).show(10, truncate=False)

print("\n📉 WORST LOCATIONS BY PROFIT MARGIN:")
final.orderBy(col('profit_margin_pct').asc()).show(5, truncate=False)

final.write.mode('overwrite').parquet(f'{PARQUET}/location_analysis/')
print("\n🎉 LOCATION COMPARISON COMPLETE!")
spark.stop()