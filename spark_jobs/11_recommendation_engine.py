import os
import sys
from datetime import datetime

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lit, concat, round as spark_round

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUET = os.path.join(BASE_DIR, 'parquet_data')

spark = SparkSession.builder.appName("DineIQ_Recommendations").config("spark.driver.memory","4g").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

print("📊 RECOMMENDATION ENGINE...")

menu = spark.read.parquet(f'{PARQUET}/menu_classified/')

# Wastage risk se wastage_pct lein (agar wastage_analysis nahi hai to)
try:
    wastage = spark.read.parquet(f'{PARQUET}/wastage_analysis/').select('item_id','wastage_pct')
    print("✅ Using wastage_analysis")
except:
    try:
        wastage = spark.read.parquet(f'{PARQUET}/wastage_risk/') \
            .withColumnRenamed('waste_ratio', 'wastage_pct') \
            .select('item_id','wastage_pct')
        print("✅ Using wastage_risk (fallback)")
    except:
        wastage = None
        print("⚠️ No wastage data, using 0")

if wastage is not None:
    df = menu.join(wastage, on='item_id', how='left').fillna(0)
else:
    df = menu.withColumn('wastage_pct', lit(0.0))

# Generate recommendations
recommendations = df.withColumn(
    'recommendation',
    when((col('performance_class') == 'Hidden Opportunity') & (col('profit_pct') > 60), 
         'PROMOTE - High margin, low visibility')
    .when((col('performance_class') == 'Low Performer') & (col('wastage_pct') > 15), 
         'REVIEW - Low performer with high wastage')
    .when((col('performance_class') == 'Low Performer') & (col('profit_pct') < 30), 
         'REMOVE or REDESIGN - Low margin')
    .when((col('performance_class') == 'Volume Driver') & (col('profit_pct') < 40), 
         'REPRICE - High volume, low margin')
    .when((col('performance_class') == 'Profit Driver'), 
         'MAINTAIN - Strong performer')
    .otherwise('MONITOR')
)

recommendations = recommendations.withColumn(
    'priority',
    when(col('profit_pct') > 65, 'Critical')
    .when(col('profit_pct') > 55, 'High')
    .when(col('profit_pct') > 45, 'Medium')
    .otherwise('Low')
)

recommendations = recommendations.withColumn(
    'evidence',
    concat(
        lit("Margin: "), spark_round(col('profit_pct'), 2), lit("% | "),
        lit("Sold: "), col('qty_sold'), lit(" | "),
        lit("Wastage: "), spark_round(col('wastage_pct'), 2), lit("%")
    )
)

print("\n💡 SAMPLE RECOMMENDATIONS:")
recommendations.select('item_name','performance_class','recommendation','priority','evidence') \
    .orderBy('priority').show(15, truncate=False)

print("\n📊 RECOMMENDATIONS BY PRIORITY:")
recommendations.groupBy('priority').count().show()

recommendations.write.mode('overwrite').parquet(f'{PARQUET}/recommendations/')
print(f"\n✅ Saved: {PARQUET}/recommendations/")
print("🎉 RECOMMENDATION ENGINE COMPLETE!")
spark.stop()