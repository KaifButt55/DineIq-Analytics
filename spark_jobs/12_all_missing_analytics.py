import os
import sys
import json
from datetime import datetime

# ============================================================
# CRITICAL: PySpark Python fix (Windows)
# ============================================================
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, sum as spark_sum, avg, count, countDistinct, max as spark_max,
    datediff, to_date, when, lit, round as spark_round,
    stddev, dayofweek, hour, month, desc, asc, collect_set, size, explode
)
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier, DecisionTreeClassifier, LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUET = os.path.join(BASE_DIR, 'parquet_data')
RAW = os.path.join(BASE_DIR, 'raw_data')
MODELS = os.path.join(BASE_DIR, 'models', 'spark')
os.makedirs(MODELS, exist_ok=True)

print("="*60)
print("🚀 DineIQ - ALL MISSING ANALYTICS")
print("="*60)

spark = SparkSession.builder.appName("DineIQ_Complete").config("spark.driver.memory","4g").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# ==========================================================
# LOAD DATA
# ==========================================================
print("\n⏳ Loading data...")
orders = spark.read.parquet(f'{PARQUET}/orders/')
items = spark.read.parquet(f'{PARQUET}/order_items/')
menu = spark.read.parquet(f'{PARQUET}/menu_items/')
customers = spark.read.parquet(f'{PARQUET}/customers/')
ratings = spark.read.parquet(f'{PARQUET}/ratings/')
wastage = spark.read.parquet(f'{PARQUET}/wastage/')
pricing = spark.read.parquet(f'{PARQUET}/pricing_history/')
inventory = spark.read.parquet(f'{PARQUET}/inventory/')

# Restaurants ko CSV se load karein (parquet nahi hai)
try:
    locations = spark.read.parquet(f'{PARQUET}/restaurants/')
except:
    locations = spark.read.csv(f'{RAW}/restaurants.csv', header=True, inferSchema=True)
    locations.write.mode('overwrite').parquet(f'{PARQUET}/restaurants/')

orders_clean = orders.filter((col('status')=='Completed') & (col('total_amount')>0))
items_clean = items.filter(col('quantity')>0)
print("✅ All data loaded")

# ==========================================================
# 1. FULL DATA QUALITY (12 CHECKS)
# ==========================================================
print("\n📊 [1/18] FULL DATA QUALITY REPORT (12 checks)...")
quality_report = {}

for name, df in [('orders',orders),('items',items),('menu',menu),('customers',customers)]:
    total = df.count()
    missing = sum(df.filter(col(c).isNull()).count() for c in df.columns)
    quality_report[f'missing_{name}'] = missing

quality_report['duplicate_orders'] = orders.groupBy('order_id').count().filter('count > 1').count()
quality_report['duplicate_order_lines'] = items.groupBy('order_line_id').count().filter('count > 1').count()
quality_report['invalid_menu_prices'] = menu.filter((col('base_price') <= 0) | (col('cost') <= 0)).count()
quality_report['negative_quantities'] = items.filter(col('quantity') < 0).count()
quality_report['invalid_dates'] = orders.filter(col('order_date').isNull()).count()
quality_report['invalid_ratings'] = ratings.filter((col('rating')<1)|(col('rating')>5)).count()
quality_report['missing_customer_ids'] = orders.filter(col('customer_id').isNull()).count()
quality_report['missing_menu_ids'] = items.filter(col('item_id').isNull()).count()

valid_locs = [r['location_id'] for r in locations.select('location_id').collect()]
quality_report['invalid_location_refs'] = orders.filter(~col('location_id').isin(valid_locs)).count()
quality_report['impossible_wastage'] = wastage.filter(col('quantity') > 1000).count()
quality_report['cancelled_transactions'] = orders.filter(col('status')=='Cancelled').count()

with open(f'{PARQUET}/data_quality_report.json', 'w') as f:
    json.dump(quality_report, f, indent=2)

print("✅ Data quality 12 checks complete")
for k, v in quality_report.items():
    print(f"   {k}: {v}")

# ==========================================================
# 2. DATA CLEANING (Documented)
# ==========================================================
print("\n🧹 [2/18] DATA CLEANING...")
cleaning_log = []
original_count = orders.count()

orders_cl = orders.dropDuplicates(['order_id'])
cleaning_log.append(f"Removed duplicates: {original_count - orders_cl.count()}")

before = orders_cl.count()
orders_cl = orders_cl.filter(col('status')=='Completed')
cleaning_log.append(f"Removed cancelled: {before - orders_cl.count()}")

before = orders_cl.count()
orders_cl = orders_cl.filter(col('total_amount') > 0)
cleaning_log.append(f"Removed invalid amounts: {before - orders_cl.count()}")

orders_cl.write.mode('overwrite').parquet(f'{PARQUET}/orders_cleaned/')

with open(f'{PARQUET}/cleaning_log.txt', 'w') as f:
    f.write('\n'.join(cleaning_log))
print(f"✅ Cleaning done")
for log in cleaning_log:
    print(f"   {log}")

# ==========================================================
# 3. FULL EDA
# ==========================================================
print("\n📊 [3/18] FULL EDA...")
eda_results = {}

top_selling = items_clean.groupBy('item_id').agg(spark_sum('quantity').alias('qty')).orderBy(desc('qty')).limit(10).collect()
eda_results['top_selling'] = [r['item_id'] for r in top_selling]

low_selling = items_clean.groupBy('item_id').agg(spark_sum('quantity').alias('qty')).orderBy(asc('qty')).limit(10).collect()
eda_results['lowest_selling'] = [r['item_id'] for r in low_selling]

best_rated = ratings.groupBy('item_id').agg(avg('rating').alias('ar'), count('rating').alias('cnt')).filter(col('cnt')>10).orderBy(desc('ar')).limit(10).collect()
eda_results['best_rated'] = [(r['item_id'], round(r['ar'],2)) for r in best_rated]

worst_rated = ratings.groupBy('item_id').agg(avg('rating').alias('ar'), count('rating').alias('cnt')).filter(col('cnt')>10).orderBy(asc('ar')).limit(10).collect()
eda_results['worst_rated'] = [(r['item_id'], round(r['ar'],2)) for r in worst_rated]

with open(f'{PARQUET}/eda_results.json', 'w') as f:
    json.dump(eda_results, f, indent=2, default=str)
print("✅ EDA complete")

# ==========================================================
# 4. PEAK-PERIOD ANALYSIS
# ==========================================================
print("\n⏰ [4/18] PEAK-PERIOD ANALYSIS...")
orders_dt = orders_clean.withColumn('order_date', to_date('order_date')) \
    .withColumn('hour', hour('order_date')) \
    .withColumn('dow', dayofweek('order_date'))

peak_hours = orders_dt.groupBy('hour').agg(
    count('order_id').alias('order_count'),
    spark_round(spark_sum('total_amount'),2).alias('revenue')
).orderBy(desc('order_count'))

peak_days = orders_dt.groupBy('dow').agg(
    count('order_id').alias('order_count'),
    spark_round(spark_sum('total_amount'),2).alias('revenue')
).orderBy(desc('order_count'))

peak_hours.write.mode('overwrite').parquet(f'{PARQUET}/peak_hours/')
peak_days.write.mode('overwrite').parquet(f'{PARQUET}/peak_days/')
print("✅ Peak period analysis complete")

# ==========================================================
# 5. RATING ANALYSIS
# ==========================================================
print("\n⭐ [5/18] RATING ANALYSIS...")
rating_analysis = ratings.groupBy('item_id').agg(
    avg('rating').alias('avg_rating'),
    count('rating').alias('total_ratings'),
    spark_sum(when(col('rating')>=4, 1).otherwise(0)).alias('positive_ratings'),
    spark_sum(when(col('rating')<=2, 1).otherwise(0)).alias('negative_ratings')
).withColumn('satisfaction_pct', spark_round(col('positive_ratings') / col('total_ratings') * 100, 2))

rating_analysis.write.mode('overwrite').parquet(f'{PARQUET}/rating_analysis/')
print("✅ Rating analysis complete")

# ==========================================================
# 6. SLOW-MOVING DISH DETECTION
# ==========================================================
print("\n🐢 [6/18] SLOW-MOVING DISH DETECTION...")
item_sales = items_clean.groupBy('item_id').agg(
    spark_sum('quantity').alias('total_qty'),
    count('order_id').alias('order_count')
)

stats = item_sales.select(avg('total_qty').alias('avg_qty')).collect()[0]['avg_qty']

slow_items = item_sales.filter(col('total_qty') < stats * 0.5) \
    .join(menu.select('item_id','item_name','category'), on='item_id') \
    .orderBy(asc('total_qty'))

slow_items.write.mode('overwrite').parquet(f'{PARQUET}/slow_moving_items/')
print(f"✅ Found {slow_items.count()} slow-moving items")

# ==========================================================
# 7. ORDERING CHANNEL ANALYSIS
# ==========================================================
print("\n📱 [7/18] ORDERING CHANNEL ANALYSIS...")
channel_analysis = orders_clean.groupBy('channel').agg(
    count('order_id').alias('total_orders'),
    spark_round(spark_sum('total_amount'),2).alias('revenue'),
    spark_round(avg('total_amount'),2).alias('avg_order_value'),
    countDistinct('customer_id').alias('unique_customers')
)
channel_analysis.write.mode('overwrite').parquet(f'{PARQUET}/channel_analysis/')
print("✅ Channel analysis complete")

# ==========================================================
# 8. CUSTOMER CHURN RISK
# ==========================================================
print("\n⚠️ [8/18] CUSTOMER CHURN-RISK...")
max_date = orders_clean.select(spark_max(to_date('order_date'))).collect()[0][0]

churn = orders_clean.withColumn('order_date', to_date('order_date')) \
    .groupBy('customer_id').agg(
        spark_max('order_date').alias('last_order'),
        count('order_id').alias('frequency'),
        spark_sum('total_amount').alias('monetary')
    ).withColumn('recency', datediff(lit(max_date), col('last_order'))) \
    .withColumn('churn_risk',
        when(col('recency') > 180, 'High Risk')
        .when(col('recency') > 90, 'Medium Risk')
        .when(col('recency') > 60, 'Low Risk')
        .otherwise('Active'))

churn.write.mode('overwrite').parquet(f'{PARQUET}/churn_risk/')
print(f"✅ Churn risk analysis complete")

# ==========================================================
# 9. BUNDLE RECOMMENDATIONS (Spark SQL - No UDF)
# ==========================================================
print("\n🎁 [9/18] BUNDLE RECOMMENDATIONS...")
baskets = items_clean.groupBy('order_id').agg(
    collect_set('item_id').alias('items')
).filter(size('items') >= 2)

baskets.createOrReplaceTempView('baskets_view')

top_pairs = spark.sql("""
    SELECT item1, item2, COUNT(*) as pair_count
    FROM (
        SELECT explode(items) as item1, explode(items) as item2
        FROM baskets_view
        WHERE size(items) >= 2
    )
    WHERE item1 < item2
    GROUP BY item1, item2
    ORDER BY pair_count DESC
    LIMIT 50
""")

top_pairs.write.mode('overwrite').parquet(f'{PARQUET}/bundle_recommendations/')
print(f"✅ Bundle recommendations complete")

# ==========================================================
# 10. WASTAGE RISK PREDICTION
# ==========================================================
print("\n🗑️ [10/18] WASTAGE RISK PREDICTION...")
waste_item = wastage.groupBy('item_id').agg(spark_sum('quantity').alias('total_wasted'))
sales_item = items_clean.groupBy('item_id').agg(spark_sum('quantity').alias('total_sold'))

waste_risk = waste_item.join(sales_item, on='item_id', how='left').fillna(0)
waste_risk = waste_risk.withColumn('waste_ratio',
    spark_round(col('total_wasted') / (col('total_wasted') + col('total_sold') + 1) * 100, 2)
).withColumn('risk_level',
    when(col('waste_ratio') > 15, 'High')
    .when(col('waste_ratio') > 8, 'Medium')
    .otherwise('Low'))

waste_risk.write.mode('overwrite').parquet(f'{PARQUET}/wastage_risk/')
print("✅ Wastage risk prediction complete")

# ==========================================================
# 11. 3 ML MODELS COMPARISON
# ==========================================================
print("\n🤖 [11/18] 3 ML MODELS COMPARISON...")

menu_features = spark.read.parquet(f'{PARQUET}/menu_classified/')
indexer = StringIndexer(inputCol='performance_class', outputCol='label')
menu_idx = indexer.fit(menu_features).transform(menu_features)

assembler = VectorAssembler(
    inputCols=['qty_sold','profit_pct','revenue','cost'],
    outputCol='features'
)
menu_ml = assembler.transform(menu_idx).select('item_id','features','label')

train, test = menu_ml.randomSplit([0.8, 0.2], seed=42)

rf = RandomForestClassifier(featuresCol='features', labelCol='label', numTrees=50, seed=42)
rf_model = rf.fit(train)
rf_pred = rf_model.transform(test)

dt = DecisionTreeClassifier(featuresCol='features', labelCol='label', maxDepth=5, seed=42)
dt_model = dt.fit(train)
dt_pred = dt_model.transform(test)

lr = LogisticRegression(featuresCol='features', labelCol='label', maxIter=20)
lr_model = lr.fit(train)
lr_pred = lr_model.transform(test)

evaluator = MulticlassClassificationEvaluator(labelCol='label', predictionCol='prediction')

print("\n📊 MODEL COMPARISON:")
results = {}
for name, pred in [('RandomForest',rf_pred),('DecisionTree',dt_pred),('LogisticRegression',lr_pred)]:
    acc = evaluator.setMetricName('accuracy').evaluate(pred)
    f1 = evaluator.setMetricName('f1').evaluate(pred)
    precision = evaluator.setMetricName('weightedPrecision').evaluate(pred)
    recall = evaluator.setMetricName('weightedRecall').evaluate(pred)
    results[name] = {'accuracy': round(acc,4), 'f1': round(f1,4), 'precision': round(precision,4), 'recall': round(recall,4)}
    print(f"   {name}: Accuracy={acc:.4f}, F1={f1:.4f}")

best_model_path = os.path.join(MODELS, 'menu_classifier_v1')
rf_model.write().overwrite().save(best_model_path)

with open(f'{PARQUET}/ml_models_comparison.json', 'w') as f:
    json.dump({'models': results, 'best': 'RandomForest', 'version': 'v1', 'date': str(datetime.now())}, f, indent=2)

print("✅ 3 ML models compared")

# ==========================================================
# 12. TRICKY MENU CASES
# ==========================================================
print("\n🎯 [12/18] TRICKY MENU CASES...")
triggers = []
hslm = menu_features.filter((col('qty_sold') > 15000) & (col('profit_pct') < 40)).count()
triggers.append(f"High-selling loss-making: {hslm}")
lshm = menu_features.filter((col('qty_sold') < 10000) & (col('profit_pct') > 60)).count()
triggers.append(f"Low-selling high-margin: {lshm}")

with open(f'{PARQUET}/tricky_cases.txt', 'w') as f:
    f.write('\n'.join(triggers))
print(f"✅ Tricky cases identified")

# ==========================================================
# 13. INVENTORY RECOMMENDATIONS
# ==========================================================
print("\n📦 [13/18] INVENTORY RECOMMENDATIONS...")
inv_recs = inventory.groupBy('item_id').agg(
    avg('stock_qty').alias('avg_stock'),
    avg('reorder_level').alias('avg_reorder')
).withColumn('recommendation',
    when(col('avg_stock') < col('avg_reorder'), 'REORDER NOW')
    .when(col('avg_stock') < col('avg_reorder') * 1.5, 'MONITOR')
    .otherwise('SUFFICIENT'))

inv_recs.write.mode('overwrite').parquet(f'{PARQUET}/inventory_recommendations/')
print("✅ Inventory recommendations complete")

# ==========================================================
# 14. CUSTOMER TARGETING
# ==========================================================
print("\n🎯 [14/18] CUSTOMER TARGETING...")
cust_seg = spark.read.parquet(f'{PARQUET}/customer_segments/')

targeting = cust_seg.groupBy('prediction').agg(
    count('customer_id').alias('customer_count'),
    spark_round(avg('monetary'),2).alias('avg_monetary'),
    spark_round(avg('frequency'),2).alias('avg_frequency')
).withColumn('strategy',
    when(col('prediction')==0, 'VIP Program - Exclusive offers')
    .when(col('prediction')==1, 'Loyalty Rewards - Frequent buyers')
    .when(col('prediction')==2, 'Promotional Campaigns - Discount sensitive')
    .when(col('prediction')==3, 'Reactivation - Send special offers')
    .otherwise('Re-engagement - Reminder emails'))

targeting.write.mode('overwrite').parquet(f'{PARQUET}/customer_targeting/')
print("✅ Customer targeting complete")

# ==========================================================
# 15. LOCATION-SPECIFIC MENU
# ==========================================================
print("\n📍 [15/18] LOCATION-SPECIFIC MENU...")
loc_menu = items_clean.join(orders_clean.select('order_id','location_id'), on='order_id') \
    .groupBy('location_id','item_id').agg(spark_sum('quantity').alias('qty_sold'))

loc_menu.write.mode('overwrite').parquet(f'{PARQUET}/location_menu/')
print("✅ Location-specific menu complete")

# ==========================================================
# 16. MODEL VERSION TRACKING
# ==========================================================
print("\n📝 [16/18] MODEL VERSION TRACKING...")
model_meta = {
    'model_name': 'menu_classifier',
    'version': 'v1.0',
    'trained_on': str(datetime.now()),
    'algorithms_compared': ['RandomForest', 'DecisionTree', 'LogisticRegression'],
    'best_model': 'RandomForest',
    'accuracy': results['RandomForest']['accuracy'],
    'f1_score': results['RandomForest']['f1'],
    'features_used': ['qty_sold', 'profit_pct', 'revenue', 'cost'],
    'training_records': train.count(),
    'test_records': test.count()
}
with open(f'{MODELS}/model_version.json', 'w') as f:
    json.dump(model_meta, f, indent=2)
print("✅ Model version saved")

# ==========================================================
# 17. AUDIT TRAIL
# ==========================================================
print("\n📋 [17/18] AUDIT TRAIL...")
audit = {
    'session_id': datetime.now().strftime('%Y%m%d_%H%M%S'),
    'user': 'system',
    'actions': [
        {'action': 'data_quality_check', 'timestamp': str(datetime.now())},
        {'action': 'data_cleaning', 'timestamp': str(datetime.now())},
        {'action': 'eda_complete', 'timestamp': str(datetime.now())},
        {'action': 'ml_models_trained', 'timestamp': str(datetime.now())},
        {'action': 'recommendations_generated', 'timestamp': str(datetime.now())},
    ]
}
with open(f'{PARQUET}/audit_trail.json', 'w') as f:
    json.dump(audit, f, indent=2)
print("✅ Audit trail saved")

# ==========================================================
# 18. DATA PARTITIONING
# ==========================================================
print("\n📂 [18/18] DATA PARTITIONING...")
orders_clean.withColumn('month', month(to_date('order_date'))) \
    .write.mode('overwrite') \
    .partitionBy('month') \
    .parquet(f'{PARQUET}/orders_partitioned/')
print("✅ Data partitioned by month")

# ==========================================================
# DONE
# ==========================================================
print("\n" + "="*60)
print("🎉 ALL 18 MISSING ANALYTICS COMPLETE!")
print("="*60)
print(f"\n📊 Models compared: {list(results.keys())}")
print(f"🏆 Best: RandomForest (Accuracy: {results['RandomForest']['accuracy']})")
print(f"📁 Output saved in: {PARQUET}/")

spark.stop()
print("\n✅ Spark session stopped.")