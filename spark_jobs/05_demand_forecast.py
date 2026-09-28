import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, sum as spark_sum, dayofweek, month, dayofyear, lag
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor, LinearRegression, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUET_PATH = os.path.join(BASE_DIR, 'parquet_data')

print("⏳ Spark Session start...")
spark = SparkSession.builder.appName("DineIQ_Forecast").config("spark.driver.memory","4g").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# Data
print("⏳ Loading...")
orders = spark.read.parquet(f'{PARQUET_PATH}/orders/').filter(col('status')=='Completed')
order_items = spark.read.parquet(f'{PARQUET_PATH}/order_items/')

df = order_items.join(orders.select('order_id','order_date'), on='order_id') \
    .withColumn('order_date', to_date('order_date')) \
    .groupBy('order_date').agg(spark_sum('quantity').alias('demand')) \
    .orderBy('order_date')

# Features
w = Window.orderBy('order_date')
df = df.withColumn('dayofweek', dayofweek('order_date')) \
    .withColumn('month', month('order_date')) \
    .withColumn('dayofyear', dayofyear('order_date')) \
    .withColumn('lag_1', lag('demand', 1).over(w)) \
    .withColumn('lag_7', lag('demand', 7).over(w)) \
    .dropna()

# Time-based split (NO leakage)
split_idx = int(df.count() * 0.8)
train = df.limit(split_idx)
test = df.subtract(train)

print(f"✅ Train: {train.count()}, Test: {test.count()}")

# Models
feature_cols = ['dayofweek','month','dayofyear','lag_1','lag_7']
assembler = VectorAssembler(inputCols=feature_cols, outputCol='features')
train_v = assembler.transform(train)
test_v = assembler.transform(test)

models = {
    'LinearRegression': LinearRegression(featuresCol='features', labelCol='demand'),
    'RandomForest': RandomForestRegressor(featuresCol='features', labelCol='demand', numTrees=50),
    'GBT': GBTRegressor(featuresCol='features', labelCol='demand', maxIter=20)
}

evaluator = RegressionEvaluator(labelCol='demand', predictionCol='prediction')

for name, model in models.items():
    fitted = model.fit(train_v)
    preds = fitted.transform(test_v)
    rmse = evaluator.evaluate(preds, {evaluator.metricName:'rmse'})
    mae = evaluator.evaluate(preds, {evaluator.metricName:'mae'})
    r2 = evaluator.evaluate(preds, {evaluator.metricName:'r2'})
    print(f"\n{name}: RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.4f}")

# Best model save
best_model = RandomForestRegressor(featuresCol='features', labelCol='demand', numTrees=50).fit(train_v)
output_path = os.path.join(BASE_DIR, 'models', 'spark', 'demand_forecast')
os.makedirs(output_path, exist_ok=True)
best_model.write().overwrite().save(output_path)
print(f"\n✅ Model saved: {output_path}")
print("🎉 DEMAND FORECAST COMPLETE!")
spark.stop()