import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# ==========================================================
# PATH SETUP
# ==========================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(BASE_DIR, 'raw_data')
PARQUET_PATH = os.path.join(BASE_DIR, 'parquet_data')

print(f"📁 Base Dir: {BASE_DIR}")
print(f"📁 Raw Data Path: {RAW_PATH}")
print(f"📁 Parquet Path: {PARQUET_PATH}")

if not os.path.exists(RAW_PATH):
    print(f"❌ ERROR: raw_data folder nahi mila: {RAW_PATH}")
    exit()

os.makedirs(PARQUET_PATH, exist_ok=True)

# ==========================================================
# SPARK SESSION
# ==========================================================
print("\n⏳ Spark Session start ho raha hai...")
spark = SparkSession.builder \
    .appName("DineIQ_Ingestion") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.driver.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "8") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")
print("✅ Spark Session ready!")

# ==========================================================
# DATA LOAD
# ==========================================================
def load_csv(name, filename):
    path = os.path.join(RAW_PATH, filename)
    if not os.path.exists(path):
        print(f"❌ {name}: File nahi mili: {path}")
        return None
    df = spark.read.csv(path, header=True, inferSchema=True)
    count = df.count()
    print(f"✅ {name}: {count} records loaded")
    return df

print("\n⏳ Files load ho rahi hain...")
print("=" * 60)

orders = load_csv("Orders", "orders.csv")
order_items = load_csv("Order Items", "order_items.csv")
menu = load_csv("Menu Items", "menu_items.csv")
customers = load_csv("Customers", "customers.csv")
ratings = load_csv("Ratings", "ratings.csv")
wastage = load_csv("Wastage", "wastage.csv")
pricing = load_csv("Pricing History", "pricing_history.csv")
promotions = load_csv("Promotions", "promotions.csv")
inventory = load_csv("Inventory", "inventory.csv")

# ==========================================================
# DATA QUALITY REPORT
# ==========================================================
print("\n📊 DATA QUALITY REPORT")
print("=" * 60)

def quality_check(df, name):
    if df is None:
        return
    total = df.count()
    print(f"\n{name} (Total: {total}):")
    null_found = False
    for c in df.columns:
        nulls = df.filter(col(c).isNull()).count()
        pct = (nulls / total * 100) if total > 0 else 0
        if nulls > 0:
            print(f"  ⚠️  {c}: {nulls} nulls ({pct:.2f}%)")
            null_found = True
    if not null_found:
        print(f"  ✅ No nulls found")

quality_check(orders, "Orders")
quality_check(order_items, "Order Items")
quality_check(menu, "Menu Items")
quality_check(customers, "Customers")
quality_check(ratings, "Ratings")
quality_check(wastage, "Wastage")

# Duplicate check
print("\n🔍 DUPLICATE CHECK:")
print("=" * 60)

if orders is not None:
    dup = orders.groupBy('order_id').count().filter('count > 1').count()
    print(f"  Duplicate Order IDs: {dup}")

if customers is not None:
    dup = customers.groupBy('customer_id').count().filter('count > 1').count()
    print(f"  Duplicate Customer IDs: {dup}")

if menu is not None:
    dup = menu.groupBy('item_id').count().filter('count > 1').count()
    print(f"  Duplicate Menu Item IDs: {dup}")

# ==========================================================
# SAVE AS PARQUET
# ==========================================================
print("\n⏳ Parquet mein save ho raha hai...")
print("=" * 60)

def save_parquet(df, name, folder):
    if df is None:
        return
    path = os.path.join(PARQUET_PATH, folder)
    df.write.mode('overwrite').parquet(path)
    print(f"✅ {name} → parquet_data/{folder}/")

save_parquet(orders, "Orders", "orders")
save_parquet(order_items, "Order Items", "order_items")
save_parquet(menu, "Menu Items", "menu_items")
save_parquet(customers, "Customers", "customers")
save_parquet(ratings, "Ratings", "ratings")
save_parquet(wastage, "Wastage", "wastage")
save_parquet(pricing, "Pricing History", "pricing_history")
save_parquet(promotions, "Promotions", "promotions")
save_parquet(inventory, "Inventory", "inventory")

# ==========================================================
# SUMMARY
# ==========================================================
print("\n" + "=" * 60)
print("🎉 INGESTION COMPLETE!")
print("=" * 60)
print(f"✅ Data loaded from: {RAW_PATH}")
print(f"✅ Data saved as Parquet in: {PARQUET_PATH}")
print(f"✅ Spark version: {spark.version}")

spark.stop()
print("\n✅ Spark session stopped.")