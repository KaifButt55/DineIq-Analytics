import pandas as pd
import random
import os
from datetime import datetime, timedelta

os.makedirs('raw_data', exist_ok=True)
random.seed(42)

menu = pd.read_csv('raw_data/menu_items.csv')
customers = pd.read_csv('raw_data/customers.csv')
locations = pd.read_csv('raw_data/restaurants.csv')
orders = pd.read_csv('raw_data/orders.csv')

menu_ids = menu['item_id'].tolist()
cust_ids = customers['customer_id'].tolist()
loc_ids = locations['location_id'].tolist()
order_ids = orders['order_id'].tolist()

# 1. Pricing History
print("⏳ Generating Pricing History...")
pricing = []
for item in menu_ids:
    for _ in range(random.randint(1, 3)):
        pricing.append({
            'price_id': f'PRC{len(pricing)+1:06d}',
            'item_id': item,
            'old_price': round(random.uniform(50, 500), 2),
            'new_price': round(random.uniform(50, 500), 2),
            'change_date': datetime(2024, 1, 1) + timedelta(days=random.randint(0, 365))
        })
pd.DataFrame(pricing).to_csv('raw_data/pricing_history.csv', index=False)
print(f"✅ Pricing History: {len(pricing)} records")

# 2. Promotions
print("⏳ Generating Promotions...")
promos = []
for i in range(1, 101):
    promos.append({
        'promo_id': f'PROMO{i:03d}',
        'item_id': random.choice(menu_ids),
        'discount_pct': random.choice([10, 15, 20, 25, 30]),
        'start_date': datetime(2024, 1, 1) + timedelta(days=random.randint(0, 300)),
        'end_date': datetime(2024, 1, 1) + timedelta(days=random.randint(300, 365))
    })
pd.DataFrame(promos).to_csv('raw_data/promotions.csv', index=False)
print(f"✅ Promotions: {len(promos)} records")

# 3. Ratings
print("⏳ Generating Ratings (100,000)...")
ratings = []
for i in range(1, 100001):
    ratings.append({
        'rating_id': f'RAT{i:07d}',
        'order_id': random.choice(order_ids),
        'customer_id': random.choice(cust_ids),
        'item_id': random.choice(menu_ids),
        'rating': random.choices([1, 2, 3, 4, 5], weights=[5, 10, 15, 30, 40])[0],
        'rating_date': datetime(2024, 1, 1) + timedelta(days=random.randint(0, 365))
    })
pd.DataFrame(ratings).to_csv('raw_data/ratings.csv', index=False)
print(f"✅ Ratings: {len(ratings)} records")

# 4. Inventory
print("⏳ Generating Inventory...")
inventory = []
for loc in loc_ids:
    for item in menu_ids:
        inventory.append({
            'inventory_id': f'INV{len(inventory)+1:07d}',
            'location_id': loc,
            'item_id': item,
            'stock_qty': random.randint(0, 500),
            'reorder_level': random.randint(20, 100)
        })
pd.DataFrame(inventory).to_csv('raw_data/inventory.csv', index=False)
print(f"✅ Inventory: {len(inventory)} records")

# 5. Wastage
print("⏳ Generating Wastage (50,000)...")
wastage = []
reasons = ['Expired', 'Overproduction', 'Damaged', 'Customer Return', 'Prep Error']
for i in range(1, 50001):
    wastage.append({
        'wastage_id': f'WST{i:06d}',
        'item_id': random.choice(menu_ids),
        'location_id': random.choice(loc_ids),
        'quantity': random.randint(1, 20),
        'cost': round(random.uniform(50, 1000), 2),
        'wastage_date': datetime(2024, 1, 1) + timedelta(days=random.randint(0, 365)),
        'reason': random.choice(reasons)
    })
pd.DataFrame(wastage).to_csv('raw_data/wastage.csv', index=False)
print(f"✅ Wastage: {len(wastage)} records")