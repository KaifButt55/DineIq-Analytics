import pandas as pd
import random
import os
from datetime import datetime, timedelta

os.makedirs('raw_data', exist_ok=True)
random.seed(42)

# Load existing data
menu = pd.read_csv('raw_data/menu_items.csv')
customers = pd.read_csv('raw_data/customers.csv')
locations = pd.read_csv('raw_data/restaurants.csv')

locations_list = locations['location_id'].tolist()
channels = ['Dine-in', 'Takeaway', 'App', 'Delivery']

orders = []
order_items = []
item_line_id = 1

start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 12, 31)

print("⏳ Generating 100,000 orders and 1,000,000+ order items...")
print("⏱️  This will take 2-5 minutes. Please wait...")

for i in range(1, 300001):
    order_date = start_date + timedelta(
        days=random.randint(0, 364),
        hours=random.randint(10, 23),
        minutes=random.randint(0, 59)
    )
    
    customer = customers.sample(1).iloc[0]
    location = random.choice(locations_list)
    channel = random.choice(channels)
    
    is_weekend = order_date.weekday() >= 5
    n_items = random.randint(4, 8) if is_weekend else random.randint(2, 6
    )
    
    total = 0
    order_id = f'ORD{i:07d}'
    
    for _ in range(n_items):
        item = menu.sample(1).iloc[0]
        qty = random.randint(1, 3)
        price = item['base_price'] * random.uniform(0.9, 1.1)
        total += price * qty
        
        order_items.append({
            'order_line_id': f'OL{item_line_id:08d}',
            'order_id': order_id,
            'item_id': item['item_id'],
            'quantity': qty,
            'unit_price': round(price, 2),
            'line_total': round(price * qty, 2)
        })
        item_line_id += 1
    
    orders.append({
        'order_id': order_id,
        'customer_id': customer['customer_id'],
        'location_id': location,
        'order_date': order_date,
        'channel': channel,
        'total_amount': round(total, 2),
        'status': random.choices(['Completed', 'Cancelled'], weights=[95, 5])[0]
    })

orders_df = pd.DataFrame(orders)
orders_df.to_csv('raw_data/orders.csv', index=False)
print(f"✅ Orders: {len(orders_df)} records created")

items_df = pd.DataFrame(order_items)
items_df.to_csv('raw_data/order_items.csv', index=False)
print(f"✅ Order Items: {len(items_df)} records created")