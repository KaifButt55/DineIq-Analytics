import pandas as pd
import random
import os

# Folder create karein agar nahi hai
os.makedirs('raw_data', exist_ok=True)

random.seed(42)

# Categories
categories = ['Starters', 'Main Course', 'Desserts', 'Beverages', 
              'Fast Food', 'Chinese', 'Italian', 'BBQ', 'Seafood', 'Breakfast']

# Menu Items (150)
items = []
for i in range(1, 151):
    category = random.choice(categories)
    cost = round(random.uniform(50, 500), 2)
    price = round(cost * random.uniform(1.5, 3.0), 2)
    items.append({
        'item_id': f'ITM{i:04d}',
        'item_name': f'{category}_Item_{i}',
        'category': category,
        'cost': cost,
        'base_price': price,
        'is_available': random.choice([True, True, True, False])
    })

menu_df = pd.DataFrame(items)
menu_df.to_csv('raw_data/menu_items.csv', index=False)
print(f"✅ Menu items: {len(menu_df)} records created")

# Menu Categories (10)
cat_df = pd.DataFrame({'category_id': range(1, 11), 'category_name': categories})
cat_df.to_csv('raw_data/menu_categories.csv', index=False)
print(f"✅ Categories: {len(cat_df)} records created")

# Restaurants (20 Locations)
locations = []
for i in range(1, 21):
    locations.append({
        'location_id': f'LOC{i:03d}',
        'location_name': f'DineIQ Branch {i}',
        'city': random.choice(['Karachi', 'Lahore', 'Islamabad', 'Rawalpindi', 'Faisalabad']),
        'is_active': True
    })
loc_df = pd.DataFrame(locations)
loc_df.to_csv('raw_data/restaurants.csv', index=False)
print(f"✅ Restaurants: {len(loc_df)} records created")