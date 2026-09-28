import pandas as pd
import random
from faker import Faker
import os

os.makedirs('raw_data', exist_ok=True)
fake = Faker()
random.seed(42)

customers = []
for i in range(1, 50001):  # 50,000 customers
    customers.append({
        'customer_id': f'CUST{i:06d}',
        'name': fake.name(),
        'email': fake.email(),
        'phone': fake.phone_number(),
        'city': fake.city(),
        'signup_date': fake.date_between(start_date='-2y', end_date='today'),
        'preferred_channel': random.choice(['Dine-in', 'Takeaway', 'App', 'Delivery'])
    })

cust_df = pd.DataFrame(customers)
cust_df.to_csv('raw_data/customers.csv', index=False)
print(f"✅ Customers: {len(cust_df)} records created") 
