import csv
import random
import time
from faker import Faker

# --- 1. The "Mock" Gen AI Layer ---
def ask_ai_for_culture(region, category):
    knowledge_base = {
        "India": {
            "grocery_stores": ["D-Mart", "Reliance Fresh", "Big Bazaar", "JioMart"],
            "street_food": ["Vada Pav", "Pani Puri", "Samosa", "Chole Bhature"]
        },
        "USA": {
            "grocery_stores": ["Walmart", "Target", "Costco", "Whole Foods"],
            "street_food": ["Hot Dog", "Pretzel", "Tacos", "Burger"]
        },
        "China": {
            "grocery_stores": ["99 Ranch", "Hema Fresh", "Yonghui", "CR Vanguard"],
            "street_food": ["Baozi", "Jianbing", "Dumplings", "Stinky Tofu"]
        }
    }
    region_data = knowledge_base.get(region, {})
    options = region_data.get(category, [f"Generic {category}"])
    return random.choice(options)

# --- 2. The Generator (The Factory) ---
def generate_synthetic_data(region_name, region_code, num_rows):
    fake = Faker(region_code)
    
    for i in range(num_rows):
        # A. Fetch Data
        name = fake.name()
        phone = fake.phone_number()
        store = ask_ai_for_culture(region_name, "grocery_stores")
        food = ask_ai_for_culture(region_name, "street_food")
        
        # B. Yield Row
        yield {
            "ID": i + 1,
            "Name": name,
            "Phone": phone,
            "Favorite Store": store,
            "Last Ordered Food": food,
            "Region": region_name
        }

# --- 3. The CSV Writer (The Exporter) ---
def write_to_csv(filename, generator_func, total_rows):
    print(f"🚀 Starting generation: {total_rows} rows to '{filename}'...")
    
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = None
        
        # We iterate through the generator
        for i, row in enumerate(generator_func):
            
            # Create headers dynamically based on the first row's keys
            if i == 0:
                writer = csv.DictWriter(file, fieldnames=row.keys())
                writer.writeheader()
            
            # Write the row immediately to disk
            writer.writerow(row)
            
            # Simple progress tracker (overwrites the same line)
            print(f"   ... Processed {i + 1}/{total_rows} rows", end='\r')

    print(f"\n✅ Done! File saved as: {filename}")

# --- 4. Execution ---
if _name_ == "_main_":
    
    # Configuration
    REGION = "India"      # Change to 'USA' or 'China'
    CODE = "en_IN"        # Change to 'en_US' or 'zh_CN'
    ROWS = 100            # Try changing this to 100000
    OUTPUT_FILE = "user_data.csv"

    # Create the generator object (does not run yet)
    data_gen = generate_synthetic_data(REGION, CODE, ROWS)

    # Pass the generator to the writer (runs and writes simultaneously)
    write_to_csv(OUTPUT_FILE, data_gen, ROWS)