import os
import certifi # Ensure 'pip install certifi' is run
import csv
import random
import json
import google.generativeai as genai
from faker import Faker
from dotenv import load_dotenv

# ==========================================
# 1. NETWORK & API SETUP
# ==========================================

# 🟢 ACTION: Load the variables from the .env file into Python's memory
load_dotenv()

# 🟢 FIX 1: Tell Python to trust the valid certificates (Solves SSL Error)
os.environ['GRPC_DEFAULT_SSL_ROOTS_FILE_PATH'] = certifi.where()

# 🟢 FIX 2: Your API Key
API_KEY = os.getenv("GEMINI_API_KEY")

# 🟢 FIX 3: Use REST transport (Solves "hanging" on corporate networks)
genai.configure(api_key=API_KEY, transport='rest')

# 🟢 FIX 4: Use a model that actually exists in your list
# We are using Gemini 2.0 Flash - it's fast and available to you.
model = genai.GenerativeModel('gemini-2.0-flash')

# ==========================================
# 2. INTELLIGENT CACHING (The "Brain")
# ==========================================
AI_CACHE = {} 

def ask_ai_for_culture(region, category):
    """
    Fetches cultural context. Uses cache to avoid hitting API rate limits.
    """
    cache_key = f"{region}_{category}"

    # A. Check Cache First
    if cache_key in AI_CACHE:
        return random.choice(AI_CACHE[cache_key])

    # B. Ask Gemini (Only if we don't know yet)
    print(f"\n🤖 (Gemini 2.0) Learning about '{category}' in '{region}'...", end=" ", flush=True)
    
    prompt = (
        f"List 15 popular {category} in {region}. "
        "Return ONLY a raw JSON array of strings. "
        "Do not use markdown formatting. "
        "Example: [\"Item 1\", \"Item 2\"]"
    )

    try:
        response = model.generate_content(prompt)
        text = response.text.replace('json', '').replace('', '').strip()
        data_list = json.loads(text)
        
        AI_CACHE[cache_key] = data_list
        print("✅ Done!")
        return random.choice(data_list)

    except Exception as e:
        print(f"❌ Error: {e}. Using generic fallback.")
        return f"Generic {category}"

# ==========================================
# 3. DATA GENERATOR (The "Factory")
# ==========================================
def generate_synthetic_data(region_name, region_code, num_rows):
    fake = Faker(region_code)
    
    for i in range(num_rows):
        yield {
            "ID": i + 1,
            "Name": fake.name(),
            "Phone": fake.phone_number(),
            "Region": region_name,
            # AI Columns
            "Top Grocery Brand": ask_ai_for_culture(region_name, "grocery store chains"),
            "Local Street Food": ask_ai_for_culture(region_name, "street food items"),
            "Famous Landmark": ask_ai_for_culture(region_name, "historical landmarks")
        }

# ==========================================
# 4. CSV WRITER (The "Exporter")
# ==========================================
def write_to_csv(filename, generator, total_rows):
    print(f"🚀 Generating {total_rows} rows to '{filename}'...")
    
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = None
        for i, row in enumerate(generator):
            if i == 0:
                writer = csv.DictWriter(file, fieldnames=row.keys())
                writer.writeheader()
            
            writer.writerow(row)
            # flush=True ensures you see the progress bar update in real-time
            print(f"   ... Row {i + 1}/{total_rows} saved", end='\r', flush=True)

    print(f"\n✨ Success! Data saved to {filename}")

# ==========================================
# 5. EXECUTION
# ==========================================
if _name_ == "_main_":
    
    # ⚙️ USER CONFIG
    REGION = "Brazil"    # Try "Japan", "France", "India"
    CODE = "pt_BR"       # Matches region (e.g., ja_JP, fr_FR, en_IN)
    COUNT = 10          # How many rows?
    
    if not API_KEY:
        print("❌ STOP: You forgot to paste your API Key in the script!")
    else:
        gen = generate_synthetic_data(REGION, CODE, COUNT)
        write_to_csv(f"final_{REGION}_{COUNT}_data.csv", gen, COUNT)
