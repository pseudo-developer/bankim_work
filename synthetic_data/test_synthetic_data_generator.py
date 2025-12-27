import os
import json
import re
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import AzureOpenAI


# =========================
# ENV + CLIENT SETUP
# =========================
load_dotenv()

API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

if not all([API_KEY, ENDPOINT, DEPLOYMENT, API_VERSION]):
    raise ValueError("Missing one or more Azure OpenAI environment variables")

client = AzureOpenAI(
    api_key=API_KEY,
    azure_endpoint=ENDPOINT,
    api_version=API_VERSION
)


# =========================
# PATH HANDLING (SAFE)
# =========================
BASE_DIR = Path(__file__).resolve().parent
INPUT_CSV = BASE_DIR / "input_file.csv"
OUTPUT_CSV = BASE_DIR / "output_file.csv"


# =========================
# CSV ANALYSIS
# =========================
def analyze_csv(csv_path, sample_size=5):
    df = pd.read_csv(csv_path)

    analysis = {
        "row_count": len(df),
        "columns": []
    }

    for col in df.columns:
        analysis["columns"].append({
            "name": col,
            "dtype": str(df[col].dtype),
            "sample_values": (
                df[col]
                .dropna()
                .astype(str)
                .head(sample_size)
                .tolist()
            )
        })

    return analysis


# =========================
# JSON EXTRACTION (ROBUST)
# =========================
def extract_json(text: str):
    """
    Safely extract JSON array from LLM output.
    """
    text = text.strip()

    # Remove markdown fences if present
    text = re.sub(r"^```json", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()

    return json.loads(text)


# =========================
# LLM SYNTHETIC DATA GENERATION
# =========================
def generate_synthetic_data(
    client,
    deployment_name,
    dataset_profile,
    target_rows,
    target_region
):
    prompt = f"""
You are a synthetic data generator.

Input dataset profile:
{json.dumps(dataset_profile, indent=2)}

User requirements:
- Generate exactly {target_rows} rows
- Target region: {target_region}
- Output must be a JSON ARRAY of objects
- Each object must strictly follow the input column names
- Do NOT reuse original values
- Data must look realistic for the target region
- Phone numbers, names, addresses, emails must match regional formats
- NO explanations
- NO markdown
- Return ONLY valid JSON
"""

    response = client.chat.completions.create(
        model=deployment_name,
        messages=[
            {"role": "system", "content": "You return clean JSON only."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.8,
        max_tokens=2000
    )

    raw_output = response.choices[0].message.content

    print("\n🔍 RAW MODEL OUTPUT:\n")
    print(raw_output)
    print("\n------------------------\n")

    return extract_json(raw_output)


# =========================
# WRITE CSV
# =========================
def write_csv(data, output_path):
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)


# =========================
# MAIN EXECUTION
# =========================
if __name__ == "__main__":
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    print("📄 Reading input CSV...")
    analysis = analyze_csv(INPUT_CSV)

    print("🤖 Generating synthetic data...")
    synthetic_data = generate_synthetic_data(
        client=client,
        deployment_name=DEPLOYMENT,
        dataset_profile=analysis,
        target_rows=20,
        target_region="China"
    )

    print("💾 Writing output CSV...")
    write_csv(synthetic_data, OUTPUT_CSV)

    print(f"✅ Synthetic CSV generated successfully: {OUTPUT_CSV}")
