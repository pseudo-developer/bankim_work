# app.py
import os
import json
import pandas as pd
import argparse
import sys
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import logging
from openai import AzureOpenAI
from dotenv import load_dotenv
import re
import csv
from pathlib import Path

# Load environment variables from current or parent directory
def find_and_load_env():
    """Find .env file in current or parent directories"""
    current_dir = Path(__file__).parent
    
    # Check current directory and up to 2 parent directories
    for i in range(3):
        check_dir = current_dir
        for _ in range(i):
            check_dir = check_dir.parent
        
        env_path = check_dir / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            print(f"✓ Loaded environment from: {env_path}")
            return True
    
    print("⚠ .env file not found. Using system environment variables.")
    return False

# Find and load .env
find_and_load_env()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class UserRequest:
    """Data class to store user request parameters"""
    region: str
    num_rows: int
    description: str
    input_file: Optional[str] = None
    file_content: Optional[str] = None
    data_type: Optional[str] = None  # Store detected data type


class Config:
    """Configuration manager"""
    def __init__(self):
        self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
        self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
        
        # Token limits
        self.max_tokens_per_request = 4000
        self.max_rows_per_chunk = 30  # Reduced for reliability
        
        self.validate_config()
    
    def validate_config(self):
        """Validate required configuration"""
        missing_vars = []
        if not self.azure_endpoint:
            missing_vars.append("AZURE_OPENAI_ENDPOINT")
        if not self.api_key:
            missing_vars.append("AZURE_OPENAI_API_KEY")
        
        if missing_vars:
            raise ValueError(f"Missing environment variables: {', '.join(missing_vars)}")
        
        logger.info(f"Configuration loaded: Deployment={self.deployment_name}")


class AzureAIClient:
    """Handles Azure OpenAI API interactions"""
    def __init__(self, config: Config):
        self.config = config
        self.client = AzureOpenAI(
            azure_endpoint=config.azure_endpoint,
            api_key=config.api_key,
            api_version=config.api_version
        )
    
    def generate_completion(self, prompt: str, max_tokens: int = 2000) -> str:
        """Generate completion with error handling and retry logic"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.debug(f"Generating completion (attempt {attempt + 1})")
                response = self.client.chat.completions.create(
                    model=self.config.deployment_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful data generation assistant that follows instructions precisely."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=0.7,
                    response_format={"type": "json_object"}
                )
                return response.choices[0].message.content
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)
        return ""


class DataGenerator:
    """Main data generation logic with chunking"""
    
    def __init__(self, config: Config, ai_client: AzureAIClient):
        self.config = config
        self.ai_client = ai_client
    
    def parse_user_input(self, input_text: str, input_file: Optional[str] = None) -> UserRequest:
        """Use AI to intelligently parse user input"""
        
        print(f"\n🔍 Analyzing your request: '{input_text}'")
        
        # Step 1: Use AI to understand the request
        analysis = self._analyze_request_with_ai(input_text)
        
        # Step 2: Show what AI understood
        self._display_ai_analysis(analysis)
        
        # Step 3: Get region (from AI or user)
        region = self._get_region_from_user(analysis)
        
        # Step 4: Get row count (from AI or user)
        num_rows = self._get_row_count_from_user(analysis)
        
        # Step 5: Handle file if provided
        file_content = self._process_input_file(input_file)
        
        return UserRequest(
            region=region,
            num_rows=num_rows,
            description=input_text,
            input_file=input_file,
            file_content=file_content,
            data_type=analysis.get("data_type")
        )
    
    def _analyze_request_with_ai(self, input_text: str) -> Dict[str, Any]:
        """Use AI to analyze the user request"""
        prompt = f"""
        Analyze this data generation request and extract key information.
        
        USER REQUEST: "{input_text}"
        
        Return JSON with these fields:
        1. "region": The geographical region mentioned (e.g., "California", "New York", "India"). 
           Return null if no region is specified.
        2. "num_rows": Number of rows/records requested as integer. Default to 50 if not specified.
        3. "data_type": Type of data being requested (e.g., "customer_records", "sales_data", "demographic_data", "employee_records", "product_data")
        4. "has_region": Boolean indicating if region was mentioned
        5. "has_row_count": Boolean indicating if row count was mentioned
        
        Be intelligent about understanding:
        - "for California" means region: California
        - "in Texas" means region: Texas
        - "50 records" means num_rows: 50
        - "customer data" means data_type: customer_records
        - "sales figures" means data_type: sales_data
        - "employee information" means data_type: employee_records
        
        IMPORTANT: Return ONLY valid JSON.
        """
        
        try:
            response = self.ai_client.generate_completion(prompt, max_tokens=500)
            result = json.loads(response)
            
            # Validate and clean the result
            if result.get("region") in ["null", None, ""]:
                result["region"] = None
                result["has_region"] = False
            
            # Ensure num_rows is valid
            try:
                result["num_rows"] = int(result.get("num_rows", 50))
            except (ValueError, TypeError):
                result["num_rows"] = 50
            
            return result
            
        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            return self._fallback_analysis(input_text)
    
    def _fallback_analysis(self, input_text: str) -> Dict[str, Any]:
        """Fallback analysis when AI fails"""
        input_lower = input_text.lower()
        
        # Simple region detection
        region = None
        region_patterns = [
            r'for\s+([a-z\s]+?)(?:\s|$|,)',
            r'in\s+([a-z\s]+?)(?:\s|$|,)',
            r'from\s+([a-z\s]+?)(?:\s|$|,)',
            r'based\sin\s+([a-z\s]+?)(?:\s|$|,)'
        ]
        
        for pattern in region_patterns:
            match = re.search(pattern, input_lower)
            if match:
                region = match.group(1).strip()
                # Clean up common words
                region = re.sub(r'\s+(?:data|records|information|details|figures|analytics)$', '', region)
                if len(region) > 1:
                    break
                else:
                    region = None
        
        # Simple row count detection
        num_rows = 50
        row_match = re.search(r'(\d+)\s*(?:rows?|records?|entries?|data\s+points?)', input_lower)
        if row_match:
            num_rows = int(row_match.group(1))
        
        # Simple data type detection
        data_type = "generic_data"
        type_keywords = {
            "customer": "customer_records",
            "sales": "sales_data",
            "employee": "employee_records",
            "demographic": "demographic_data",
            "product": "product_data",
            "financial": "financial_data",
            "medical": "medical_records",
            "real estate": "real_estate_data",
            "ecommerce": "ecommerce_data"
        }
        
        for keyword, dtype in type_keywords.items():
            if keyword in input_lower:
                data_type = dtype
                break
        
        return {
            "region": region,
            "num_rows": num_rows,
            "data_type": data_type,
            "has_region": region is not None,
            "has_row_count": row_match is not None
        }
    
    def _display_ai_analysis(self, analysis: Dict[str, Any]):
        """Display what AI understood"""
        print("\n" + "="*60)
        print("🤖 AI ANALYSIS RESULTS:")
        print("="*60)
        
        data_type = analysis.get("data_type", "generic_data").replace("_", " ").title()
        print(f"📊 Data Type: {data_type}")
        
        region = analysis.get("region")
        if region and str(region).lower() != "null" and str(region).strip():
            print(f"📍 Region Detected: {region}")
        else:
            print(f"📍 Region: Not specified in prompt")
        
        rows = analysis.get("num_rows", 50)
        print(f"📈 Rows Requested: {rows}")
        print("="*60)
    
    def _get_region_from_user(self, analysis: Dict[str, Any]) -> str:
        """Get region from analysis or prompt user"""
        region = analysis.get("region")
        
        # Check if region was detected and is valid
        if (region and 
            str(region).lower() != "null" and 
            str(region).strip() and
            len(str(region).strip()) > 1):
            
            confirm = input(f"\n📍 Use '{region}' as the region? (y/n): ").strip().lower()
            if confirm in ['y', 'yes', '']:
                return str(region).strip()
        
        # Ask user for region
        while True:
            region_input = input("\n📍 Please specify the region (e.g., California, India, Europe): ").strip()
            if region_input:
                return region_input
            print("⚠ Region is required. Please enter a valid region.")
    
    def _get_row_count_from_user(self, analysis: Dict[str, Any]) -> int:
        """Get row count from analysis or prompt user"""
        try:
            # Get rows from analysis
            num_rows = int(analysis.get("num_rows", 50))
            num_rows = min(max(1, num_rows), 200)
            
            # Ask for confirmation
            print(f"\n📊 AI detected request for {num_rows} rows.")
            user_input = input(f"Generate {num_rows} rows? (y/n, or enter different number): ").strip().lower()
            
            if user_input in ['y', 'yes', '']:
                return num_rows
            elif user_input in ['n', 'no']:
                return self._prompt_for_row_count()
            else:
                # Check if user entered a number directly
                try:
                    new_rows = int(user_input)
                    if 1 <= new_rows <= 200:
                        return new_rows
                except ValueError:
                    pass
                return self._prompt_for_row_count()
                
        except (ValueError, TypeError) as e:
            logger.error(f"Error parsing row count: {e}")
            return self._prompt_for_row_count()
    
    def _prompt_for_row_count(self) -> int:
        """Prompt user for row count"""
        while True:
            try:
                user_input = input("\n📊 How many rows of data? (1-200, default 50): ").strip()
                if not user_input:
                    return 50
                
                num_rows = int(user_input)
                if 1 <= num_rows <= 200:
                    return num_rows
                
                print("⚠ Please enter a number between 1 and 200.")
            except ValueError:
                print("⚠ Please enter a valid number.")
    
    def _process_input_file(self, input_file: Optional[str]) -> Optional[str]:
        """Process input file if provided"""
        if not input_file:
            return None
        
        if not os.path.exists(input_file):
            print(f"⚠ Input file not found: {input_file}")
            return None
        
        try:
            if input_file.endswith('.csv'):
                df = pd.read_csv(input_file)
                return f"CSV file with columns: {list(df.columns)}. First few rows:\n{df.head().to_string()}"
            else:
                with open(input_file, 'r', encoding='utf-8') as f:
                    return f.read(1000)
        except Exception as e:
            logger.error(f"Error reading input file: {e}")
            print(f"⚠ Could not read input file: {e}")
            return None
    
    def analyze_request(self, request: UserRequest) -> Dict[str, Any]:
        """Analyze user request to determine data schema"""
        
        prompt = f"""
        Create a schema for data generation in {request.region}.
        
        User Request: "{request.description}"
        Region: {request.region}
        Number of Rows: {request.num_rows}
        
        IMPORTANT CONSTRAINTS:
        1. Use ONLY these data types: string, integer, float, date, boolean
        2. DO NOT use "array" or "object" types (they don't work in CSV)
        3. For preferences/tags, use comma-separated strings instead of arrays
        4. Keep column names simple and CSV-friendly
        5. Maximum 10 columns for reliability
        
        Provide JSON with:
        1. data_type: Type of data (customer_records, sales_data, etc.)
        2. columns: List of column objects with name, type, description
        3. description: Brief description
        4. examples: Example values for each column (2-3 examples)
        
        Keep it simple and realistic for {request.region}.
        """
        
        try:
            print("\n📋 Determining data schema...")
            response = self.ai_client.generate_completion(prompt, max_tokens=1500)
            schema = json.loads(response)
            
            # Validate and fix schema if needed
            schema = self._validate_and_fix_schema(schema)
            
            return schema
            
        except Exception as e:
            logger.error(f"Error analyzing request: {e}")
            print(f"⚠ Using fallback schema due to error: {e}")
            return self._get_fallback_schema(request)
    
    def _validate_and_fix_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and fix schema to ensure CSV compatibility"""
        if "columns" not in schema:
            return self._get_fallback_schema(UserRequest(region="Unknown", num_rows=50, description=""))
        
        # Fix array types to strings
        for col in schema["columns"]:
            if col.get("type") in ["array", "object"]:
                col["type"] = "string"
                print(f"⚠ Changed column '{col['name']}' type from array/object to string")
        
        # Limit to 10 columns for reliability
        if len(schema["columns"]) > 10:
            print(f"⚠ Limiting columns from {len(schema['columns'])} to 10 for reliability")
            schema["columns"] = schema["columns"][:10]
        
        # Fix examples if they contain arrays
        if "examples" in schema:
            for col_name, examples in schema["examples"].items():
                if isinstance(examples, list):
                    # Convert any arrays in examples to strings
                    for i, example in enumerate(examples):
                        if isinstance(example, list):
                            schema["examples"][col_name][i] = ",".join(map(str, example))
        
        return schema
    
    def _get_fallback_schema(self, request: UserRequest) -> Dict[str, Any]:
        """Provide fallback schema if analysis fails"""
        return {
            "data_type": "customer_records",
            "columns": [
                {"name": "id", "type": "integer", "description": "Unique identifier"},
                {"name": "first_name", "type": "string", "description": "First name"},
                {"name": "last_name", "type": "string", "description": "Last name"},
                {"name": "email", "type": "string", "description": "Email address"},
                {"name": "city", "type": "string", "description": "City"},
                {"name": "signup_date", "type": "date", "description": "Date of signup"}
            ],
            "description": f"Customer records for {request.region}",
            "examples": {
                "id": [1, 2, 3],
                "first_name": ["John", "Jane", "Bob"],
                "last_name": ["Smith", "Doe", "Johnson"],
                "email": ["john@email.com", "jane@email.com", "bob@email.com"],
                "city": ["Los Angeles", "San Francisco", "San Diego"],
                "signup_date": ["2023-01-15", "2023-02-20", "2023-03-10"]
            }
        }
    
    def generate_data_chunk(self, schema: Dict[str, Any], region: str, chunk_size: int, 
                           start_id: int = 1, attempt: int = 1) -> List[Dict]:
        """Generate a chunk of data rows - SIMPLIFIED AND ROBUST"""
        column_names = [col["name"] for col in schema["columns"]]
        
        # SIMPLE, CLEAR PROMPT
        prompt = f"""
        Create {chunk_size} rows of data for {region}.
        
        COLUMNS: {', '.join(column_names)}
        
        FORMAT REQUIREMENT: Return a JSON object with one key "data" that contains an array.
        Each item in the array should be an object with the column names as keys.
        
        EXAMPLE OF CORRECT FORMAT:
        {{
          "data": [
            {{"id": 1, "first_name": "John", "last_name": "Smith", "email": "john@example.com", "city": "Los Angeles", "signup_date": "2023-01-15"}},
            {{"id": 2, "first_name": "Jane", "last_name": "Doe", "email": "jane@example.com", "city": "San Francisco", "signup_date": "2023-02-20"}}
          ]
        }}
        
        Start IDs from {start_id}.
        Make data realistic for {region}.
        Dates in YYYY-MM-DD format.
        
        Return ONLY the JSON object with the "data" key.
        """
        
        try:
            print(f"   📤 Requesting {chunk_size} rows from AI...")
            response = self.ai_client.generate_completion(prompt, max_tokens=3000)
            
            # Clean response
            response = response.strip()
            
            # Remove markdown code blocks if present
            if response.startswith('```json'):
                response = response[7:]
                if response.endswith('```'):
                    response = response[:-3]
            elif response.startswith('```'):
                response = response[3:]
                if response.endswith('```'):
                    response = response[:-3]
            
            response = response.strip()
            
            # Try to parse JSON
            try:
                parsed = json.loads(response)
            except json.JSONDecodeError:
                # Try to find JSON object in the text
                match = re.search(r'\{.*\}', response, re.DOTALL)
                if match:
                    response = match.group(0)
                    parsed = json.loads(response)
                else:
                    raise
            
            # Extract data from the response
            if isinstance(parsed, dict) and "data" in parsed:
                data = parsed["data"]
                if isinstance(data, list):
                    print(f"   ✅ Extracted {len(data)} rows from 'data' key")
                    return data
            
            # Alternative: if response is already an array
            elif isinstance(parsed, list):
                print(f"   ✅ Found {len(parsed)} rows as direct array")
                return parsed
            
            # Alternative: look for any array in the object
            elif isinstance(parsed, dict):
                for key, value in parsed.items():
                    if isinstance(value, list):
                        print(f"   ✅ Extracted {len(value)} rows from '{key}' key")
                        return value
            
            print(f"   ❌ Could not find data array in response")
            raise ValueError("No data array found in response")
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error (attempt {attempt}): {e}")
            print(f"   ❌ JSON Parse Error: {e}")
            
            if attempt <= 2:
                print(f"   🔄 Retrying (attempt {attempt})...")
                # Reduce chunk size and retry
                reduced_size = max(3, chunk_size // 2)
                return self.generate_data_chunk(schema, region, reduced_size, start_id, attempt + 1)
            else:
                print("   ❌ Max retries exceeded")
                return []
        
        except Exception as e:
            logger.error(f"Error generating chunk: {e}")
            print(f"   ❌ Error: {str(e)}")
            
            if attempt <= 2:
                reduced_size = max(3, chunk_size // 2)
                print(f"   🔄 Retrying with reduced size: {reduced_size}")
                return self.generate_data_chunk(schema, region, reduced_size, start_id, attempt + 1)
            return []
    
    def generate_all_data(self, request: UserRequest, schema: Dict[str, Any]) -> pd.DataFrame:
        """Generate all requested data with chunking"""
        all_data = []
        remaining_rows = request.num_rows
        current_id = 1
        
        print(f"\n🚀 Starting data generation...")
        print(f"   Target: {request.num_rows} rows for {request.region}")
        print(f"   Maximum chunk size: {self.config.max_rows_per_chunk} rows per request")
        
        chunk_number = 1
        
        while remaining_rows > 0:
            chunk_size = min(self.config.max_rows_per_chunk, remaining_rows)
            
            print(f"\n   📦 Chunk {chunk_number}: Generating {chunk_size} rows (remaining: {remaining_rows})")
            
            chunk_data = self.generate_data_chunk(
                schema=schema,
                region=request.region,
                chunk_size=chunk_size,
                start_id=current_id
            )
            
            if not chunk_data:
                print(f"   ❌ Failed to generate chunk {chunk_number}")
                if len(all_data) > 0:
                    print(f"   ⚠ Stopping early. Generated {len(all_data)} rows total.")
                    break
                else:
                    print(f"   ❌ No data generated at all.")
                    return pd.DataFrame()
            
            print(f"   ✅ Generated {len(chunk_data)} rows")
            
            all_data.extend(chunk_data)
            current_id += len(chunk_data)
            remaining_rows -= len(chunk_data)
            chunk_number += 1
            
            if remaining_rows > 0:
                time.sleep(2)  # Slightly longer delay for reliability
        
        if all_data:
            df = pd.DataFrame(all_data)
            print(f"\n✅ Total generated: {len(df)} rows")
            
            if len(df) > request.num_rows:
                df = df.head(request.num_rows)
                print(f"   Trimmed to requested {request.num_rows} rows")
            
            return df
        else:
            print("\n❌ No data was generated")
            return pd.DataFrame()
    
    def save_to_csv(self, df: pd.DataFrame, request: UserRequest) -> str:
        """Save DataFrame to CSV file"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        safe_region = re.sub(r'[^\w\-_]', '_', request.region)
        filename = f"generated_data_{safe_region}_{timestamp}.csv"
        
        try:
            # Convert any list-like values to strings
            for column in df.columns:
                if df[column].apply(lambda x: isinstance(x, list)).any():
                    df[column] = df[column].apply(lambda x: ','.join(map(str, x)) if isinstance(x, list) else x)
            
            df.to_csv(filename, index=False, quoting=csv.QUOTE_ALL, encoding='utf-8')
            logger.info(f"Data saved to: {filename}")
            
            print(f"\n{'='*60}")
            print("✅ GENERATION COMPLETE!")
            print('='*60)
            print(f"📁 File: {filename}")
            print(f"📊 Rows generated: {len(df)}")
            print(f"📍 Region: {request.region}")
            print(f"📋 Columns: {list(df.columns)}")
            print('='*60)
            
            print("\n📄 First 5 rows:")
            print(df.head().to_string(index=False))
            
            return filename
            
        except Exception as e:
            logger.error(f"Error saving CSV: {e}")
            print(f"❌ Error saving CSV: {e}")
            
            # Try alternative save method
            try:
                with open(filename, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=df.columns)
                    writer.writeheader()
                    for _, row in df.iterrows():
                        row_dict = row.to_dict()
                        for key, value in row_dict.items():
                            if isinstance(value, list):
                                row_dict[key] = ','.join(map(str, value))
                        writer.writerow(row_dict)
                print(f"✅ Saved using alternative method: {filename}")
                return filename
            except Exception as e2:
                print(f"❌ Alternative save also failed: {e2}")
                raise


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Generate region-specific data using AI")
    parser.add_argument("--input", "-i", help="Input text prompt")
    parser.add_argument("--file", "-f", help="Input file path (CSV or text)")
    parser.add_argument("--region", "-r", help="Target region (overrides detected region)")
    parser.add_argument("--rows", "-n", type=int, help="Number of rows (1-200, overrides detected count)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    try:
        if args.verbose:
            logger.setLevel(logging.DEBUG)
        
        # Initialize
        config = Config()
        ai_client = AzureAIClient(config)
        generator = DataGenerator(config, ai_client)
        
        # Get input
        input_text = args.input
        if not input_text:
            input_text = input("\n📝 Enter your data request: ").strip()
            if not input_text:
                input_text = "Generate sample data"
        
        # Parse request
        request = generator.parse_user_input(
            input_text=input_text,
            input_file=args.file
        )
        
        # Override with command line args if provided
        if args.region:
            request.region = args.region
            print(f"\n📍 Overriding region to: {args.region}")
        
        if args.rows:
            request.num_rows = min(max(1, args.rows), 200)
            print(f"\n📊 Overriding row count to: {request.num_rows}")
        
        print(f"\n✅ Final Parameters:")
        print(f"   Region: {request.region}")
        print(f"   Rows: {request.num_rows}")
        if request.input_file:
            print(f"   Input file: {request.input_file}")
        
        # Analyze request to get schema
        schema = generator.analyze_request(request)
        
        print(f"\n📋 Schema: {schema['data_type']}")
        print("Columns to generate:")
        for i, col in enumerate(schema["columns"], 1):
            print(f"   {i:2d}. {col['name']} ({col['type']})")
        
        # Confirm generation
        confirm = input("\n🚀 Proceed with generation? (y/n): ").strip().lower()
        if confirm not in ['y', 'yes', '']:
            print("Generation cancelled.")
            return 0
        
        # Generate data
        df = generator.generate_all_data(request, schema)
        
        if df.empty:
            print("\n❌ No data was generated.")
            print("\nPossible solutions:")
            print("1. Try with fewer rows (e.g., --rows 5)")
            print("2. Try a simpler request")
            print("3. Check Azure OpenAI service status")
            return 1
        
        # Save to CSV
        filename = generator.save_to_csv(df, request)
        print(f"\n💾 Data saved to: {os.path.abspath(filename)}")
        
        # Ask to open file
        open_file = input("\n🔓 Open the generated file? (y/n): ").strip().lower()
        if open_file in ['y', 'yes']:
            try:
                if sys.platform == "win32":
                    os.startfile(filename)
                elif sys.platform == "darwin":
                    os.system(f"open {filename}")
                else:
                    os.system(f"xdg-open {filename}")
                print("✅ File opened.")
            except:
                print("❌ Could not open file automatically.")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⏹ Operation cancelled by user.")
        return 130
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        print(f"\n❌ ERROR: {str(e)}")
        print("\n🔧 Troubleshooting:")
        print("1. Check .env file has correct Azure credentials")
        print("2. Verify API key and endpoint are correct")
        print("3. Check deployment name matches Azure portal")
        print("4. Try a simpler request with fewer rows (e.g., --rows 5)")
        print("5. Check Azure OpenAI quota and limits")
        return 1


if __name__ == "__main__":
    # Check if required environment variables are set
    required_vars = ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY"]
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        print("\nPlease ensure your .env file contains:")
        print("AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/")
        print("AZURE_OPENAI_API_KEY=your_api_key_here")
        print("AZURE_OPENAI_API_VERSION=2024-02-01")
        print("AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini")
        sys.exit(1)
    
    sys.exit(main())