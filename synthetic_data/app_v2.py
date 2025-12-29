# app.py
import os
import json
import pandas as pd
import argparse
import sys
import time
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import logging
from logging.handlers import RotatingFileHandler
from openai import AzureOpenAI
from dotenv import load_dotenv
import re
import csv
from pathlib import Path

# ================== LOGGING CONFIGURATION ==================
# Configure logging with rotation for developers/operators
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Create logs directory if it doesn't exist
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

# Setup rotating file handler (10MB max, keep 5 backup files)
file_handler = RotatingFileHandler(
    logs_dir / 'app_debug.log',
    maxBytes=10*1024*1024,  # 10 MB
    backupCount=5,
    encoding='utf-8'
)
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.INFO)

# Console handler for user-facing messages (simpler format)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(message)s'))
console_handler.setLevel(logging.WARNING)  # Only show warnings/errors to console

# Configure root logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# ================== USER-FACING PRINT FUNCTIONS ==================
def print_user(message: str):
    """Print messages for end users (no timestamps, clean format)"""
    print(message)

def print_error(message: str):
    """Print error messages for end users"""
    print(f"\n❌ {message}")

def print_success(message: str):
    """Print success messages for end users"""
    print(f"\n✅ {message}")

def print_info(message: str):
    """Print informational messages for end users"""
    print(f"\nℹ️  {message}")

def print_warning(message: str):
    """Print warning messages for end users"""
    print(f"\n⚠️  {message}")

# ================== ENVIRONMENT SETUP ==================
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
            print_success(f"Loaded environment from: {env_path}")
            return True
    
    print_warning(".env file not found. Using system environment variables.")
    return False

# Find and load .env
find_and_load_env()


@dataclass
class UserRequest:
    """Data class to store user request parameters"""
    region: str
    num_rows: int
    description: str
    input_file: Optional[str] = None
    file_content: Optional[str] = None
    data_type: Optional[str] = None  # Store detected data type
    file_schema: Optional[Dict[str, Any]] = None  # Schema extracted from file
    user_specified_columns: Optional[List[str]] = None  # Columns specified by user


class Config:
    """Configuration manager with configurable limits"""
    def __init__(self):
        self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
        self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
        
        # Configurable limits from environment with warnings
        self.max_rows_total = self._get_max_rows_limit()
        self.max_rows_per_chunk = min(50, self.max_rows_total // 4)  # Dynamic chunk size
        
        # Token limits
        self.max_tokens_per_request = 4000
        self.max_file_content_chars = 2000  # Limit file content for AI analysis
        self.file_sample_rows = 5  # Only sample 5 rows for analysis (reduced from 100)
        
        self.validate_config()
    
    def _get_max_rows_limit(self) -> int:
        """Get max rows limit from environment with warnings"""
        max_rows_str = os.getenv("MAX_ROWS_LIMIT", "200")
        
        try:
            max_rows = int(max_rows_str)
            
            # Warning for high limits
            if max_rows > 200:
                print_warning(f"MAX_ROWS_LIMIT set to {max_rows} (high value)")
                print_info("⚠️  High row counts may:")
                print_info("   • Increase AI token usage and costs")
                print_info("   • Require more API calls and time")
                print_info("   • Potentially hit rate limits")
                print_info("   • Consider using 200 or less for optimal performance")
            
            elif max_rows < 10:
                print_warning(f"MAX_ROWS_LIMIT set to {max_rows} (very low)")
                max_rows = 10  # Set minimum reasonable limit
            
            return max(max_rows, 1)  # Ensure at least 1
            
        except ValueError:
            print_warning(f"Invalid MAX_ROWS_LIMIT value: {max_rows_str}. Using default 200.")
            return 200
    
    def validate_config(self):
        """Validate required configuration"""
        missing_vars = []
        if not self.azure_endpoint:
            missing_vars.append("AZURE_OPENAI_ENDPOINT")
        if not self.api_key:
            missing_vars.append("AZURE_OPENAI_API_KEY")
        
        if missing_vars:
            logger.error(f"Missing environment variables: {missing_vars}")
            print_error(f"Missing environment variables: {', '.join(missing_vars)}")
            raise ValueError(f"Missing environment variables: {', '.join(missing_vars)}")
        
        logger.info(f"Configuration loaded: Deployment={self.deployment_name}, MaxRows={self.max_rows_total}")
        print_success(f"Connected to Azure OpenAI: {self.deployment_name}")
        print_info(f"Maximum rows allowed: {self.max_rows_total}")


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
    
    def parse_user_input(self, input_text: str, input_file: Optional[str] = None, 
                         cli_region: Optional[str] = None, cli_rows: Optional[int] = None) -> UserRequest:
        """Parse user input with priority: CLI args > File analysis > AI analysis > User prompt"""
        
        print_user(f"\n🔍 Processing request: '{input_text}'")
        
        # ================== PROPER FILE VALIDATION ==================
        file_content = None
        file_schema = None
        file_was_provided = input_file is not None
        
        if input_file:
            print_info(f"Checking input file: {input_file}")
            
            # Check if file exists
            if not os.path.exists(input_file):
                print_error(f"Input file not found: {input_file}")
                print_user("\n📋 Options:")
                print_user("1. Place the file in correct location and re-run")
                print_user("2. Continue without file (use text prompt only)")
                
                choice = input("\nContinue without file? (y/n): ").strip().lower()
                if choice in ['y', 'yes']:
                    print_info("Continuing with text prompt only...")
                    input_file = None  # Clear file reference
                else:
                    print_user("\n📝 Please:")
                    print_user("   • Place the file and re-run the command")
                    print_user("   • Or run without --file parameter for text-only mode")
                    sys.exit(1)
            else:
                # File exists, analyze it
                print_info("File found, analyzing...")
                file_content, file_schema = self._analyze_input_file(input_file)
                
                if not file_content:
                    print_warning("Could not analyze file. Continuing with text prompt...")
                    input_file = None  # Clear file reference
        
        # ================== AI ANALYSIS FOR TEXT PROMPTS ==================
        analysis = None
        user_specified_columns = None
        
        if not input_file and not file_was_provided:
            # Only analyze text if no file was provided or file was removed
            print_info("Analyzing text prompt...")
            analysis = self._analyze_request_with_ai(input_text)
            
            # Extract user-specified columns from analysis
            if analysis and "mentioned_columns" in analysis:
                user_specified_columns = analysis.get("mentioned_columns")
                if user_specified_columns:
                    print_info(f"Detected user-specified columns: {user_specified_columns}")
        
        # ================== GET REGION WITH PRIORITY ==================
        region = self._get_region_with_priority(
            cli_region=cli_region,
            file_context=file_content,
            ai_analysis=analysis,
            user_input=input_text,
            has_file=input_file is not None
        )
        
        # ================== GET ROW COUNT WITH PRIORITY ==================
        num_rows = self._get_rows_with_priority(
            cli_rows=cli_rows,
            ai_analysis=analysis,
            user_input=input_text,
            has_file=input_file is not None
        )
        
        return UserRequest(
            region=region,
            num_rows=num_rows,
            description=input_text,
            input_file=input_file,
            file_content=file_content,
            data_type=analysis.get("data_type") if analysis else None,
            file_schema=file_schema,
            user_specified_columns=user_specified_columns
        )
    
    def _analyze_input_file(self, file_path: str) -> Tuple[Optional[str], Optional[Dict]]:
        """Analyze input file and extract schema/content - ONLY 5 ROWS FOR ANALYSIS"""
        try:
            file_ext = os.path.splitext(file_path)[1].lower()
            print_info(f"Analyzing {file_ext.upper()} file (sampling 5 rows)...")
            
            if file_ext == '.csv':
                return self._analyze_csv_file(file_path)
            elif file_ext == '.json':
                return self._analyze_json_file(file_path)
            elif file_ext in ['.txt', '.text', '.md']:
                return self._analyze_text_file(file_path)
            else:
                print_warning(f"Unsupported file type: {file_ext}")
                print_user(f"Supported formats: .csv, .json, .txt, .text, .md")
                return None, None
                
        except Exception as e:
            logger.error(f"Error analyzing file {file_path}: {e}")
            print_error(f"Error analyzing file: {str(e)}")
            return None, None
    
    def _analyze_csv_file(self, file_path: str) -> Tuple[str, Dict]:
        """Analyze CSV file - ONLY 5 ROWS for analysis"""
        try:
            print_info("Reading CSV file (5 rows for analysis)...")
            
            # Read ONLY 5 rows for analysis (reduced from 100)
            df = pd.read_csv(file_path, nrows=self.config.file_sample_rows)
            
            if len(df) == 0:
                print_error("CSV file is empty or could not be read")
                return None, None
            
            print_success(f"Successfully read {len(df)} sample rows, {len(df.columns)} columns")
            
            # Create schema from actual file columns
            schema = {
                "source": "csv_file",
                "columns": [],
                "data_types": {},
                "sample_values": {},
                "total_estimated_rows": self._estimate_total_rows(file_path),
                "file_columns": list(df.columns),
                "analysis_rows_used": len(df)
            }
            
            for column in df.columns:
                col_type = str(df[column].dtype)
                # Map pandas dtypes to our types
                if 'int' in col_type:
                    dtype = 'integer'
                elif 'float' in col_type:
                    dtype = 'float'
                elif 'bool' in col_type:
                    dtype = 'boolean'
                elif 'datetime' in col_type:
                    dtype = 'date'
                else:
                    dtype = 'string'
                
                schema["columns"].append({
                    "name": column,
                    "type": dtype,
                    "description": f"Column from CSV file"
                })
                schema["data_types"][column] = dtype
                
                # Get sample non-null values from the 5 rows
                non_null = df[column].dropna().head(3).tolist()
                schema["sample_values"][column] = non_null
            
            # Prepare file content for AI (limited size)
            file_content = f"CSV File: {os.path.basename(file_path)}\n"
            file_content += f"Total Estimated Rows: {schema['total_estimated_rows']}\n"
            file_content += f"Sample Rows Analyzed: {len(df)}\n"
            file_content += f"Columns: {list(df.columns)}\n"
            file_content += f"Sample Data ({len(df)} rows):\n"
            file_content += df.to_string(index=False)
            
            print_info(f"Extracted {len(df.columns)} columns from {len(df)} sample rows")
            
            return file_content[:self.config.max_file_content_chars], schema
            
        except Exception as e:
            logger.error(f"Error reading CSV file: {e}")
            print_error(f"Failed to read CSV: {str(e)}")
            return None, None
    
    def _estimate_total_rows(self, file_path: str) -> int:
        """Efficiently estimate total rows in file without reading all"""
        try:
            # Count lines efficiently (faster than reading entire file)
            with open(file_path, 'r', encoding='utf-8') as f:
                line_count = sum(1 for line in f)
            return max(1, line_count - 1)  # Subtract header row
        except:
            return 0  # Return 0 if can't estimate
    
    def _analyze_json_file(self, file_path: str) -> Tuple[str, Dict]:
        """Analyze JSON file - limited sampling"""
        try:
            print_info("Reading JSON file...")
            
            # Read entire file but limit size for analysis
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read(min(10000, self.config.max_file_content_chars))
            
            data = json.loads(content)
            print_success("Successfully parsed JSON")
            
            # For large arrays, sample first 5 items
            if isinstance(data, list):
                sampled_data = data[:self.config.file_sample_rows]
                if len(data) > self.config.file_sample_rows:
                    print_info(f"Sampled {len(sampled_data)} items from {len(data)} total")
                
                if sampled_data and len(sampled_data) > 0:
                    first_item = sampled_data[0]
                    if isinstance(first_item, dict):
                        columns = list(first_item.keys())
                        
                        schema = {
                            "source": "json_file",
                            "columns": [],
                            "structure": "array_of_objects",
                            "total_items": len(data),
                            "sample_count": len(sampled_data)
                        }
                        
                        for col in columns:
                            schema["columns"].append({
                                "name": col,
                                "type": self._infer_type(first_item.get(col)),
                                "description": f"Column from JSON file"
                            })
                        
                        file_content = f"JSON File: {os.path.basename(file_path)}\n"
                        file_content += f"Array with {len(data)} objects\n"
                        file_content += f"Sample Items Analyzed: {len(sampled_data)}\n"
                        file_content += f"Columns: {columns}\n"
                        
                        print_info(f"Found {len(columns)} columns in JSON array")
                        return file_content, schema
            
            elif isinstance(data, dict):
                # Single object or nested structure
                file_content = f"JSON File: {os.path.basename(file_path)}\n"
                file_content += f"Object with keys: {list(data.keys())}\n"
                
                return file_content, {"source": "json_file", "structure": "object", "keys": list(data.keys())}
            
            print_warning("Unsupported JSON structure")
            return f"JSON file: {file_path}", {"source": "json_file", "structure": "unknown"}
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON file: {e}")
            print_error(f"Invalid JSON format: {str(e)}")
            return None, None
        except Exception as e:
            logger.error(f"Error reading JSON file: {e}")
            print_error(f"Error reading JSON: {str(e)}")
            return None, None
    
    def _analyze_text_file(self, file_path: str) -> Tuple[str, Dict]:
        """Analyze text file - limited sampling"""
        try:
            print_info("Reading text file...")
            
            # Read limited content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read(min(5000, self.config.max_file_content_chars))
            
            lines = content.split('\n')
            sampled_lines = lines[:10]  # Only first 10 lines
            
            file_content = f"Text File: {os.path.basename(file_path)}\n"
            file_content += f"Total Lines: {len(lines)}\n"
            file_content += f"Sample Lines Analyzed: {len(sampled_lines)}\n"
            file_content += f"First {min(3, len(sampled_lines))} lines:\n"
            file_content += '\n'.join(sampled_lines[:3])
            
            print_success(f"Read {len(sampled_lines)} sample lines from text file")
            
            return file_content, {"source": "text_file", "line_count": len(lines), "sampled_lines": len(sampled_lines)}
            
        except Exception as e:
            logger.error(f"Error reading text file: {e}")
            print_error(f"Error reading text file: {str(e)}")
            return None, None
    
    def _infer_type(self, value) -> str:
        """Infer data type from value"""
        if value is None:
            return "string"
        elif isinstance(value, int):
            return "integer"
        elif isinstance(value, float):
            return "float"
        elif isinstance(value, bool):
            return "boolean"
        elif isinstance(value, str):
            # Check if it looks like a date
            date_patterns = [
                r'\d{4}-\d{2}-\d{2}',
                r'\d{2}/\d{2}/\d{4}',
                r'\d{2}-\d{2}-\d{4}'
            ]
            for pattern in date_patterns:
                if re.match(pattern, value):
                    return "date"
            return "string"
        else:
            return "string"
    
    def _get_region_with_priority(self, cli_region: Optional[str], file_context: Optional[str], 
                                 ai_analysis: Optional[Dict], user_input: str, has_file: bool) -> str:
        """Get region with priority: CLI > File > AI > User input"""
        
        # Priority 1: CLI argument
        if cli_region:
            print_success(f"Using command line region: {cli_region}")
            return cli_region
        
        # Priority 2: Extract from file context using AI
        if file_context and has_file:
            region = self._extract_region_from_context(file_context)
            if region:
                confirm = input(f"\n📍 Extracted region '{region}' from file. Use it? (y/n): ").strip().lower()
                if confirm in ['y', 'yes', '']:
                    return region
        
        # Priority 3: AI analysis of user input
        if ai_analysis and ai_analysis.get("region"):
            region = ai_analysis["region"]
            if str(region).lower() != "null" and str(region).strip():
                confirm = input(f"\n📍 AI detected region '{region}'. Use it? (y/n): ").strip().lower()
                if confirm in ['y', 'yes', '']:
                    return str(region).strip()
        
        # Priority 4: Ask user
        while True:
            region_input = input("\n📍 Please specify the region (e.g., California, India, Europe): ").strip()
            if region_input:
                return region_input
            print_warning("Region is required. Please enter a valid region.")
    
    def _get_rows_with_priority(self, cli_rows: Optional[int], ai_analysis: Optional[Dict], 
                               user_input: str, has_file: bool) -> int:
        """Get row count with priority: CLI > AI > User input"""
        
        # Priority 1: CLI argument
        if cli_rows is not None:
            num_rows = min(max(1, cli_rows), self.config.max_rows_total)
            print_success(f"Using command line row count: {num_rows}")
            
            # Warning if close to or at max limit
            if num_rows >= self.config.max_rows_total * 0.8:  # 80% of limit
                print_info(f"Note: Generating {num_rows} rows (max limit: {self.config.max_rows_total})")
            
            return num_rows
        
        # Priority 2: AI analysis
        if ai_analysis and not has_file:  # Only use AI rows for text prompts
            try:
                ai_rows = int(ai_analysis.get("num_rows", 50))
                ai_rows = min(max(1, ai_rows), self.config.max_rows_total)
                
                user_input = input(f"\n📊 AI detected request for {ai_rows} rows. Use this? (y/n, or enter different number): ").strip().lower()
                
                if user_input in ['y', 'yes', '']:
                    return ai_rows
                elif user_input in ['n', 'no']:
                    return self._prompt_for_row_count()
                else:
                    # Check if user entered a number
                    try:
                        new_rows = int(user_input)
                        new_rows = min(max(1, new_rows), self.config.max_rows_total)
                        if new_rows >= self.config.max_rows_total * 0.8:  # 80% of limit
                            print_info(f"Note: Generating {new_rows} rows (max limit: {self.config.max_rows_total})")
                        return new_rows
                    except ValueError:
                        pass
                    return self._prompt_for_row_count()
                    
            except (ValueError, TypeError):
                pass
        
        # Priority 3: Prompt user
        return self._prompt_for_row_count()
    
    def _prompt_for_row_count(self) -> int:
        """Prompt user for row count with configurable max"""
        while True:
            try:
                prompt = f"\n📊 How many rows of data? (1-{self.config.max_rows_total}, default 50): "
                user_input = input(prompt).strip()
                
                if not user_input:
                    return 50
                
                num_rows = int(user_input)
                if 1 <= num_rows <= self.config.max_rows_total:
                    # Warning if close to max limit
                    if num_rows >= self.config.max_rows_total * 0.8:  # 80% of limit
                        print_info(f"Note: Generating {num_rows} rows (max limit: {self.config.max_rows_total})")
                    return num_rows
                
                print_warning(f"Please enter a number between 1 and {self.config.max_rows_total}.")
            except ValueError:
                print_warning("Please enter a valid number.")
    
    def _extract_region_from_context(self, context: str) -> Optional[str]:
        """Try to extract region from file context"""
        try:
            prompt = f"""
            Extract any geographical region mentioned in this content:
            
            CONTENT:
            {context[:1000]}
            
            Return JSON with: {{"region": "extracted region or null"}}
            """
            
            response = self.ai_client.generate_completion(prompt, max_tokens=500)
            result = json.loads(response)
            region = result.get("region")
            
            if region and str(region).lower() != "null":
                return str(region).strip()
                
        except Exception:
            pass
        
        return None
    
    def _analyze_request_with_ai(self, input_text: str) -> Dict[str, Any]:
        """Use AI to analyze the user request - ENHANCED for column detection"""
        prompt = f"""
        Analyze this data generation request and extract key information.
        
        USER REQUEST: "{input_text}"
        
        Extract the following and return as JSON:
        1. "region": Geographical region mentioned (e.g., "California", "New York", "India"). 
           Return null if no region is specified.
        2. "num_rows": Number of rows/records requested as integer. Default to 50 if not specified.
        3. "data_type": Type of data being requested (e.g., "customer_records", "sales_data", 
           "demographic_data", "employee_records", "product_data")
        4. "mentioned_columns": List of any specific columns mentioned by the user.
           Look for phrases like:
           - "with columns:" 
           - "including columns:" 
           - "having fields:" 
           - "with fields:" 
           - "containing:" 
           - "columns like:" 
           - "with data for:" 
           - Example: "customer data with columns: id, name, email, phone" → ["id", "name", "email", "phone"]
        5. "has_region": Boolean indicating if region was mentioned
        6. "has_row_count": Boolean indicating if row count was mentioned
        
        Be intelligent about understanding:
        - "for California" means region: California
        - "in Texas" means region: Texas
        - "50 records" means num_rows: 50
        - "customer data" means data_type: customer_records
        - "sales figures" means data_type: sales_data
        - "employee information" means data_type: employee_records
        
        For columns, extract any comma-separated list after column-related keywords.
        
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
            
            # Clean and validate mentioned columns
            mentioned_columns = result.get("mentioned_columns", [])
            if mentioned_columns:
                # Ensure it's a list and clean column names
                if isinstance(mentioned_columns, str):
                    # Try to parse string as list
                    mentioned_columns = [col.strip() for col in mentioned_columns.split(',')]
                elif isinstance(mentioned_columns, list):
                    mentioned_columns = [str(col).strip() for col in mentioned_columns if str(col).strip()]
                else:
                    mentioned_columns = []
                
                # Remove empty strings and clean
                mentioned_columns = [col for col in mentioned_columns if col]
                result["mentioned_columns"] = mentioned_columns
            
            return result
            
        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            return self._fallback_analysis(input_text)
    
    def _fallback_analysis(self, input_text: str) -> Dict[str, Any]:
        """Fallback analysis when AI fails - ENHANCED for column detection"""
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
        
        # Simple column detection with regex
        mentioned_columns = []
        column_patterns = [
            r'with columns?:?\s*([^\.\n]+)',
            r'including columns?:?\s*([^\.\n]+)', 
            r'having fields?:?\s*([^\.\n]+)',
            r'with fields?:?\s*([^\.\n]+)',
            r'containing:?\s*([^\.\n]+)',
            r'columns like:?\s*([^\.\n]+)',
            r'with data for:?\s*([^\.\n]+)'
        ]
        
        for pattern in column_patterns:
            match = re.search(pattern, input_text, re.IGNORECASE)
            if match:
                columns_text = match.group(1)
                # Split by commas and clean
                columns = [col.strip() for col in re.split(r'[,\s]+', columns_text) if col.strip()]
                mentioned_columns.extend(columns)
                break
        
        return {
            "region": region,
            "num_rows": num_rows,
            "data_type": data_type,
            "mentioned_columns": mentioned_columns if mentioned_columns else None,
            "has_region": region is not None,
            "has_row_count": row_match is not None
        }
    
    def analyze_request(self, request: UserRequest) -> Dict[str, Any]:
        """Analyze user request to determine data schema - ENHANCED for user columns"""
        
        # ================== CHECK FOR USER-SPECIFIED COLUMNS ==================
        if request.user_specified_columns:
            print_info(f"Using user-specified columns: {request.user_specified_columns}")
            
            # Create schema from user-specified columns
            columns = []
            for col_name in request.user_specified_columns:
                # Clean column name and infer type
                clean_name = col_name.strip().lower().replace(' ', '_')
                columns.append({
                    "name": clean_name,
                    "type": self._infer_column_type(clean_name),
                    "description": f"User-specified column: {col_name}"
                })
            
            schema = {
                "data_type": request.data_type or "custom_data",
                "columns": columns,
                "description": f"Custom data for {request.region} with user-specified columns",
                "examples": {}
            }
            
            print_success(f"Created schema from {len(columns)} user-specified columns")
            return self._validate_and_fix_schema(schema)
        
        # ================== FILE-BASED REQUEST ==================
        if request.input_file and request.file_schema:
            print_info(f"Creating schema from file: {os.path.basename(request.input_file)}")
            
            if "file_columns" in request.file_schema:
                # Create schema from actual file columns
                columns = []
                file_columns = request.file_schema["file_columns"]
                
                for col_name in file_columns:
                    col_type = request.file_schema.get("data_types", {}).get(col_name, "string")
                    columns.append({
                        "name": col_name,
                        "type": col_type,
                        "description": f"From {os.path.basename(request.input_file)}"
                    })
                
                schema = {
                    "data_type": f"file_based_data",
                    "columns": columns,
                    "description": f"Data similar to {os.path.basename(request.input_file)} for {request.region}",
                    "examples": request.file_schema.get("sample_values", {})
                }
                
                print_success(f"Created schema from {len(columns)} file columns")
                return self._validate_and_fix_schema(schema)
            else:
                # File exists but no columns - use file-specific fallback
                print_warning("Could not extract columns from file, using file-based fallback")
                return self._get_file_based_fallback_schema(request)
        
        # ================== TEXT-ONLY REQUEST ==================
        else:
            print_info("Creating schema from text prompt...")
            
            # Enhanced prompt that mentions user can specify columns
            prompt = f"""
            Create a schema for data generation in {request.region}.
            
            User Request: "{request.description}"
            Region: {request.region}
            Number of Rows: {request.num_rows}
            
            IMPORTANT: If the user mentioned specific columns in their request, 
            make sure to include those columns in the schema.
            
            CONSTRAINTS:
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
                response = self.ai_client.generate_completion(prompt, max_tokens=1500)
                schema = json.loads(response)
                
                # Validate and fix schema if needed
                schema = self._validate_and_fix_schema(schema)
                
                return schema
                
            except Exception as e:
                logger.error(f"Error analyzing request: {e}")
                print_warning(f"AI connection failed: {str(e)}")
                print_info("Using fallback schema...")
                return self._get_text_based_fallback_schema(request)
    
    def _infer_column_type(self, column_name: str) -> str:
        """Infer column type from column name"""
        column_lower = column_name.lower()
        
        if any(word in column_lower for word in ['id', 'num', 'code', 'index', 'no', 'number']):
            return 'integer'
        elif any(word in column_lower for word in ['date', 'time', 'year', 'month', 'day']):
            return 'date'
        elif any(word in column_lower for word in ['price', 'amount', 'cost', 'salary', 'value', 'total', 'sum']):
            return 'float'
        elif any(word in column_lower for word in ['active', 'status', 'flag', 'is_', 'has_']):
            return 'boolean'
        else:
            return 'string'
    
    def _get_file_based_fallback_schema(self, request: UserRequest) -> Dict[str, Any]:
        """Fallback for file-based requests when AI fails"""
        file_name = os.path.basename(request.input_file) if request.input_file else "data"
        print_info(f"Creating schema based on file: {file_name}")
        
        # Try to infer type from filename
        file_lower = file_name.lower()
        
        if any(word in file_lower for word in ["customer", "client", "user"]):
            data_type = "customer_records"
            columns = [
                {"name": "customer_id", "type": "integer", "description": "Customer ID"},
                {"name": "name", "type": "string", "description": "Customer name"},
                {"name": "email", "type": "string", "description": "Email address"},
                {"name": "phone", "type": "string", "description": "Phone number"},
                {"name": "address", "type": "string", "description": "Address"},
                {"name": "city", "type": "string", "description": "City"},
                {"name": "signup_date", "type": "date", "description": "Signup date"},
            ]
        elif any(word in file_lower for word in ["sales", "transaction", "order"]):
            data_type = "sales_data"
            columns = [
                {"name": "transaction_id", "type": "integer", "description": "Transaction ID"},
                {"name": "product", "type": "string", "description": "Product name"},
                {"name": "quantity", "type": "integer", "description": "Quantity"},
                {"name": "amount", "type": "float", "description": "Transaction amount"},
                {"name": "customer", "type": "string", "description": "Customer name"},
                {"name": "date", "type": "date", "description": "Transaction date"},
            ]
        elif any(word in file_lower for word in ["employee", "staff", "worker"]):
            data_type = "employee_records"
            columns = [
                {"name": "employee_id", "type": "integer", "description": "Employee ID"},
                {"name": "name", "type": "string", "description": "Employee name"},
                {"name": "department", "type": "string", "description": "Department"},
                {"name": "position", "type": "string", "description": "Position"},
                {"name": "salary", "type": "float", "description": "Salary"},
                {"name": "hire_date", "type": "date", "description": "Hire date"},
            ]
        else:
            data_type = "generic_data"
            columns = [
                {"name": "id", "type": "integer", "description": "Unique ID"},
                {"name": "name", "type": "string", "description": "Name"},
                {"name": "value", "type": "float", "description": "Value"},
                {"name": "category", "type": "string", "description": "Category"},
                {"name": "date", "type": "date", "description": "Date"},
            ]
        
        return {
            "data_type": data_type,
            "columns": columns,
            "description": f"{data_type.replace('_', ' ').title()} for {request.region} (from {file_name})",
            "examples": {}
        }
    
    def _get_text_based_fallback_schema(self, request: UserRequest) -> Dict[str, Any]:
        """Fallback for text-only requests when AI fails"""
        print_info("Using generic customer schema as fallback")
        
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
    
    def _validate_and_fix_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and fix schema to ensure CSV compatibility"""
        if "columns" not in schema:
            return self._get_text_based_fallback_schema(UserRequest(region="Unknown", num_rows=50, description=""))
        
        # Fix array types to strings
        for col in schema["columns"]:
            if col.get("type") in ["array", "object"]:
                col["type"] = "string"
                print_warning(f"Changed column '{col['name']}' type from array/object to string")
        
        # Limit to 10 columns for reliability
        if len(schema["columns"]) > 10:
            print_warning(f"Limiting columns from {len(schema['columns'])} to 10 for reliability")
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
    
    def generate_data_chunk(self, schema: Dict[str, Any], region: str, chunk_size: int, 
                           start_id: int = 1, attempt: int = 1) -> List[Dict]:
        """Generate a chunk of data rows - SIMPLIFIED AND ROBUST"""
        column_names = [col["name"] for col in schema["columns"]]
        
        # Include user-specified columns in prompt if available
        columns_description = ', '.join(column_names)
        
        # SIMPLE, CLEAR PROMPT
        prompt = f"""
        Create {chunk_size} rows of data for {region}.
        
        COLUMNS: {columns_description}
        
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
            print_info(f"Requesting {chunk_size} rows from AI...")
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
                    print_success(f"Extracted {len(data)} rows from 'data' key")
                    return data
            
            # Alternative: if response is already an array
            elif isinstance(parsed, list):
                print_success(f"Found {len(parsed)} rows as direct array")
                return parsed
            
            # Alternative: look for any array in the object
            elif isinstance(parsed, dict):
                for key, value in parsed.items():
                    if isinstance(value, list):
                        print_success(f"Extracted {len(value)} rows from '{key}' key")
                        return value
            
            print_error("Could not find data array in response")
            raise ValueError("No data array found in response")
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error (attempt {attempt}): {e}")
            print_error(f"JSON Parse Error: {e}")
            
            if attempt <= 2:
                print_info(f"Retrying (attempt {attempt})...")
                # Reduce chunk size and retry
                reduced_size = max(3, chunk_size // 2)
                return self.generate_data_chunk(schema, region, reduced_size, start_id, attempt + 1)
            else:
                print_error("Max retries exceeded")
                return []
        
        except Exception as e:
            logger.error(f"Error generating chunk: {e}")
            print_error(f"Error: {str(e)}")
            
            if attempt <= 2:
                reduced_size = max(3, chunk_size // 2)
                print_info(f"Retrying with reduced size: {reduced_size}")
                return self.generate_data_chunk(schema, region, reduced_size, start_id, attempt + 1)
            return []
    
    def generate_all_data(self, request: UserRequest, schema: Dict[str, Any]) -> pd.DataFrame:
        """Generate all requested data with chunking"""
        all_data = []
        remaining_rows = request.num_rows
        current_id = 1
        
        print_user(f"\n🚀 Starting data generation...")
        print_info(f"Target: {request.num_rows} rows for {request.region}")
        print_info(f"Maximum chunk size: {self.config.max_rows_per_chunk} rows per request")
        
        # Warning for large requests
        if request.num_rows > 100:
            print_info(f"Note: Generating {request.num_rows} rows may take some time...")
        
        chunk_number = 1
        
        while remaining_rows > 0:
            chunk_size = min(self.config.max_rows_per_chunk, remaining_rows)
            
            print_user(f"\n   📦 Chunk {chunk_number}: Generating {chunk_size} rows (remaining: {remaining_rows})")
            
            chunk_data = self.generate_data_chunk(
                schema=schema,
                region=request.region,
                chunk_size=chunk_size,
                start_id=current_id
            )
            
            if not chunk_data:
                print_error(f"Failed to generate chunk {chunk_number}")
                if len(all_data) > 0:
                    print_warning(f"Stopping early. Generated {len(all_data)} rows total.")
                    break
                else:
                    print_error(f"No data generated at all.")
                    return pd.DataFrame()
            
            print_success(f"Generated {len(chunk_data)} rows")
            
            all_data.extend(chunk_data)
            current_id += len(chunk_data)
            remaining_rows -= len(chunk_data)
            chunk_number += 1
            
            if remaining_rows > 0:
                time.sleep(2)  # Slightly longer delay for reliability
        
        if all_data:
            df = pd.DataFrame(all_data)
            print_success(f"Total generated: {len(df)} rows")
            
            if len(df) > request.num_rows:
                df = df.head(request.num_rows)
                print_info(f"Trimmed to requested {request.num_rows} rows")
            
            return df
        else:
            print_error("No data was generated")
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
            
            print_user(f"\n{'='*60}")
            print_success("GENERATION COMPLETE!")
            print_user('='*60)
            print_info(f"File: {filename}")
            print_info(f"Rows generated: {len(df)}")
            print_info(f"Region: {request.region}")
            print_info(f"Columns: {list(df.columns)}")
            print_user('='*60)
            
            print_user("\n📄 First 5 rows:")
            print(df.head().to_string(index=False))
            
            return filename
            
        except Exception as e:
            logger.error(f"Error saving CSV: {e}")
            print_error(f"Error saving CSV: {e}")
            
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
                print_success(f"Saved using alternative method: {filename}")
                return filename
            except Exception as e2:
                print_error(f"Alternative save also failed: {e2}")
                raise


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Generate region-specific data using AI")
    parser.add_argument("--input", "-i", help="Input text prompt")
    parser.add_argument("--file", "-f", help="Input file path (CSV, JSON, or text)")
    parser.add_argument("--region", "-r", help="Target region (highest priority)")
    parser.add_argument("--rows", "-n", type=int, help="Number of rows (1-MAX_ROWS_LIMIT, highest priority)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    try:
        if args.verbose:
            logger.setLevel(logging.DEBUG)
            print_info("Verbose logging enabled")
        
        # Initialize
        config = Config()
        ai_client = AzureAIClient(config)
        generator = DataGenerator(config, ai_client)
        
        # Get input
        input_text = args.input
        if not input_text:
            if args.file:
                input_text = f"Generate data similar to {args.file}"
            else:
                input_text = input("\n📝 Enter your data request: ").strip()
                if not input_text:
                    input_text = "Generate sample data"
        
        print_user(f"\n{'='*60}")
        print_user("🚀 DATA GENERATION REQUEST")
        print_user('='*60)
        if args.input:
            print_info(f"Input prompt: {args.input}")
        if args.file:
            print_info(f"Input file: {args.file}")
        if args.region:
            print_info(f"CLI Region: {args.region}")
        if args.rows:
            print_info(f"CLI Rows: {args.rows}")
        print_user('='*60)
        
        # Parse request with CLI args as priority
        request = generator.parse_user_input(
            input_text=input_text,
            input_file=args.file,
            cli_region=args.region,
            cli_rows=args.rows
        )
        
        print_user(f"\n✅ Final Parameters:")
        print_info(f"   Region: {request.region}")
        print_info(f"   Rows: {request.num_rows}")
        if request.input_file:
            print_info(f"   Input file: {request.input_file}")
        if request.user_specified_columns:
            print_info(f"   User columns: {request.user_specified_columns}")
        
        # Analyze request to get schema
        schema = generator.analyze_request(request)
        
        print_user(f"\n📋 Schema: {schema['data_type']}")
        print_user("Columns to generate:")
        for i, col in enumerate(schema["columns"], 1):
            print_info(f"   {i:2d}. {col['name']} ({col['type']}) - {col['description']}")
        
        # Confirm generation
        confirm = input("\n🚀 Proceed with generation? (y/n): ").strip().lower()
        if confirm not in ['y', 'yes', '']:
            print_info("Generation cancelled.")
            return 0
        
        # Generate data
        df = generator.generate_all_data(request, schema)
        
        if df.empty:
            print_error("No data was generated.")
            print_user("\n📋 Possible solutions:")
            print_info("1. Try with fewer rows (e.g., --rows 5)")
            print_info("2. Try a simpler request")
            print_info("3. Check Azure OpenAI service status")
            return 1
        
        # Save to CSV
        filename = generator.save_to_csv(df, request)
        print_success(f"Data saved to: {os.path.abspath(filename)}")
        
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
                print_success("File opened.")
            except:
                print_warning("Could not open file automatically.")
        
        return 0
        
    except KeyboardInterrupt:
        print_user("\n\n⏹ Operation cancelled by user.")
        return 130
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        print_error(f"ERROR: {str(e)}")
        print_user("\n🔧 Troubleshooting:")
        print_info("1. Check .env file has correct Azure credentials")
        print_info("2. Verify API key and endpoint are correct")
        print_info("3. Check deployment name matches Azure portal")
        print_info("4. Try a simpler request with fewer rows (e.g., --rows 5)")
        print_info("5. Check Azure OpenAI quota and limits")
        return 1


if __name__ == "__main__":
    # Check if required environment variables are set
    required_vars = ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY"]
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print_error(f"Missing environment variables: {', '.join(missing_vars)}")
        print_user("\n📝 Please ensure your .env file contains:")
        print_info("AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/")
        print_info("AZURE_OPENAI_API_KEY=your_api_key_here")
        print_info("AZURE_OPENAI_API_VERSION=2024-02-01")
        print_info("AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini")
        print_user("\n💡 Optional configuration in .env:")
        print_info("MAX_ROWS_LIMIT=500  # Maximum rows allowed (default: 200)")
        sys.exit(1)
    
    sys.exit(main())