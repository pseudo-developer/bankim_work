# app.py
import os
import json
import pandas as pd
import argparse
import sys
import time
import math
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


class TokenEstimator:
    """Estimate token usage for data generation requests"""
    
    def __init__(self, config):
        self.config = config
    
    def estimate_tokens_per_row(self, schema: Dict[str, Any], sample_row: Dict[str, Any]) -> float:
        """Estimate tokens needed per row based on schema and sample data"""
        total_tokens = 0
        
        # Estimate based on column names and sample values
        for col in schema["columns"]:
            col_name = col["name"]
            
            # Tokens for column name in JSON (with quotes, colon)
            col_name_tokens = len(f'"{col_name}":') * self.config.tokens_per_char_estimate
            
            # Tokens for value (use sample or estimate based on type)
            if col_name in sample_row:
                sample_value = sample_row[col_name]
                if isinstance(sample_value, (list, dict)):
                    value_str = json.dumps(sample_value)
                else:
                    value_str = str(sample_value)
                value_tokens = len(value_str) * self.config.tokens_per_char_estimate
            else:
                # Estimate based on data type
                if col["type"] == "string":
                    value_tokens = 20  # Average short string
                elif col["type"] == "integer":
                    value_tokens = 3   # Average number
                elif col["type"] == "float":
                    value_tokens = 5   # Decimal number
                elif col["type"] == "date":
                    value_tokens = 12  # YYYY-MM-DD
                elif col["type"] == "boolean":
                    value_tokens = 1   # true/false
                else:
                    value_tokens = 10  # Default
            
            total_tokens += col_name_tokens + value_tokens
        
        # Add JSON structure tokens (commas, braces)
        total_tokens += len(schema["columns"]) * 2  # Commas and formatting
        
        return total_tokens
    
    def calculate_safe_chunk_size(self, 
                                 schema: Dict[str, Any], 
                                 sample_data: List[Dict],
                                 max_tokens_per_request: int) -> int:
        """Calculate safe chunk size based on token estimation"""
        
        if not sample_data:
            # If no sample data, use conservative estimate
            avg_tokens_per_col = 25
            estimated_tokens_per_row = len(schema["columns"]) * avg_tokens_per_col
        else:
            # Calculate average from sample
            token_counts = []
            for row in sample_data[:3]:  # Use first 3 samples
                token_counts.append(self.estimate_tokens_per_row(schema, row))
            
            if token_counts:
                estimated_tokens_per_row = sum(token_counts) / len(token_counts)
            else:
                estimated_tokens_per_row = 30 * len(schema["columns"])  # Fallback
        
        # Add JSON overhead
        total_overhead = self.config.json_overhead_tokens
        
        # Calculate safe rows
        available_tokens = max_tokens_per_request * self.config.safety_factor
        tokens_for_data = available_tokens - total_overhead
        
        if estimated_tokens_per_row <= 0:
            return self.config.min_rows_per_chunk
        
        max_possible_rows = int(tokens_for_data / estimated_tokens_per_row)
        
        # Apply limits
        safe_rows = max(self.config.min_rows_per_chunk, 
                       min(max_possible_rows, self.config.max_rows_per_chunk))
        
        logger.info(f"Token estimation: {estimated_tokens_per_row:.1f} tokens/row, "
                   f"safe chunk size: {safe_rows}")
        
        return safe_rows


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
        self.max_tokens_per_request = int(os.getenv("MAX_TOKENS_PER_REQUEST", "4000"))
        self.max_file_content_chars = 2000  # Limit file content for AI analysis
        self.file_sample_rows = 5  # Only sample 5 rows for analysis (reduced from 100)
        
        # NEW: Configurable column limits
        self.max_columns_total = int(os.getenv("MAX_COLUMNS_LIMIT", "15"))
        self.max_columns_warning = int(os.getenv("COLUMNS_WARNING_THRESHOLD", "8"))
        self.min_rows_per_chunk = int(os.getenv("MIN_ROWS_PER_CHUNK", "3"))
        
        # NEW: Token estimation constants
        self.tokens_per_char_estimate = 0.25  # Rough estimate (4 chars ≈ 1 token)
        self.safety_factor = float(os.getenv("SAFETY_FACTOR", "0.7"))  # Use only 70% of available tokens
        self.json_overhead_tokens = 200  # JSON structure overhead
        
        # NEW: Minimal test batch size
        self.test_batch_rows = 2  # Only 2 rows for test batch
        
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
        print_info(f"Maximum columns allowed: {self.max_columns_total}")


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
        self.token_estimator = TokenEstimator(config)  # Add token estimator
    
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
    
    def analyze_request(self, request: UserRequest) -> Dict[str, Any]:
        """Analyze user request to determine data schema - ENHANCED for user columns"""
        
        # ================== CHECK FOR USER-SPECIFIED COLUMNS ==================
        if request.user_specified_columns:
            print_info(f"Using user-specified columns: {request.user_specified_columns}")
            
            # Limit columns to maximum allowed
            if len(request.user_specified_columns) > self.config.max_columns_total:
                print_warning(f"User specified {len(request.user_specified_columns)} columns, limiting to {self.config.max_columns_total}")
                request.user_specified_columns = request.user_specified_columns[:self.config.max_columns_total]
            
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
                
                # Limit columns to maximum allowed
                if len(file_columns) > self.config.max_columns_total:
                    print_warning(f"File has {len(file_columns)} columns, limiting to {self.config.max_columns_total}")
                    file_columns = file_columns[:self.config.max_columns_total]
                
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
            5. Maximum {self.config.max_columns_total} columns for reliability
            
            Provide JSON with:
            1. data_type: Type of data (customer_records, sales_data, etc.)
            2. columns: List of column objects with name, type, description
            3. description: Brief description
            4. examples: Example values for each column (2-3 examples)
            
            Keep it simple and realistic for {request.region}.
            """
            
            try:
                response = self.ai_client.generate_completion(prompt, max_tokens=1500)
                
                # FIXED: More robust JSON parsing
                try:
                    schema = json.loads(response)
                except json.JSONDecodeError:
                    # Try to extract JSON from the response
                    match = re.search(r'\{.*\}', response, re.DOTALL)
                    if match:
                        schema = json.loads(match.group(0))
                    else:
                        # If still can't parse, use fallback
                        raise ValueError(f"Could not parse JSON from AI response: {response[:200]}...")
                
                # Validate and fix schema if needed
                schema = self._validate_and_fix_schema(schema)
                
                return schema
                
            except Exception as e:
                logger.error(f"Error analyzing request: {e}")
                print_warning(f"AI schema generation failed: {str(e)}")
                print_info("Using fallback schema...")
                return self._get_text_based_fallback_schema(request)
    
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
        
        # Limit to maximum columns for reliability
        if len(schema["columns"]) > self.config.max_columns_total:
            print_warning(f"Limiting columns from {len(schema['columns'])} to {self.config.max_columns_total} for reliability")
            schema["columns"] = schema["columns"][:self.config.max_columns_total]
        
        # Fix array types to strings
        for col in schema["columns"]:
            if col.get("type") in ["array", "object"]:
                col["type"] = "string"
                print_warning(f"Changed column '{col['name']}' type from array/object to string")
        
        # Fix examples if they contain arrays
        if "examples" in schema:
            for col_name, examples in schema["examples"].items():
                if isinstance(examples, list):
                    # Convert any arrays in examples to strings
                    for i, example in enumerate(examples):
                        if isinstance(example, list):
                            schema["examples"][col_name][i] = ",".join(map(str, example))
        
        return schema
    
    def _extract_format_guidelines(self, sample_data: List[Dict], schema: Dict[str, Any]) -> Dict[str, Any]:
        """Extract data format guidelines from sample data to ensure consistency"""
        if not sample_data:
            return {}
        
        guidelines = {}
        
        for col_info in schema["columns"]:
            col_name = col_info["name"]
            col_type = col_info["type"]
            
            # Collect sample values for this column
            sample_values = []
            for row in sample_data:
                if col_name in row:
                    sample_values.append(row[col_name])
            
            if sample_values:
                # Take first sample as reference
                first_sample = sample_values[0]
                
                guidelines[col_name] = {
                    "type": col_type,
                    "example": first_sample,
                    "format_hint": self._get_format_hint(first_sample, col_type)
                }
        
        return guidelines
    
    def _get_format_hint(self, value: Any, col_type: str) -> str:
        """Generate a format hint from a sample value"""
        if value is None:
            return ""
        
        if col_type == "date":
            if isinstance(value, str):
                if re.match(r'\d{4}-\d{2}-\d{2}', value):
                    return "YYYY-MM-DD"
                elif re.match(r'\d{2}/\d{2}/\d{4}', value):
                    return "MM/DD/YYYY"
                elif re.match(r'\d{2}-\d{2}-\d{4}', value):
                    return "DD-MM-YYYY"
        
        elif col_type == "float":
            if isinstance(value, (int, float)):
                value_str = str(value)
                if '.' in value_str:
                    decimals = len(value_str.split('.')[1])
                    return f"{decimals} decimal places"
        
        elif col_type == "boolean":
            if isinstance(value, bool):
                return "boolean"
            elif isinstance(value, str):
                if value.upper() in ["TRUE", "FALSE"]:
                    return "TRUE/FALSE"
                elif value.lower() in ["true", "false"]:
                    return "true/false"
        
        return ""
    
    def _create_test_batch_prompt(self, schema: Dict[str, Any], region: str, 
                                 chunk_size: int, start_id: int) -> str:
        """Create prompt for test batch (minimal, focused on format)"""
        column_names = [col["name"] for col in schema["columns"]]
        
        return f"""
        Create EXACTLY {chunk_size} rows of sample data for {region}.
        
        COLUMNS: {', '.join(column_names)}
        
        IMPORTANT: This is a test batch for format consistency.
        Use realistic but SIMPLE values that can be replicated.
        
        FORMAT REQUIREMENT: Return a JSON object with one key "data" that contains an array.
        Example format:
        {{
          "data": [
            {{"id": 1, "name": "Sample 1", "date": "2023-01-15"}}
          ]
        }}
        
        Start IDs from {start_id}.
        Keep values minimal but representative.
        
        Return ONLY the JSON object with "data" key.
        """
    
    def _create_consistent_prompt(self, schema: Dict[str, Any], region: str, 
                                 chunk_size: int, start_id: int,
                                 format_guidelines: Dict[str, Any]) -> str:
        """Create prompt that ensures consistency with test batch"""
        column_names = [col["name"] for col in schema["columns"]]
        
        # Build format instructions from guidelines
        format_instructions = ""
        if format_guidelines:
            format_instructions = "\nFORMAT CONSISTENCY REQUIREMENTS (based on test batch):\n"
            for col_name, guidelines in format_guidelines.items():
                if "example" in guidelines:
                    example = guidelines["example"]
                    format_hint = guidelines.get("format_hint", "")
                    format_instructions += f"- {col_name}: Follow format like '{example}'"
                    if format_hint:
                        format_instructions += f" ({format_hint})"
                    format_instructions += "\n"
        
        return f"""
        Create {chunk_size} rows of data for {region}.
        
        COLUMNS: {', '.join(column_names)}
        
        {format_instructions}
        
        CRITICAL REQUIREMENTS:
        1. Maintain EXACTLY the same data formats as shown in the examples above
        2. Date formats must be identical (same pattern: DD-MM-YYYY, MM/DD/YYYY, or YYYY-MM-DD)
        3. Number formats must match (same decimal places, same thousand separators if any)
        4. Boolean values must use the same representation (TRUE/FALSE or true/false)
        5. String formats must be similar (same capitalization patterns)
        
        FORMAT REQUIREMENT: Return a JSON object with one key "data" that contains an array.
        Example of correct format:
        {{
          "data": [
            {{"id": 1, "name": "Consistent Format", "date": "2023-01-15"}},
            {{"id": 2, "name": "Same Style Here", "date": "2023-01-16"}}
          ]
        }}
        
        Start IDs from {start_id}.
        Make data realistic for {region}.
        
        Return ONLY the JSON object with the "data" key.
        """
    
    def _create_regular_prompt(self, schema: Dict[str, Any], region: str, 
                              chunk_size: int, start_id: int) -> str:
        """Create prompt for regular batch generation (fallback without guidelines)"""
        column_names = [col["name"] for col in schema["columns"]]
        
        return f"""
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
    
    def _parse_ai_response(self, response: str, attempt: int, chunk_size: int) -> List[Dict]:
        """Parse AI response with robust error handling"""
        try:
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
                    raise ValueError(f"Invalid JSON response on attempt {attempt}")
            
            # Extract data from the response
            if isinstance(parsed, dict) and "data" in parsed:
                data = parsed["data"]
                if isinstance(data, list):
                    logger.info(f"Extracted {len(data)} rows from 'data' key")
                    return data
            
            # Alternative: if response is already an array
            elif isinstance(parsed, list):
                logger.info(f"Found {len(parsed)} rows as direct array")
                return parsed
            
            # Alternative: look for any array in the object
            elif isinstance(parsed, dict):
                for key, value in parsed.items():
                    if isinstance(value, list):
                        logger.info(f"Extracted {len(value)} rows from '{key}' key")
                        return value
            
            raise ValueError("No data array found in response")
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"JSON parse error (attempt {attempt}): {e}")
            raise
    
    def generate_data_chunk(self, schema: Dict[str, Any], region: str, chunk_size: int, 
                           start_id: int = 1, attempt: int = 1, 
                           is_test_batch: bool = False,
                           format_guidelines: Optional[Dict] = None) -> List[Dict]:
        """Generate a chunk of data rows with format consistency"""
        
        # Add chunk size validation
        if chunk_size <= 0:
            logger.error(f"Invalid chunk size: {chunk_size}")
            return []
        
        # For test batches, use test prompt
        if is_test_batch:
            prompt = self._create_test_batch_prompt(schema, region, chunk_size, start_id)
            max_tokens = min(2000, self.config.max_tokens_per_request)
        else:
            # For regular chunks, use consistent prompt with guidelines
            prompt = self._create_consistent_prompt(
                schema, region, chunk_size, start_id, format_guidelines
            )
            max_tokens = self.config.max_tokens_per_request
        
        try:
            print_info(f"Requesting {chunk_size} rows from AI...")
            response = self.ai_client.generate_completion(prompt, max_tokens=max_tokens)
            
            return self._parse_ai_response(response, attempt, chunk_size)
            
        except Exception as e:
            logger.error(f"Error generating chunk: {e}")
            
            # Adaptive retry logic
            if attempt <= 3:
                # Calculate new chunk size based on error type
                if "token" in str(e).lower() or "length" in str(e).lower():
                    # Token limit error - reduce significantly
                    new_size = max(self.config.min_rows_per_chunk, chunk_size // 3)
                else:
                    # Other error - reduce moderately
                    new_size = max(self.config.min_rows_per_chunk, chunk_size // 2)
                
                print_warning(f"Retrying with reduced chunk size: {new_size} rows")
                return self.generate_data_chunk(
                    schema, region, new_size, start_id, attempt + 1, 
                    is_test_batch, format_guidelines
                )
            
            print_error(f"Max retries exceeded for chunk")
            return []
    
    def _validate_generated_data(self, df: pd.DataFrame, schema: Dict[str, Any]):
        """Validate generated data matches schema"""
        print_info("\n🔍 Validating generated data...")
        
        # Check column match
        generated_columns = set(df.columns)
        schema_columns = set(col["name"] for col in schema["columns"])
        
        missing_columns = schema_columns - generated_columns
        extra_columns = generated_columns - schema_columns
        
        if missing_columns:
            print_warning(f"Missing columns in output: {missing_columns}")
        
        if extra_columns:
            print_warning(f"Extra columns in output: {extra_columns}")
        
        # Check for null values
        null_counts = df.isnull().sum()
        # FIXED: Corrected the comment syntax
        high_null_columns = null_counts[null_counts > len(df) * 0.5]  # >50% null
        
        if not high_null_columns.empty:
            print_warning(f"High null values in columns: {list(high_null_columns.index)}")
        
        print_success("Validation complete")
    
    def generate_all_data(self, request: UserRequest, schema: Dict[str, Any]) -> pd.DataFrame:
        """Generate all requested data with dynamic chunking AND consistency"""
        
        # ================== COLUMN COUNT WARNING ==================
        column_count = len(schema["columns"])
        if column_count > self.config.max_columns_warning:
            print_warning(f"⚠️  High column count detected: {column_count} columns")
            print_info(f"High column counts may:")
            print_info("   • Increase AI token usage and costs")
            print_info("   • Reduce rows per API call")
            print_info("   • Potentially cause incomplete responses")
            
            if column_count > self.config.max_columns_warning * 1.5:  # If significantly high
                print_user(f"\n🔧 Column Management Options:")
                print_info(f"   1. Generate with all {column_count} columns (higher cost)")
                print_info(f"   2. Reduce to {self.config.max_columns_warning} essential columns (lower cost)")
                print_info(f"   3. Cancel and re-run with fewer columns")
                
                while True:
                    choice = input(f"\nChoose option (1/2/3): ").strip()
                    
                    if choice == '1':
                        print_info(f"Proceeding with {column_count} columns")
                        break
                    elif choice == '2':
                        # Keep only essential columns
                        essential_columns = schema["columns"][:self.config.max_columns_warning]
                        print_info(f"Reduced to {len(essential_columns)} essential columns")
                        schema["columns"] = essential_columns
                        column_count = len(essential_columns)
                        break
                    elif choice == '3':
                        print_info("Generation cancelled by user.")
                        return pd.DataFrame()
                    else:
                        print_warning("Please enter 1, 2, or 3")
        
        # ================== GENERATE MINIMAL TEST BATCH ==================
        print_user(f"\n🚀 Starting data generation...")
        print_info(f"Target: {request.num_rows} rows for {request.region}")
        print_info(f"Columns: {column_count}")
        
        # NEW: Generate only 1-2 rows for test batch
        test_chunk_size = min(self.config.test_batch_rows, request.num_rows)
        print_info(f"📊 Generating minimal test batch ({test_chunk_size} rows)...")
        
        test_chunk = self.generate_data_chunk(
            schema=schema,
            region=request.region,
            chunk_size=test_chunk_size,
            start_id=1,
            is_test_batch=True
        )
        
        if not test_chunk or len(test_chunk) < 1:
            print_error(f"Failed to generate test batch (got {len(test_chunk) if test_chunk else 0} rows)")
            return pd.DataFrame()
        
        print_success(f"Test batch generated: {len(test_chunk)} sample row(s)")
        
        # Display sample data for user verification
        if test_chunk:
            print_info("📋 Sample data format:")
            sample_df = pd.DataFrame(test_chunk[:1])  # Show just first row
            print(sample_df.to_string(index=False))
        
        # ================== EXTRACT FORMAT GUIDELINES FROM TEST BATCH ==================
        format_guidelines = self._extract_format_guidelines(test_chunk, schema)
        
        # ================== DYNAMIC CHUNK SIZE CALCULATION ==================
        # Calculate safe chunk size based on test batch
        safe_chunk_size = self.token_estimator.calculate_safe_chunk_size(
            schema=schema,
            sample_data=test_chunk,
            max_tokens_per_request=self.config.max_tokens_per_request
        )
        
        # Apply upper limit
        safe_chunk_size = min(safe_chunk_size, self.config.max_rows_per_chunk)
        
        print_success(f"Token analysis complete")
        print_info(f"Safe chunk size: {safe_chunk_size} rows per request")
        
        # Adjust if safe chunk size is very small
        if safe_chunk_size < self.config.min_rows_per_chunk:
            safe_chunk_size = self.config.min_rows_per_chunk
            print_warning(f"Adjusted to minimum chunk size: {safe_chunk_size} rows")
        
        estimated_api_calls = math.ceil(request.num_rows / safe_chunk_size)
        print_info(f"Estimated total API calls: {estimated_api_calls}")
        
        # Confirm if chunk size is very small
        if safe_chunk_size < 5 and request.num_rows > 20:
            print_warning(f"⚠️  Very small chunk size ({safe_chunk_size} rows)")
            print_info("This indicates high token usage per row.")
            print_info("This will result in many API calls and higher cost.")
            
            confirm = input("Continue with many small API calls? (y/n): ").strip().lower()
            if confirm not in ['y', 'yes', '']:
                print_info("Generation cancelled.")
                return pd.DataFrame()
        
        # ================== MAIN GENERATION WITH CONSISTENCY ==================
        all_data = []
        remaining_rows = request.num_rows
        current_id = 1
        
        # Use test batch as first chunk if we have at least 1 valid row
        if test_chunk and len(test_chunk) >= 1:
            # Use all test batch rows as first chunk
            print_info(f"Using test batch as first chunk ({len(test_chunk)} rows)")
            all_data.extend(test_chunk)
            current_id += len(test_chunk)
            remaining_rows -= len(test_chunk)
        
        chunk_number = 2 if all_data else 1
        
        while remaining_rows > 0:
            # Dynamically adjust chunk size based on remaining rows
            current_chunk_size = min(safe_chunk_size, remaining_rows)
            
            # For last chunk, adjust to avoid very small final request
            if remaining_rows - current_chunk_size < 3 and remaining_rows > 3:
                current_chunk_size = remaining_rows
            
            print_user(f"\n   📦 Chunk {chunk_number}: Generating {current_chunk_size} rows (remaining: {remaining_rows})")
            
            # Pass format guidelines to ensure consistency with test batch
            chunk_data = self.generate_data_chunk(
                schema=schema,
                region=request.region,
                chunk_size=current_chunk_size,
                start_id=current_id,
                format_guidelines=format_guidelines  # Pass consistency guidelines
            )
            
            if not chunk_data:
                print_error(f"Failed to generate chunk {chunk_number}")
                
                # Try with smaller chunk size
                if current_chunk_size > self.config.min_rows_per_chunk:
                    print_info(f"Retrying with smaller chunk...")
                    reduced_size = max(self.config.min_rows_per_chunk, current_chunk_size // 2)
                    chunk_data = self.generate_data_chunk(
                        schema=schema,
                        region=request.region,
                        chunk_size=reduced_size,
                        start_id=current_id,
                        format_guidelines=format_guidelines
                    )
                
                if not chunk_data:
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
            
            # Adaptive delay based on chunk size
            if remaining_rows > 0:
                delay = min(3, 1 + (current_chunk_size / 10))  # 1-3 seconds
                time.sleep(delay)
        
        # ================== FINAL PROCESSING ==================
        if all_data:
            df = pd.DataFrame(all_data)
            print_success(f"✅ Total generated: {len(df)} rows")
            
            if len(df) > request.num_rows:
                df = df.head(request.num_rows)
                print_info(f"Trimmed to requested {request.num_rows} rows")
            
            # Final validation
            self._validate_generated_data(df, schema)
            
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
            print_info("2. Try with fewer columns")
            print_info("3. Try a simpler request")
            print_info("4. Check Azure OpenAI service status")
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
        print_info("MAX_COLUMNS_LIMIT=15  # Maximum columns allowed (default: 15)")
        print_info("COLUMNS_WARNING_THRESHOLD=8  # Warn when columns exceed this (default: 8)")
        print_info("MAX_TOKENS_PER_REQUEST=4000  # Max tokens per API call (default: 4000)")
        print_info("SAFETY_FACTOR=0.7  # Use only X% of token limit (default: 0.7)")
        print_info("MIN_ROWS_PER_CHUNK=3  # Minimum rows per chunk (default: 3)")
        sys.exit(1)
    
    sys.exit(main())