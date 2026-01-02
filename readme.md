**app_v1.py:**
ADDED_FEATURES:
- Can generate any data as asked by user request.
- Can understand user prompt and give output as csv

LIMITATIONS:
- MAX_ROWS hard coded as 200, can't exceed beyond 200.
- Would also fail if column count rasies beyond 8

**app_v2_withincreasedrows.py**
ADDED_FEATURES:
- rows can be increased as per user request, but there is hard limit also mentioned in .env file. It is easily configurable. Default max_rows is 200, hard coded in code script.

LIMITATIONS:
- Would also fail if column count rasies betond 8
- Chunk_size is hard-cded in script to 30 rows per chunk. We will make it dynamic in next version of script so that app doesnt fail even if column count input by user increases, and chunking can be done intelligently.


**app_v3.py:**
How It Works Now:
   - Minimal test batch: Only 2 rows generated for format sampling
   - Format extraction: Extracts formats from test batch (dates, booleans, numbers)
   - Consistency enforcement: All subsequent chunks follow the exact same formats
   - User preview: Shows sample data format before full generation
   - Configurable: All limits are adjustable via .env file

ADDED_FEATURES:
1. User requests data with 12 columns
2. System warns: "⚠️ High column count: 12 columns"
3. Offers choice: "Generate with 12 columns or reduce to 8?"
4. If user chooses "full":
   - Generates 5-row test batch
   - Analyzes token usage
   - Calculates safe chunk size (e.g., 8 rows per request)
   - Shows: "Safe chunk size: 8 rows, Estimated API calls: 13"
   - Asks for confirmation due to many API calls
5. Generates data with dynamic chunking
6. Validates output matches schema


    Key Enhancements Made:
    1. Dynamic Chunk Sizing:
    - TokenEstimator class to calculate tokens per row
    - Test batch generation (5 rows) to estimate real token usage
    - Safety factor (70%) to never hit token limits
    - Adaptive retry logic that reduces chunk size on errors

    2. Column Management:
    - **Configurable column limits** via environment variables
    - User warnings when column count exceeds threshold (8 by default)
    - User choice to reduce columns or proceed with high cost
    - Column validation in final output

    3. Robust Error Handling:
    - JSON parsing improvements with multiple fallback strategies
    - Adaptive retry logic based on error type
    - Final data validation to catch schema mismatches
    - Clear user communication about potential issues







    Safety Mechanisms:
    -   Configurable column limits via environment variables
    -   User warnings when column count exceeds threshold (8 by default)
    -   Column validation in final output
    -   70% safety factor - never use full token limit
    -   Minimum rows per chunk - prevent degenerate cases
    -   Adaptive retries - reduce chunk size on errors
    -   User confirmation for expensive operations
    -   Validation step - ensure data quality
    -   This approach should prevent the JSON parsing failures you mentioned while giving users control over the cost/quality trade-off.