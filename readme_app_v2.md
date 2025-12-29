🔹 Step-by-step flow

1. Loads environment variables
Finds .env automatically
Validates Azure OpenAI credentials

2. Parses user intent (with AI)
Detects:
region
number of rows
data type (customer, sales, employee, etc.)
Falls back to regex if AI fails

3. Confirms with user
Region
Row count
Final “Proceed?” check

4. Generates a schema (AI-driven)
CSV-safe
Max 10 columns
Allowed types only (string, int, float, date, boolean)

5. Chunked data generation
Max 30 rows per AI call
Automatic retries
Adaptive chunk reduction on failure

6. Assembles final dataset
Combines chunks
Trims excess rows safely

7. Exports to CSV
Timestamped filename
Proper quoting
Prints preview

8. Optional auto-open of file

----------------------------------------
----------------------------------------
----------------------------------------


app_v2:

1️. Simplest (interactive)
python app_v2.py

You’ll be prompted for:
Data request
Region confirmation
Row count confirmation


2️. Fully controlled (recommended)
python app_v2.py --input "Generate similar customers data" --file input_file.csv --rows 100 --region mumbai,india

✔ No guessing
✔ Minimal prompts
✔ Fast and reliable
