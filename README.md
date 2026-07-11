# EDA Workflow

An AI-powered exploratory data analysis workflow that performs consistent, first-pass analysis of datasets using LangChain and LangGraph. The workflow runs a fixed set of analysis tools, uses an LLM to extract observations after each step, and synthesizes findings into a summary with actionable recommendations.

## How It Works

The workflow follows a sequential process:

1. **Analyze** — Runs predefined pandas-based analysis tools on the dataset
2. **Observe** — After each tool, the LLM extracts concise, evidence-based observations from the results
3. **Synthesize** — Once all tools have run, the LLM summarizes findings and provides actionable recommendations

This approach combines deterministic pandas-based analysis with LLM-powered interpretation focused on **data quality and reliability**.

### Pipeline Order

```
profile_dataset
  → extract_observations
  → analyze_missingness
  → extract_observations
  → validate_data_integrity
  → extract_observations
  → compute_aggregates
  → extract_observations
  → analyze_relationships
  → extract_observations
  → analyze_timeseries
  → extract_observations
  → synthesize_findings
```

Running `example_usage.py` also saves a visual diagram of this graph to `graph.png`.

## Analysis Tools

Each analysis step loads the dataframe from workflow state, runs deterministic checks, and stores structured results under `results["<step_name>"]`.

### `profile_dataset_node`

Generates a baseline dataset profile.

| Output key | Description |
|---|---|
| `shape` | Row and column counts |
| `columns` | All column names |
| `dtypes` | Data type per column |
| `numeric_columns` | Numeric column names |
| `categorical_columns` | Object/category column names |
| `numeric_summary` | `describe()` statistics for numeric columns |
| `categorical_summary` | Top 10 value counts per categorical column |

### `analyze_missingness_node`

Analyzes completeness of the dataset.

| Output key | Description |
|---|---|
| `total_rows` | Total number of rows |
| `missing_count` | Null count per column |
| `missing_percentage` | Null percentage per column |
| `high_missing_columns` | Columns with > 20% missing values |
| `complete_rows` | Rows with no missing values |
| `complete_rows_pct` | Percentage of fully complete rows |

### `validate_data_integrity_node`

Validates row integrity, logical consistency, and statistical outliers.

| Output key | Description |
|---|---|
| `duplicate_rows` | Count and percentage of fully duplicated rows |
| `duplicate_keys` | Duplicate values in ID-like columns (e.g. `Transaction ID`) |
| `consistency_violations` | Rows where `quantity × rate ≠ total` (e.g. `Quantity × Price Per Unit ≠ Total Spent`) |
| `invalid_ranges` | Negative values in count/price/total-like columns |
| `constant_columns` | Columns with only one unique value |
| `statistical_outliers` | IQR-based outliers (1.5× IQR) on aggregate numeric columns, including bounds |

### `compute_aggregates_node`

Computes sums on aggregate numeric columns, with optional time-based grouping when a single date column is detected.

| Output key | Description |
|---|---|
| `numeric_sums` | Sum of all aggregate numeric columns |
| `yearly_numeric_sums` | Sums grouped by calendar year (if one date column exists) |
| `quarterly_numeric_sums` | Sums grouped by calendar quarter (e.g. `2023Q1`) |
| `monthly_numeric_sums` | Sums grouped by calendar month (e.g. `2023-05`) |

### `analyze_relationships_node`

Analyzes relationships between variables.

| Output key | Description |
|---|---|
| `numeric_correlations` | Pearson correlation matrix for aggregate numeric columns |
| `strong_correlations` | Column pairs with \|r\| ≥ 0.5 |
| `categorical_numeric_means` | Mean of aggregate numeric columns grouped by low-cardinality categorical columns (≤ 20 unique values); date columns are excluded |

### `analyze_timeseries_node`

Performs simple timeseries analysis when exactly one date/datetime column is found.

| Output key | Description |
|---|---|
| `date_column` | The column used as the time index |
| `date_range` | Start date, end date, and span in days |
| `unique_dates` | Number of distinct dates |
| `median_gap_days` | Median gap between consecutive unique dates |
| `monthly_record_count` | Row count per calendar month |
| `monthly_pct_change` | Month-over-month percentage change for aggregate numeric columns |
| `strong_monthly_trends` | Months with unusually strong upward or downward changes compared to all other month-to-month changes (leave-one-out z-score ≥ 1.5) |

## Shared Conventions

Several tools apply the same column-detection logic:

### Aggregate numeric columns

Numeric columns are excluded from sums, correlations, outlier detection, and timeseries aggregates when their normalized name contains `_per_` or `_pr_`. This filters out normalized rate columns such as `Price Per Unit` or `price_per_item`, so totals are not distorted.

### Date column detection (`_get_date_columns`)

1. Native `datetime` / `datetime64` columns are used directly.
2. Otherwise, object/string columns are tested only if all non-null values match the ISO date pattern `YYYY-MM-DD`.
3. Parsing uses `format="ISO8601"` to avoid pandas format-inference warnings.

Time-based analysis (`compute_aggregates`, `analyze_timeseries`) runs only when **exactly one** date column is detected.

## Helper Functions

| Function | Description |
|---|---|
| `load_prompt(filename)` | Loads an LLM prompt template from `eda_workflow/prompts/` |
| `_get_date_columns(df)` | Returns native datetime columns and fully parseable ISO date string columns |
| `_normalize_column_name(name)` | Lowercases a column name and replaces spaces with underscores |
| `_get_agg_numeric_columns(df)` | Returns numeric columns excluding normalized rate columns |
| `make_eda_baseline_workflow(...)` | Factory that builds and compiles the LangGraph workflow |

## LLM Steps

### `extract_observations_node`

Runs after each analysis step. Uses `extract_observations_system.txt` and `extract_observations_human.txt` to produce 1–2 concise, actionable, evidence-based observations tied to the step's results. Observations focus on **data quality**, not business interpretation.

### `synthesize_findings_node`

Runs after all analysis steps. Uses `synthesize_findings_system.txt` and `synthesize_findings_human.txt` to produce:

- A 2–3 sentence summary of key findings
- 3–5 actionable recommendations

## `EDAWorkflow` API

| Method | Returns | Description |
|---|---|---|
| `invoke_workflow(filepath)` | `None` | Loads a CSV, runs the full pipeline, stores results in `self.response` |
| `get_summary()` | `str` | Final synthesized summary |
| `get_recommendations()` | `list[str]` | Final actionable recommendations |
| `get_observations()` | `dict[str, list[str]]` | LLM observations keyed by analysis step |
| `get_results()` | `dict` | Full structured results from all analysis steps |

## Setup

### Prerequisites

- **Python 3.10 or 3.11**
- **Poetry** (dependency manager)
- **OpenAI API Key**

### Installation Steps

1. **Install Poetry** (if not already installed):

   **Windows (PowerShell)**:
   ```powershell
   (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
   ```

   **macOS/Linux**:
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

   After installation, restart your terminal. If `poetry` command is not found:
   - **Windows**: Add `%APPDATA%\Python\Scripts` to your system PATH
   - **macOS/Linux**: Add `export PATH="$HOME/.local/bin:$PATH"` to your `~/.bashrc` or `~/.zshrc`

2. **Install dependencies**:
   ```bash
   poetry install
   ```

   This will install all dependencies with the exact versions specified in `poetry.lock`, ensuring consistency across all environments.

3. **Set up your OpenAI API key**:

   **Windows**:
   ```powershell
   copy .env.example .env
   ```

   **macOS/Linux**:
   ```bash
   cp .env.example .env
   ```

   Then edit `.env` and add your OpenAI API key:
   ```
   OPENAI_API_KEY=sk-your-key-here
   ```

### Multiple Python Versions?

If you have multiple Python versions installed and want to use a specific one:

```bash
# Tell Poetry which Python to use
poetry env use python3.11  # or python3.10

# Then install dependencies
poetry install
```

Poetry will create a virtual environment with your chosen Python version.

## Usage

### Python API

```python
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from eda_workflow.eda_workflow import EDAWorkflow

load_dotenv()

# Initialize the workflow with an LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
workflow = EDAWorkflow(model=llm)

# Run analysis on a dataset
workflow.invoke_workflow("data/cafe_sales.csv")

# Retrieve results
summary = workflow.get_summary()              # str
recommendations = workflow.get_recommendations()  # list[str]
observations = workflow.get_observations()    # dict[str, list[str]]
results = workflow.get_results()              # dict
```

### Running the Example

```bash
poetry run python example_usage.py
```

This runs a full analysis on the sample `cafe_sales.csv` dataset, prints results and observations for each step, and saves the workflow graph to `graph.png`.

### Sample Dataset

`data/cafe_sales.csv` contains cafe transaction records with columns:

- `Transaction ID`, `Item`, `Quantity`, `Price Per Unit`, `Total Spent`, `Transaction Date`

## Project Structure

```
DS_bootcamp_26_eda_workflow/
├── data/
│   └── cafe_sales.csv             # Sample dataset
├── eda_workflow/
│   ├── __init__.py
│   ├── eda_workflow.py            # Main workflow class, analysis nodes, and graph
│   └── prompts/                   # LLM prompt templates
│       ├── extract_observations_system.txt
│       ├── extract_observations_human.txt
│       ├── synthesize_findings_system.txt
│       └── synthesize_findings_human.txt
├── .env.example                   # Environment variable template
├── example_usage.py               # Example script
├── graph.png                      # Workflow diagram (generated by example_usage.py)
├── pyproject.toml                 # Dependencies configuration
├── poetry.lock                    # Locked dependency versions
└── README.md
```

**Important**: The `poetry.lock` file is committed to ensure all users get identical, tested dependency versions.
