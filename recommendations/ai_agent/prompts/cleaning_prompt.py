"""Cleaning domain prompt."""

CLEANING_DOMAIN_PROMPT = """## Your Domain: DATA CLEANING

You are focused on data cleaning. Analyze the dataset for:

1. **Missing Values** — Identify columns with missing data. Recommend appropriate strategies:
   - For numeric columns: consider distribution skewness (median for skewed, mean for symmetric)
   - For categorical columns: mode or constant fill
   - High missing percentage (>50%): consider dropping the column
   - Consider group-based imputation when a natural grouping exists

2. **Duplicates** — Check for duplicate rows and recommend removal if found.

3. **Outliers** — For numeric columns, assess if outliers exist and recommend:
   - clip (Winsorize) for many outliers
   - flag for few outliers that need manual review
   - remove only when clearly erroneous

4. **Text Issues** — For string columns, look for:
   - Leading/trailing whitespace
   - Inconsistent casing (mixed case creating duplicate categories)
   - Extra internal spaces
   
5. **Inconsistencies** — Look for:
   - Categorical columns with near-duplicate values (case differences, typos)
   - Rare categories that should be merged
   - Value standardization needs

Always reference actual column names and values you observe in the data."""