import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('your_data.csv')  # Update with your file path

# ── Pipeline Steps ──────────────────────────────────

# Step 1: Clean text in 'city': trim_whitespace, collapse_whitespace, lowercase
df[['city']] = df[['city']].apply(lambda x: x.str.strip())
df[['city']] = df[['city']].apply(lambda x: x.str.lower())

# Step 2: Clean text in 'name': trim_whitespace, collapse_whitespace
df[['name']] = df[['name']].apply(lambda x: x.str.strip())

# Step 3: Clean text in 'department': lowercase
df[['department']] = df[['department']].apply(lambda x: x.str.lower())

# Step 4: Clean text in 'is_active': lowercase
df[['is_active']] = df[['is_active']].apply(lambda x: x.str.lower())

# Step 5: Impute 'age' (3 missing values) using median
df[['age']] = df[['age']].fillna(df[['age']].median())

# Step 6: Impute 'email' (3 missing values) using mode
for col in ['email']:
    df[col] = df[col].fillna(df[col].mode().iloc[0])

# Step 7: Impute 'salary' (4 missing values) using median
df[['salary']] = df[['salary']].fillna(df[['salary']].median())

# Step 8: Impute 'department' (1 missing values) using mode
for col in ['department']:
    df[col] = df[col].fillna(df[col].mode().iloc[0])

# Step 9: Impute 'rating' (3 missing values) using median
df[['rating']] = df[['rating']].fillna(df[['rating']].median())

# Step 10: Handle 4 outliers in 'salary' (8.0%) using IQR method → clip
for col in ['salary']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    df[col] = df[col].clip(lower=Q1 - 1.5*IQR, upper=Q3 + 1.5*IQR)

# Step 11: Impute 'is_active' (1 missing values) using mode
for col in ['is_active']:
    df[col] = df[col].fillna(df[col].mode().iloc[0])

# Step 12: Handle 1 outliers in 'rating' (2.0%) using IQR method → flag
# Handle outliers: method=iqr, behavior=flag

# Step 13: Handle 1 outliers in 'rating' (2.0%) using IQR method → flag
# Handle outliers: method=iqr, behavior=flag

# Step 14: Convert 'join_date' from object to datetime64[ns] (confidence: 100%)
df['join_date'] = pd.to_datetime(df['join_date'], errors='coerce')

# Step 15: Convert 'is_active' from object to bool (confidence: 100%)
df['is_active'] = df['is_active'].astype('bool')

# Step 16: Convert 'is_active' from bool to category (confidence: 96%)
df['is_active'] = df['is_active'].astype('category')

# Step 17: Convert 'rating_outlier_flag' from int64 to category (confidence: 96%)
df['rating_outlier_flag'] = df['rating_outlier_flag'].astype('category')

# Step 18: Impute missing values in ['join_date'] using mode
for col in ['join_date']:
    df[col] = df[col].fillna(df[col].mode().iloc[0])

# ── Export Result ────────────────────────────────────
df.to_csv('cleaned_data.csv', index=False)
print(f'Final shape: {df.shape}')