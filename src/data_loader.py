# import pandas as pd
# from pathlib import Path

# DATA_PATH = Path("data/raw/ashrae_db2.01.csv")

# def load_raw_data():
#     """
#     Load the raw ASHRAE thermal comfort dataset.
#     """
#     if not DATA_PATH.exists():
#         raise FileNotFoundError(
#             "Dataset not found. Please place ashrae_db.01.csv in data/raw/"
#         )
    
#     df = pd.read_csv(DATA_PATH)
#     return df

# print(f"Successfully imported dataset from: {DATA_PATH}\n")

import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/sample_ashrae.csv")

def load_raw_data():
    """Load the raw ASHRAE thermal comfort dataset."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Dataset not found. Please place ashrae_db.01.csv in data/raw/"
        )
    
    df = pd.read_csv(DATA_PATH, low_memory=False) # low_memory=False often helps with large CSVs
    return df

# 1. Load the data into a variable named 'df'
df = load_raw_data()
print(f"Successfully imported dataset from: {DATA_PATH}\n")

# --- DATA INSPECTION ---

# Check Shape (Rows, Columns)
print(f"Shape of dataset: {df.shape}")

# Check Size (Total number of elements: rows * columns)
print(f"Total size: {df.size}")

# Check for Null values (Sums the missing values per column)
print("\nMissing values per column:")
print(df.isnull().sum())

# Statistical Summary
print("\nStatistical Description:")
print(df.describe())

# Preview the first 5 rows
print("\nFirst 5 rows:")
print(df.head())
