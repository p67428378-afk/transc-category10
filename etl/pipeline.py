import pandas as pd
from datetime import datetime

def clean_transaction_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw transaction data in a DataFrame.
    - Handles null values in original_description.
    - Trims whitespace and normalizes spaces in original_description.
    - Converts amount to float.
    - Converts transaction_date to datetime.
    - Creates a 'cleaned_description' column.
    """
    if df.empty:
        # Ensure 'cleaned_description' column exists even for empty DataFrame
        if 'cleaned_description' not in df.columns:
            df['cleaned_description'] = pd.Series(dtype=str)
        if 'amount' not in df.columns:
            df['amount'] = pd.Series(dtype=float)
        if 'transaction_date' not in df.columns:
            df['transaction_date'] = pd.Series(dtype='datetime64[ns]')
        return df

    # Make a copy to avoid SettingWithCopyWarning and ensure a new DataFrame is returned
    df_cleaned = df.copy()

    # Handle nulls and clean description
    if 'original_description' in df_cleaned.columns:
        df_cleaned['original_description'] = df_cleaned['original_description'].fillna("").astype(str)
        df_cleaned['cleaned_description'] = df_cleaned['original_description'].str.strip().replace(r'\s+', ' ', regex=True)
    else:
        df_cleaned['cleaned_description'] = pd.Series(dtype=str) # Add empty column if original_description is missing

    # Convert amount to float, coercing errors
    if 'amount' in df_cleaned.columns:
        df_cleaned['amount'] = pd.to_numeric(df_cleaned['amount'], errors='coerce')
    else:
        df_cleaned['amount'] = pd.Series(dtype=float)

    # Convert transaction_date to datetime, coercing errors, explicitly setting dayfirst=False for MM/DD/YYYY
    if 'transaction_date' in df_cleaned.columns:
        df_cleaned['transaction_date'] = pd.to_datetime(df_cleaned['transaction_date'], errors='coerce', dayfirst=False)
    else:
        df_cleaned['transaction_date'] = pd.Series(dtype='datetime64[ns]')

    return df_cleaned

def deduplicate_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicates transactions based on a combination of key fields.
    Assumes 'cleaned_description' is already present from clean_transaction_data.
    """
    if df.empty:
        return df

    # Make a copy to avoid SettingWithCopyWarning
    df_deduplicated = df.copy()

    # Define key fields for deduplication
    key_fields = ['user_id', 'cleaned_description', 'amount', 'transaction_date']

    # Filter out key_fields that are not in the DataFrame to prevent errors
    existing_key_fields = [field for field in key_fields if field in df_deduplicated.columns]

    if not existing_key_fields:
        # If no key fields exist, no meaningful deduplication can occur, return original
        return df_deduplicated

    # Drop duplicates based on the existing key fields, keeping the first occurrence
    deduplicated_df = df_deduplicated.drop_duplicates(subset=existing_key_fields, keep='first')

    return deduplicated_df
