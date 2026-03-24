import pandas as pd
import pytest
from datetime import datetime
from etl.pipeline import clean_transaction_data, deduplicate_transactions

def test_clean_transaction_data_handles_nulls_and_whitespace():
    # Test case: Null values, leading/trailing whitespace, extra spaces
    data = {
        'original_description': ["  Grocery Store  ", "Online Purchase", None, "  Cafe   Latte  "],
        'amount': [100.50, 25.00, 50.00, 12.30],
        'transaction_date': ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"]
    }
    df = pd.DataFrame(data)
    cleaned_df = clean_transaction_data(df.copy())

    # Assertions for expected cleaning
    expected_descriptions = ["Grocery Store", "Online Purchase", "", "Cafe Latte"]
    assert list(cleaned_df['cleaned_description']) == expected_descriptions
    assert not cleaned_df['cleaned_description'].isnull().any()

def test_clean_transaction_data_normalizes_dates_and_amounts():
    # Test case: Different date formats, amount as string
    data = {
        'original_description': ["Test", "Another Test"],
        'amount': ["10.99", 5.50],
        'transaction_date': ["01/05/2023", "2023-01-06 10:30:00"]
    }
    df = pd.DataFrame(data)
    cleaned_df = clean_transaction_data(df.copy())

    # Assertions for expected type normalization
    assert pd.api.types.is_float_dtype(cleaned_df['amount'])
    assert pd.api.types.is_datetime64_any_dtype(cleaned_df['transaction_date'])
    assert cleaned_df['amount'].iloc[0] == 10.99
    assert cleaned_df['transaction_date'].iloc[0] == datetime(2023, 1, 5)

def test_deduplicate_transactions_removes_exact_duplicates():
    # Test case: Exact duplicate rows
    data = {
        'user_id': [1, 1, 2],
        'original_description': ["Coffee", "Coffee", "Lunch"],
        'amount': [5.00, 5.00, 12.00],
        'transaction_date': [datetime(2023, 1, 1), datetime(2023, 1, 1), datetime(2023, 1, 2)]
    }
    df = pd.DataFrame(data)
    deduplicated_df = deduplicate_transactions(df.copy())

    assert len(deduplicated_df) == 2
    assert (deduplicated_df.iloc[0] == df.iloc[0]).all()
    assert (deduplicated_df.iloc[1] == df.iloc[2]).all()

def test_deduplicate_transactions_identifies_duplicates_by_key_fields():
    # Test case: Duplicates based on user_id, amount, transaction_date, and cleaned_description
    data = {
        'user_id': [1, 1, 1, 2],
        'original_description': ["Store A", "Store A  ", "Store A", "Store B"],
        'cleaned_description': ["Store A", "Store A", "Store A", "Store B"],
        'amount': [10.00, 10.00, 10.00, 20.00],
        'transaction_date': [datetime(2023, 2, 1), datetime(2023, 2, 1), datetime(2023, 2, 1), datetime(2023, 2, 2)]
    }
    df = pd.DataFrame(data)
    deduplicated_df = deduplicate_transactions(df.copy())

    assert len(deduplicated_df) == 2
    assert (deduplicated_df.iloc[0] == df.iloc[0]).all()
    assert (deduplicated_df.iloc[1] == df.iloc[3]).all()

def test_deduplicate_transactions_handles_no_duplicates():
    # Test case: No duplicates present
    data = {
        'user_id': [1, 2, 3],
        'original_description': ["Item 1", "Item 2", "Item 3"],
        'amount': [1.0, 2.0, 3.0],
        'transaction_date': [datetime(2023, 3, 1), datetime(2023, 3, 2), datetime(2023, 3, 3)]
    }
    df = pd.DataFrame(data)
    deduplicated_df = deduplicate_transactions(df.copy())

    assert len(deduplicated_df) == 3
    pd.testing.assert_frame_equal(deduplicated_df, df)

def test_clean_transaction_data_empty_dataframe():
    df = pd.DataFrame(columns=['original_description', 'amount', 'transaction_date'])
    cleaned_df = clean_transaction_data(df.copy())
    assert cleaned_df.empty
    assert 'cleaned_description' in cleaned_df.columns

def test_deduplicate_transactions_empty_dataframe():
    df = pd.DataFrame(columns=['user_id', 'original_description', 'cleaned_description', 'amount', 'transaction_date'])
    deduplicated_df = deduplicate_transactions(df.copy())
    assert deduplicated_df.empty

def test_clean_transaction_data_malformed_amount_and_date():
    data = {
        'original_description': ["Valid", "Invalid Date"],
        'amount': [10.00, 20.50],
        'transaction_date': ["01/02/2023", "not-a-date"]
    }
    df = pd.DataFrame(data)
    cleaned_df = clean_transaction_data(df.copy())

    assert cleaned_df['transaction_date'].iloc[0] == datetime(2023, 1, 2)
    assert pd.isna(cleaned_df['transaction_date'].iloc[1])

# New tests for increased coverage
def test_clean_transaction_data_missing_amount_column():
    data = {
        'original_description': ["Test"],
        'transaction_date': ["2023-01-01"]
    }
    df = pd.DataFrame(data)
    cleaned_df = clean_transaction_data(df.copy())
    assert 'amount' in cleaned_df.columns
    assert pd.isna(cleaned_df['amount'].iloc[0])
    assert pd.api.types.is_float_dtype(cleaned_df['amount'])

def test_clean_transaction_data_missing_transaction_date_column():
    data = {
        'original_description': ["Test"],
        'amount': [10.00]
    }
    df = pd.DataFrame(data)
    cleaned_df = clean_transaction_data(df.copy())
    assert 'transaction_date' in cleaned_df.columns
    assert pd.isna(cleaned_df['transaction_date'].iloc[0]) # Corrected from pd.isnat to pd.isna
    assert pd.api.types.is_datetime64_any_dtype(cleaned_df['transaction_date'])

def test_deduplicate_transactions_no_key_fields():
    data = {
        'other_column': [1, 2, 3],
        'another_column': ['A', 'B', 'C']
    }
    df = pd.DataFrame(data)
    deduplicated_df = deduplicate_transactions(df.copy())
    pd.testing.assert_frame_equal(deduplicated_df, df)
