# controls/interne.py

import streamlit as st
import pandas as pd
import time

def execution_time_computation():
    """Measure and display execution time of computations."""
    start_time = time.time()
    time.sleep(3)
    execution_time = time.time() - start_time
    return execution_time

def filter_company_by_score_and_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter companies with a SCORE_SUMMARY of 10 and a FLAG_SUMMARY of 'Green'.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing the ESG data.
        
    Returns:
        pd.DataFrame: Filtered DataFrame with companies matching the criteria.
    """
    # Check if necessary columns are in the DataFrame
    required_columns = ["COMPANY_NAME_MNS", "SCORE_SUMMARY", "FLAG_SUMMARY"]
   
    # Filter based on the criteria
    filtered_df = df[(df["SCORE_SUMMARY"] == 10) & (df["FLAG_SUMMARY"] == "Green")]
    return filtered_df[["COMPANY_NAME_MNS", "SCORE_SUMMARY", "FLAG_SUMMARY"]]
