# controls/interne.py

import streamlit as st
import pandas as pd
import time,re

@st.cache_data
def esg_data_coverage(df: pd.DataFrame, esg_columns: list):
    """
    Calculate the coverage of ESG data in each specified column as a percentage.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing ESG data.
        esg_columns (list): A list of column names related to ESG metrics.
    
    Returns:
        pd.Series: Series with ESG column names as index and coverage percentage as values.
    """
    coverage = pd.DataFrame(index=["ESG Data Coverage"])
    for col in esg_columns:
        coverage[col] = 100 * df[col].notna().mean()  # Calculate non-missing values as a percentage
    return coverage.T

@st.cache_data
def filter_company_by_score_and_flag(df: pd.DataFrame,required_columns, score_summary, flag_summary) -> pd.DataFrame:
    """
    Filter companies with a SCORE_SUMMARY of 10 and a FLAG_SUMMARY of 'Green'.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing the ESG data.
        
    Returns:
        pd.DataFrame: Filtered DataFrame with companies matching the criteria.
    """
    # Check if required columns exist in the DataFrame
    if not all(col in df.columns for col in required_columns):
        st.error("Required columns not found in the DataFrame.")
   
    # Filter based on the criteria
    filtered_df = df[(df[required_columns[1]] == score_summary) & (df[required_columns[2]] == flag_summary)]
    return filtered_df[required_columns]

@st.cache_data
def check_column_names(df: pd.DataFrame, columns_to_check):
    """
    Vérifie que les noms de colonnes ne contiennent pas d'espaces ou de caractères spéciaux non permis.
    Autorise uniquement les lettres, chiffres et underscores. 
    """
    # Expression régulière pour vérifier que les noms de colonnes sont valides
    valid_pattern = re.compile(r'^[A-Za-z0-9_]+$')

    invalid_columns = []

    # Vérification des noms des colonnes 
    for col in columns_to_check:
        if col not in df.columns:
            print(f"Colonne '{col}' non présente dans le DataFrame.")
            continue
        if not valid_pattern.match(col):
            invalid_columns.append(col)
    
    return invalid_columns

