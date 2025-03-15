# controls/interne.py

import streamlit as st
import pandas as pd
import time,re,os


@st.cache_data
def filter_company_by_score_and_flag(df: pd.DataFrame, required_columns, score_summary, flag_summary: str) -> pd.DataFrame:
    """
    Filter companies with a specified SCORE_SUMMARY and FLAG_SUMMARY.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing the ESG data.
        required_columns (list): List of required columns, where:
            - required_columns[0] = Company Name/ID
            - required_columns[1] = SCORE_SUMMARY
            - required_columns[2] = FLAG_SUMMARY
        score_summary (int/float): The score value to filter on.
        flag_summary (str): The flag value (e.g., "Red", "Green", etc.), case-insensitive.
        
    Returns:
        pd.DataFrame: Filtered DataFrame with companies matching the criteria.
    """

    # Check if required columns exist in the DataFrame
    if not all(col in df.columns for col in required_columns):
        return pd.DataFrame(columns=required_columns)  # Return empty DataFrame if columns are missing

    # Convert flag_summary column & user input to lowercase for case-insensitive comparison
    df_filtered = df[df[required_columns[2]].str.lower() == flag_summary.lower()]

    # Apply score filter
    df_filtered = df_filtered[df_filtered[required_columns[1]] == score_summary]

    return df_filtered[required_columns]


@st.cache_data
def check_column_names(df: pd.DataFrame):
    """
    Vérifie que les noms de colonnes ne contiennent pas d'espaces ou de caractères spéciaux non permis.
    Autorise uniquement les lettres, chiffres et underscores. 
    """
    # Expression régulière pour vérifier que les noms de colonnes sont valides
    valid_pattern = re.compile(r'^[A-Za-z0-9_]+$')

    invalid_columns = []

    # Vérification des noms des colonnes 
    for col in df.columns:
        # check if the column is numeric
            if not valid_pattern.match(str(col)):
                invalid_columns.append(str(col))
    
    return invalid_columns

