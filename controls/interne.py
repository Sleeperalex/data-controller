# controls/interne.py

import streamlit as st
import pandas as pd
import time,re

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

