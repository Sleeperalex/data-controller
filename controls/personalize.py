import streamlit as st
import pandas as pd

@st.cache_data
def check_regex(df : pd.DataFrame, regex_pattern : str, column_name: str):
    if column_name in df.columns:
        if df[column_name].dtype not in ['int64', 'float64']:
            if not df[column_name].str.match(regex_pattern).all():
                return False
    return True

@st.cache_data
def check_column_regex(df: pd.DataFrame, regex_pattern :str):
    """
    Vérifie que les noms de colonnes ne contiennent pas d'espaces ou de caractères spéciaux non permis.
    Autorise uniquement les lettres, chiffres et underscores. 
    """
    # Expression régulière pour vérifier que les noms de colonnes sont valides
    valid_pattern = regex_pattern

    invalid_columns = []

    # Vérification des noms des colonnes 
    for col in df.columns:
        # check if the column is numeric
            if not valid_pattern.match(str(col)):
                invalid_columns.append(str(col))
    
    return invalid_columns