import pandas as pd
import streamlit as st

@st.cache_data
def fill_missing_values(data : pd.DataFrame):
    # Parcourir chaque colonne du DataFrame
    for col in data.columns:
        # Vérifier si la colonne est numérique
        if data[col].dtype in ['int64', 'float64']:
            # Remplir les valeurs manquantes avec la moyenne pour les colonnes numériques
            data[col] = data[col].fillna(data[col].mean())
        # Vérifier si la colonne est de type texte (objet)
        elif data[col].dtype == 'object':
            # Remplir les valeurs manquantes avec 'Unknown' pour les colonnes de texte
            data[col] = data[col].fillna('Unknown')
    return data

@st.cache_data
def remove_duplicates(data : pd.DataFrame):
    df=data.drop_duplicates()
    return df

@st.cache_data
def convert_to_datetime(data : pd.DataFrame, col : str):
    # Conversion des dates (format YYYYMMDD)
    data[col] = pd.to_datetime(data[col], format='%Y%m%d', errors='coerce')
    return data