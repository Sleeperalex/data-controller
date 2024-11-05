# controls/externe.py

import streamlit as st
import pandas as pd
import re
import plotly.express as px

@st.cache_data
def plot_heatmap(df: pd.DataFrame):
    """Plot a heatmap of the correlation matrix of numeric columns."""
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    if numeric_df.empty:
        raise ValueError("No numeric columns available to plot.")
    corr_matrix = numeric_df.corr()
    fig = px.imshow(corr_matrix, 
                    text_auto=True,  # Show values in the cells
                    color_continuous_scale="YlGnBu",  # Color scale
                    labels=dict(color="Correlation"),
                    title="Correlation Heatmap of Numeric Columns")
    return fig

@st.cache_data
def missing_data_percentage(df: pd.DataFrame):
    """Calculate and display the percentage of missing data in each column."""
    mdp = round(df.isnull().mean() * 100, 2)
    return mdp

@st.cache_data
def number_of_empty_values(df: pd.DataFrame):
    """Calculate and display the number of empty values in each column."""
    nofv = df.isnull().sum()
    return nofv

@st.cache_data
def verify_date_format(df_pd: pd.DataFrame, date_format: str, match_threshold=0.9) -> pd.DataFrame:
    """
    Identifies columns in the DataFrame that mostly match one of the common date formats and
    returns a DataFrame with the percentage of values matching the format.
    
    Parameters:
    - df_pd (pd.DataFrame): The input DataFrame.
    - date_format (str): The date format to verify, e.g., "YYYY-MM-DD".
    - match_threshold (float): The minimum proportion of matching values required to consider a column as valid.
    
    Returns:
    - pd.DataFrame: A DataFrame with columns 'Column' and 'Match Percentage' for columns that match the given date format.
    """
    # Define regex patterns for various date formats
    date_patterns = {
        "YYYY-MM-DD": re.compile(r'^\d{4}-\d{2}-\d{2}$'),
        "DD-MM-YYYY": re.compile(r'^\d{2}-\d{2}-\d{4}$'),
        "MM-DD-YYYY": re.compile(r'^\d{2}-\d{2}-\d{4}$'),
        "YYYY/MM/DD": re.compile(r'^\d{4}/\d{2}/\d{2}$'),
        "DD/MM/YYYY": re.compile(r'^\d{2}/\d{2}/\d{4}$'),
        "MM/DD/YYYY": re.compile(r'^\d{2}/\d{2}/\d{4}$'),
        "YYYY.MM.DD": re.compile(r'^\d{4}\.\d{2}\.\d{2}$'),
        "DD.MM.YYYY": re.compile(r'^\d{2}\.\d{2}\.\d{4}$'),
        "MM-YYYY": re.compile(r'^\d{2}-\d{4}$'),
        "YYYY": re.compile(r'^\d{4}$')
    }
    
    # Check if the specified format is supported
    if date_format not in date_patterns:
        st.error(f"Unsupported date format: {date_format}")
        return pd.DataFrame(columns=['Column', 'Match Percentage'])

    # Get the regex pattern for the specified date format
    date_regex = date_patterns[date_format]
    
    # List to store columns and match percentages
    results = []

    # Check each column
    for col in df_pd.columns:
        # Skip columns likely to be IDs or unnamed columns, or columns without any digits
        if "id" not in col.lower() and "unnamed" not in col.lower() and not any(char.isdigit() for char in col):
            # Calculate the proportion of non-null values that match the date format
            match_ratio = df_pd[col].dropna().apply(lambda x: bool(date_regex.match(str(x)))).mean()
            
            # Only include columns that meet the match threshold
            if match_ratio >= match_threshold:
                results.append({'Column': col, 'Match Percentage': round(match_ratio * 100, 2)})

    # Convert results to DataFrame
    match_df = pd.DataFrame(results, columns=['Column', 'Match Percentage'])

    return match_df

@st.cache_data
def data_quality_score(df: pd.DataFrame):
    """Calculate and display the data quality score."""
    return 100 - round(df.isnull().mean() * 100, 2).mean()

def detect_outliers_by_sector(df: pd.DataFrame):
    """Detect outliers in a selected numeric column by sector."""
    st.subheader("Détection des Valeurs Extrêmes par Secteur")
    
    # Filter for non-numeric columns that have non-unique values
    sector_columns = [
        col for col in df.columns 
        if df[col].nunique() < len(df) and not pd.api.types.is_numeric_dtype(df[col])
    ]
    # Let the user select the sector column
    sector_column = st.selectbox("Sélectionnez la colonne de secteur", sector_columns,key="sector_column")

    if sector_column not in sector_columns:
        st.write("Aucune colonne de secteur disponible pour la détection des valeurs extrêmes.")
        return
    
    if sector_column in df.columns:
        sectors = df[sector_column].unique().tolist()
        selected_sector = st.selectbox("Sélectionnez un secteur", sectors, key="selected_sector")
        
        # Get numeric columns
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        if numeric_columns:
            selected_numeric_column = st.selectbox("Sélectionnez une colonne numérique", numeric_columns, key="numeric_column")
            if st.button("Détecter les Valeurs Extrêmes par Secteur"):
                # Filter the dataframe for the selected sector
                df_sector = df[df[sector_column] == selected_sector]
                col_data = df_sector[selected_numeric_column]
                
                # Compute Q1, Q3, IQR
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                seuil_bas = Q1 - 1.5 * IQR
                seuil_haut = Q3 + 1.5 * IQR
                
                # Identify outliers
                outliers = df_sector[
                    (df_sector[selected_numeric_column] < seuil_bas) |
                    (df_sector[selected_numeric_column] > seuil_haut)
                ]
                
                if not outliers.empty:
                    # Count occurrences of unique outlier values
                    unique_outliers = (
                        outliers[selected_numeric_column]
                        .value_counts()
                        .reset_index()
                        .rename(columns={'index': selected_numeric_column, selected_numeric_column: 'extreme values'})
                    )

                    st.write(f"Valeurs extrêmes uniques dans le secteur '{selected_sector}' pour la colonne '{selected_numeric_column}':")
                    st.write(unique_outliers)
                else:
                    st.write(f"Aucune valeur extrême détectée dans le secteur '{selected_sector}' pour la colonne '{selected_numeric_column}'")
        else:
            st.write("Aucune colonne numérique disponible pour la détection des valeurs extrêmes.")
    else:
        st.write(f"La colonne de secteur '{sector_column}' n'existe pas dans le dataset.")

def update_frequency(df: pd.DataFrame, date_column):
    """Calculate and display the average update frequency."""
    st.subheader("Update Frequency")
    df[date_column] = pd.to_datetime(df[date_column])
    update_frequency = df[date_column].diff().mean()
    st.write(f"Average Update Frequency: {update_frequency}")
    return update_frequency
