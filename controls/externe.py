# controls/externe.py

import streamlit as st
import pandas as pd
import re
import plotly.express as px

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

def missing_data_percentage(df: pd.DataFrame):
    """Calculate and display the percentage of missing data in each column."""
    return round(df.isnull().mean() * 100, 2)

def number_of_empty_values(df: pd.DataFrame):
    """Calculate and display the number of empty values in each column."""
    return df.isnull().sum()

def data_variation(df_pd: pd.DataFrame):
    st.subheader("Data Variation")
    # Regex patterns for various date formats
    date_regex = re.compile(
        r'^\d{4}[-/\.]\d{2}[-/\.]\d{2}$'         # Matches YYYY-MM-DD, YYYY/MM/DD, YYYY.MM.DD
        r'|^\d{2}[-/\.]\d{2}[-/\.]\d{4}$'        # Matches DD-MM-YYYY, DD/MM/YYYY, DD.MM.YYYY
        r'|^\d{2}[-/\.]\d{2}[-/\.]\d{4}$'        # Matches MM-DD-YYYY, MM/DD/YYYY, MM.DD.YYYY
        r'|^\d{4}[-/\.]\d{2}$'                   # Matches YYYY-MM, YYYY/MM, YYYY.MM
        r'|^\d{2}[-/\.]\d{4}$'                   # Matches MM-YYYY, MM/YYYY, MM.YYYY
        r'|^\d{4}$'                              # Matches YYYY
    )

    # Set the percentage threshold for date-like values in a column (e.g., 50%)
    MIN_DATE_MATCH_PERCENTAGE = 0.5

    # Identify potential date columns using regex, excluding columns with "id" in the name
    date_columns = [
        col for col in df_pd.columns
        if (df_pd[col].apply(lambda x: bool(date_regex.match(str(x)))).mean() >= MIN_DATE_MATCH_PERCENTAGE) 
        and (("id" and "unnamed") not in col.lower())
    ]

    # Identify numeric columns
    numeric_columns = df_pd.select_dtypes(include=['float64', 'int64']).columns.tolist()

    # Display warning if no valid date-like or numeric columns are available
    if not date_columns:
        st.warning("No suitable date-like columns found for variation calculation.")
    elif not numeric_columns:
        st.warning("No numeric columns found for variation calculation.")
    else:
        # Proceed with selection only if both date and numeric columns are available
        date_column = st.selectbox("Select Date Column for Variation Calculation", date_columns)
        metric_column = st.selectbox("Select Metric Column for Variation Calculation", numeric_columns)

        if date_column and metric_column:
            # Ensure metric column is converted to numeric
            df_pd[metric_column] = pd.to_numeric(df_pd[metric_column], errors='coerce')
            
            # Call the data_variation function only if there is valid data in both columns
            if df_pd[date_column].notna().any() and df_pd[metric_column].notna().any():
                    df_pd[date_column] = pd.to_datetime(df_pd[date_column],errors='coerce')
                    df_sorted = df_pd.sort_values(by=date_column)

                    df_sorted.set_index(date_column, inplace=True)
                    df_resampled = df_sorted[metric_column].resample('W').sum()

                    df_variation = df_resampled.pct_change() * 100
                    st.line_chart(df_variation)
            else:
                st.warning("Selected columns do not contain enough valid data for variation calculation.")

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
    sector_column = st.selectbox("Sélectionnez la colonne de secteur", sector_columns)

    if sector_column not in sector_columns:
        st.write("Aucune colonne de secteur disponible pour la détection des valeurs extrêmes.")
        return
    
    if sector_column in df.columns:
        sectors = df[sector_column].unique().tolist()
        selected_sector = st.selectbox("Sélectionnez un secteur", sectors)
        
        # Get numeric columns
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        if numeric_columns:
            selected_numeric_column = st.selectbox("Sélectionnez une colonne numérique", numeric_columns)
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
