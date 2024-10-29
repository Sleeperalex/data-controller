import streamlit as st
import pandas as pd

def numerical_columns_iformation(df: pd.DataFrame):
    """Display numerical columns information."""
    st.subheader("Basic Information")
    st.write(df.describe())
    st.write("Display numerical columns information.")
    return

def missing_data_percentage(df: pd.DataFrame):
    """Calculate and display the percentage of missing data in each column."""
    st.subheader("Missing Data Percentage")
    missing_data_percent = round(df.isnull().mean() * 100, 2)
    col1, col2 = st.columns(2)
    col1.write(missing_data_percent)
    col2.bar_chart(missing_data_percent)
    st.write("Calculate and display the percentage of missing data in each column.")
    return missing_data_percent

def number_of_empty_values(df: pd.DataFrame):
    """Calculate and display the number of empty values in each column."""
    st.subheader("Number of Empty Values")
    empty_values = df.isnull().sum()
    st.write(empty_values)
    st.write("Calculate and display the number of empty values in each column.")
    return empty_values

def data_variation(df: pd.DataFrame, date_column, metric_column):
    """Calculate and display data variation (WoW, MoM, YoY)."""
    st.subheader("Data Variation")
    df[date_column] = pd.to_datetime(df[date_column])
    df_sorted = df.sort_values(by=date_column)

    df_sorted.set_index(date_column, inplace=True)
    df_resampled = df_sorted[metric_column].resample('W').sum()

    df_variation = df_resampled.pct_change() * 100
    st.line_chart(df_variation)
    return df_variation

def data_quality_score(missing_data_percent):
    """Calculate and display the data quality score."""
    st.subheader("Data Quality Score")
    data_quality_score = 100 - missing_data_percent.mean()
    st.write(f"Overall Data Quality Score: {data_quality_score:.2f}%")
    st.write("The data quality score is one minus the percentage of missing data in each column.")
    return data_quality_score


def detect_outliers_by_sector(df: pd.DataFrame):
    """Detect outliers in a selected numeric column by sector."""
    st.subheader("Détection des Valeurs Extrêmes par Secteur")
    
    # Let the user select the sector column
    sector_columns = df.columns.tolist()
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
                    st.write(f"Valeurs extrêmes dans le secteur '{selected_sector}' pour la colonne '{selected_numeric_column}':")
                    st.write(outliers[[sector_column, selected_numeric_column]])
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
