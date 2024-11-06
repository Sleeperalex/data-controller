# app.py

import streamlit as st
import pandas as pd
import os
from collections import Counter
import io

# Import external and internal control functions
from controls.externe import *
from controls.interne import *

# Configure Streamlit page
st.set_page_config(page_title="ESG Data Controller", layout="wide")

@st.cache_data
def load_data(file_path=None, uploaded_file=None) -> pd.DataFrame:
    """Load data from a CSV file or an uploaded file and cache the result."""
    if uploaded_file is not None:
        delimiter = find_csv_delimiter(uploaded_file)
        index_column = detect_index_column(uploaded_file, delimiter)
        df = pd.read_csv(uploaded_file,index_col=index_column, sep=delimiter,low_memory=False)
    elif file_path is not None:
        delimiter = find_csv_delimiter(file_path)
        index_column = detect_index_column(file_path, delimiter)
        df = pd.read_csv(file_path,index_col=index_column, sep=delimiter,low_memory=False)
    return df

def find_csv_delimiter(data_file) -> str:
    """Detect the delimiter of a CSV file, handling both file paths and file-like objects."""
    possible_delimiters = [',', ';', '\t', '|', ':']
    # Read the first few lines of the file to detect the delimiter
    if isinstance(data_file, io.BytesIO) or isinstance(data_file, io.TextIOWrapper):  # For uploaded file (file-like object)
        sample = data_file.read(1024).decode('utf-8')  # Read and decode first 1024 bytes
        data_file.seek(0)  # Reset pointer to start of file for further reading
    else:  # For file path
        with open(data_file, 'r', newline='') as file:
            sample = ''.join([next(file) for _ in range(5)])  # Read first 5 lines as a single string
    # Count occurrences of each delimiter in the sample
    delimiter_counts = Counter({delimiter: sample.count(delimiter) for delimiter in possible_delimiters})
    # Return the delimiter with the maximum count
    return delimiter_counts.most_common(1)[0][0] if delimiter_counts else None  # Most common delimiter

def detect_index_column(data_file, delimiter) -> int:
    """Detect the index column in a CSV file with a known delimiter."""
    if isinstance(data_file, io.BytesIO) or isinstance(data_file, io.TextIOWrapper):  # File-like object
        data_file.seek(0)  # Reset pointer to start
        df = pd.read_csv(data_file, sep=delimiter, nrows=1000)
        data_file.seek(0)  # Reset pointer to start again
    else:  # File path
        df = pd.read_csv(data_file, sep=delimiter, nrows=1000)

    for col_index, col_name in enumerate(df.columns):
        if pd.api.types.is_integer_dtype(df[col_name]) and df[col_name].is_unique and (df[col_name] == range(len(df))).all():
            return col_index
    return None

def load_file_options(dataset_folder):
    """Get available CSV file options from the dataset folder."""
    return [f for f in os.listdir(dataset_folder) if f.endswith('.csv')]

def external_controls_page(df_pd : pd.DataFrame):
    """Display the External Controls page content."""

    st.markdown("<br><br>", True)
    st.markdown("<h2 style='text-align: center;'>External Data Quality Controls</h2>", True)
    st.markdown("<br><br>", True)
    
    # Missing Data
    st.subheader("Missing Data")
    mdp = missing_data_percentage(df_pd)
    col1, col2, col3 = st.columns(3)
    col1.text("Missing Data Percentage")
    col1.write(mdp)
    col2.text("Number of Empty Values")
    col2.write(number_of_empty_values(df_pd))
    col3.text("Bar Chart of Missing Data Percentage")
    col3.bar_chart(mdp)
    st.write("Calculate and display the percentage of missing data in each column.")

    # Verify Date Format
    st.subheader("Verify Date Format")
    date_formats = [
        "YYYY-MM-DD", "DD-MM-YYYY", "MM-DD-YYYY",
        "YYYY/MM/DD", "DD/MM/YYYY", "MM/DD/YYYY",
        "YYYY.MM.DD", "DD.MM.YYYY", "MM-YYYY", "YYYY"
    ]
    control_format = st.selectbox("Select a date format to check:", date_formats)
    control_column = st.selectbox("Select a column to check:", df_pd.columns)
    matched_columns = verify_date_format(df_pd,col=control_column, date_format=control_format)
    if not matched_columns.empty:
        st.write(f"Columns following the format '{control_format}':")
        st.write(matched_columns)
    else:
        st.warning(f"No columns found that match the format '{control_format}'.")

    # Data Quality Score
    st.subheader("Data Quality Score")
    qs=data_quality_score(df_pd)
    st.write(f"Overall Data Quality Score: {qs:.2f}%")
    st.write("The data quality score is one minus the percentage of missing data in each column.")

    # Basic Information for Numeric Columns
    st.subheader("Basic Information")
    col1, col2 = st.columns(2)
    col1.write(df_pd.describe())
    col2.plotly_chart(plot_heatmap(df_pd), theme="streamlit")


    # Detect Outliers by Sector
    st.subheader("Détection des Valeurs Extrêmes par Secteur")
    sector_columns = [col for col in df_pd.columns if df_pd[col].nunique() < len(df_pd) and not pd.api.types.is_numeric_dtype(df_pd[col])]
    sector_column = st.selectbox("Sélectionnez la colonne de secteur", sector_columns, key="sector_column")
    if sector_column:
        sectors = df_pd[sector_column].unique().tolist()
        selected_sector = st.selectbox("Sélectionnez un secteur", sectors, key="selected_sector")
        numeric_columns = df_pd.select_dtypes(include=['float64', 'int64']).columns.tolist()
        if numeric_columns:
            selected_numeric_column = st.selectbox("Sélectionnez une colonne numérique", numeric_columns, key="numeric_column")
            if st.button("Détecter les Valeurs Extrêmes par Secteur"):
                unique_outliers_df = detect_outliers_by_sector(df_pd, sector_column, selected_sector, selected_numeric_column)
                if not unique_outliers_df.empty:
                    st.write(f"Valeurs extrêmes uniques dans le secteur '{selected_sector}' pour la colonne '{selected_numeric_column}':")
                    st.write(unique_outliers_df)
                else:
                    st.write(f"Aucune valeur extrême détectée dans le secteur '{selected_sector}' pour la colonne '{selected_numeric_column}'")
        else:
            st.warning("Aucune colonne numérique disponible pour la détection des valeurs extrêmes.")

def internal_controls_page(df_pd : pd.DataFrame):
    """Display the Internal Controls page content."""
    st.markdown("<br><br>", True)
    st.markdown("<h2 style='text-align: center;'>Internal Data Quality Controls</h2>", True)
    st.markdown("<br><br>", True)

    # Execution Time
    execution_time_computation()

    # Importance Scores
    importance_scores(df_pd)

    # Dataset Health
    dataset_health(df_pd)

def main():
    st.markdown("<h1 style='text-align: center;'>ESG Data Controller</h1>", True)

    # Specify dataset folder
    dataset_folder = 'datasets'
    csv_files = load_file_options(dataset_folder) if os.path.exists(dataset_folder) else []
    
    # File selection
    uploaded_file = st.sidebar.file_uploader("Or Upload a CSV file", type="csv")
    selected_file = st.sidebar.selectbox("Or Select a CSV file from the folder", csv_files)

    # Load data from either the uploaded file or the selected file
    if uploaded_file is not None:
        df_pd = load_data(uploaded_file=uploaded_file)
        st.success("CSV file uploaded successfully!")
    elif selected_file:
        file_path = os.path.join(dataset_folder, selected_file)
        df_pd = load_data(file_path=file_path)
    else:
        st.error("Please upload a file or select one from the folder.")
        return

    # Display dataset preview
    st.write("Dataset Preview:")
    st.write(df_pd.head(100))
    st.write("Dataset shape:", df_pd.shape)

    # Sidebar page navigation
    page = st.sidebar.selectbox("Navigate to:", ["External Controls", "Internal Controls"])

    if page == "External Controls":
        external_controls_page(df_pd)
    elif page == "Internal Controls":
        internal_controls_page(df_pd)

if __name__ == "__main__":
    main()
