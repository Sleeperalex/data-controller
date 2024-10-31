# app.py

import streamlit as st
import pandas as pd
import dask.dataframe as dd
import os, re

# Import external and internal control functions
from controls.externe import *
from controls.interne import *

# Configure Streamlit page
st.set_page_config(page_title="ESG Data Controller", layout="wide")

@st.cache_data
def load_data(file_path=None, uploaded_file=None) -> pd.DataFrame:
    """Load data from a CSV file or an uploaded file and cache the result."""
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, sep=';')
    else:
        df = dd.read_csv(file_path, sep=';').compute()  # Load from file path and convert to Pandas
    return df

def load_file_options(dataset_folder):
    """Get available CSV file options from the dataset folder."""
    return [f for f in os.listdir(dataset_folder) if f.endswith('.csv')]

def external_controls_page(df_pd):
    """Display the External Controls page content."""
    st.header("External Data Quality Controls")
    
    # Missing Data Percentage
    missing_data_percent = missing_data_percentage(df_pd)

    # Number of Empty Values
    number_of_empty_values(df_pd)


    data_variation(df_pd)

    # Data Quality Score
    data_quality_score(missing_data_percent)

    # Basic Information for Numeric Columns
    numerical_columns_iformation(df_pd)

    # Detect Outliers by Sector
    detect_outliers_by_sector(df_pd)

def internal_controls_page(df_pd):
    """Display the Internal Controls page content."""
    st.header("Internal Data Quality Controls")

    # Execution Time
    execution_time_computation()

    # Importance Scores
    importance_scores(df_pd)

    # Dataset Health
    dataset_health(df_pd)

def main():
    st.title("ESG Data Controller")

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
    st.write(df_pd.head(5))
    st.write("Dataset shape:", df_pd.shape)

    # Sidebar page navigation
    page = st.sidebar.selectbox("Navigate to:", ["External Controls", "Internal Controls"])

    if page == "External Controls":
        external_controls_page(df_pd)
    elif page == "Internal Controls":
        internal_controls_page(df_pd)

if __name__ == "__main__":
    main()
