# app.py

import streamlit as st
import pandas as pd
import os

# Import external and internal control functions
from controls.externe import *
from controls.interne import *

# Configure Streamlit page
st.set_page_config(page_title="ESG Data Controller", layout="wide")

@st.cache_data
def load_data(file_path=None, uploaded_file=None) -> pd.DataFrame:
    """Load data from a CSV file or an uploaded file and cache the result."""
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file,index_col=0, sep=";")
    else:
        df = pd.read_csv(file_path,index_col=0, sep=";")
    return df

def load_file_options(dataset_folder):
    """Get available CSV file options from the dataset folder."""
    return [f for f in os.listdir(dataset_folder) if f.endswith('.csv')]

def external_controls_page(df_pd: pd.DataFrame):
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

    # Data Variation
    #data_variation(df_pd)

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
    detect_outliers_by_sector(df_pd)

def internal_controls_page(df_pd):
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
