# app.py

import streamlit as st
import pandas as pd
import os
from collections import Counter

# Import external and internal control functions
from controls.externe import *
from controls.interne import *

# Configure Streamlit page
st.set_page_config(page_title="ESG Data Controller", layout="wide")

@st.cache_data
def load_data(file_path=None, uploaded_file=None) -> pd.DataFrame:
    """Load data from a CSV file or an uploaded file and cache the result."""
    delimiter = find_csv_delimiter(file_path)
    index_column = detect_index_column(file_path)
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file,index_col=index_column, sep=delimiter)
    else:
        df = pd.read_csv(file_path,index_col=index_column, sep=delimiter)
    return df

def detect_index_column(file_path):
    """
    Detects the index column in a CSV file.
    """
    # Read a sample of the CSV to detect potential index column
    df = pd.read_csv(file_path, sep=find_csv_delimiter(file_path))
    
    for col_index, col_name in enumerate(df.columns):
        # Check if the column is integer type, has unique values, and matches 1 to len(df)
        if pd.api.types.is_integer_dtype(df[col_name]):
            if df[col_name].is_unique and (df[col_name] == range(0, len(df))).all():
                return col_index
    return None

def find_csv_delimiter(data_file):
    """Detect the delimiter of a CSV file."""
    # Define possible delimiters
    possible_delimiters = [',', ';', '\t', '|']
    # Read the first few lines of the file to detect the delimiter
    with open(data_file, 'r', newline='') as file:
        sample_lines = [next(file) for _ in range(5)]  # Read first 5 lines
    # Count occurrences of each delimiter
    delimiter_counts = Counter()
    for line in sample_lines:
        for delimiter in possible_delimiters:
            delimiter_counts[delimiter] += line.count(delimiter)
    # Find the delimiter with the maximum count
    if delimiter_counts:
        return delimiter_counts.most_common(1)[0][0]  # Return the most common delimiter
    return None  # Return None if no delimiter found

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
