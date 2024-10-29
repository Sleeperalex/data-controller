import streamlit as st
import pandas as pd
import dask.dataframe as dd
import os

# Import external and internal control functions
from controls.externe import *
from controls.interne import *

def load_data(dataset_folder):
    """Load the data from CSV files in the specified folder."""
    # List CSV files in the folder
    csv_files = [f for f in os.listdir(dataset_folder) if f.endswith('.csv')]
    selected_file = st.sidebar.selectbox("Select a CSV file", csv_files)

    if selected_file:
        file_path = os.path.join(dataset_folder, selected_file)
        # Load dataset using Dask and convert to Pandas DataFrame
        df = dd.read_csv(file_path, sep=';')
        df_pd = df.compute()  # Convert to Pandas for easier processing
        st.write("Dataset Preview:")
        st.write(df_pd.head(5))
        return df_pd
    else:
        st.error("No CSV files found in the dataset folder.")
        return None

def main():
    st.title("ESG Data Controller")

    # Specify the dataset folder
    dataset_folder = 'datasets'

    # Check if the folder exists
    if not os.path.exists(dataset_folder):
        st.error(f"The folder '{dataset_folder}' does not exist. Please create it and place your CSV files inside.")
        return

    # Load the data
    df_pd = load_data(dataset_folder)

    # If data is loaded successfully, display page options
    if df_pd is not None:
        # Sidebar navigation for different pages
        page = st.sidebar.radio("Choose Control Type", ["External Controls", "Internal Controls"])

        # External Controls page
        if page == "External Controls":
            st.header("External Data Quality Controls")
            
            # Missing Data Percentage
            missing_data_percent = missing_data_percentage(df_pd)

            # Number of Empty Values
            number_of_empty_values(df_pd)

            # Data Variation
            st.subheader("Data Variation")
            date_columns = df_pd.select_dtypes(include=['datetime64', 'object']).columns.tolist()
            numeric_columns = df_pd.select_dtypes(include=['float64', 'int64']).columns.tolist()
            date_column = st.selectbox("Select Date Column for Variation Calculation", date_columns)
            metric_column = st.selectbox("Select Metric Column for Variation Calculation", numeric_columns)
            if date_column and metric_column:
                data_variation(df_pd, date_column, metric_column)

            # Data Quality Score
            data_quality_score(missing_data_percent)

            # Basic Information for Numeric Columns
            numerical_columns_iformation(df_pd)

            # Detect Outliers by Sector
            detect_outliers_by_sector(df_pd)

            # Update Frequency
            #update_frequency(df_pd, date_column)

        # Internal Controls page
        elif page == "Internal Controls":
            st.header("Internal Data Quality Controls")

            # Execution Time
            execution_time_computation()

            # Importance Scores
            importance_scores(df_pd)

            # Dataset Health
            dataset_health(df_pd)

if __name__ == "__main__":
    main()
