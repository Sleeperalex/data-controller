# app.py

import streamlit as st
import pandas as pd
import os
from collections import Counter
import io

# Import external and internal control functions
from controls.externe import *
from controls.interne import *
from controls.personalize import *

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

def external_controls_page(df_pd: pd.DataFrame, selected_function : str):
    """Display the External Controls page content with subpages."""

    # Tab 1: Missing Data
    if selected_function == "Missing Data":
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

    # Tab 2: Verify Date Format
    if selected_function == "Verify Date Format":
        st.subheader("Verify Date Format")
        date_formats = ["YYYY-MM-DD", "DD-MM-YYYY", "MM-DD-YYYY", "YYYY/MM/DD", "DD/MM/YYYY", "MM/DD/YYYY", "YYYY.MM.DD", "DD.MM.YYYY", "MM-YYYY", "YYYY"]
        control_format = st.selectbox("Select a date format to check:", date_formats)
        control_column = st.selectbox("Select a column to check:", df_pd.columns)
        matched_columns = verify_date_format(df_pd, col=control_column, date_format=control_format)
        if not matched_columns.empty:
            st.write(f"Columns following the format '{control_format}':")
            st.write(matched_columns)
        else:
            st.warning(f"No columns found that match the format '{control_format}'.")

    # Tab 3: Data Quality Score
    if selected_function == "Data Quality Score":
        st.subheader("Data Quality Score")
        qs = data_quality_score(df_pd)
        st.write(f"Overall Data Quality Score: {qs:.2f}%")
        st.write("The data quality score is one minus the percentage of missing data in each column.")

    # Tab 4: Basic Information for Numeric Columns
    if selected_function == "Basic Information for Numeric Columns":
        st.subheader("Basic Information")
        col1, col2 = st.columns(2)
        col1.write(df_pd.describe())
        col2.plotly_chart(plot_heatmap(df_pd), theme="streamlit")

    # Tab 5: Detect Outliers by Sector
    if selected_function == "Detect Outliers":
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
                    if unique_outliers_df is not None:
                        st.write(f"Valeurs extrêmes uniques dans le secteur '{selected_sector}' pour la colonne '{selected_numeric_column}':")
                        st.write(unique_outliers_df)
                    else:
                        st.write(f"Aucune valeur extrême détectée dans le secteur '{selected_sector}' pour la colonne '{selected_numeric_column}'")
            else:
                st.warning("Aucune colonne numérique disponible pour la détection des valeurs extrêmes.")

    # Tab 6: Duplicates Columns
    if selected_function == "Duplicates Columns":
        st.subheader("Duplicates columns")
        dc = duplicates_columns(df_pd)
        if dc.empty == False:
            st.write(f"Duplicate columns:")
            st.write(dc)
        else:
            st.write("No duplicate columns found.")

    # Tab 7: deviation by country
    if selected_function == "Deviation by Country":
        st.subheader("Deviation by Country")
        country_column = st.selectbox("Select a country column:", df_pd.columns)
        numeric_column = st.selectbox("Select a numeric column:", df_pd.select_dtypes(include=['int64', 'float64']).columns)
        deviations = calculate_deviation_by_country(df_pd, country_column, numeric_column)
        st.write(f"Deviations from the mean by country for {numeric_column}:")
        st.write(deviations)

    # Tab 8: Show Distribution
    if selected_function == "Show Distribution":
        st.subheader("Show Distribution")
        numeric_column = st.selectbox("Select a numeric column:", df_pd.select_dtypes(include=['int64', 'float64']).columns)
        fig = px.histogram(df_pd, x=numeric_column, nbins=500, title=f'Distribution of {numeric_column}')
        st.plotly_chart(fig, theme="streamlit")

def internal_controls_page(df_pd: pd.DataFrame, selected_function: str):
    """Display the Internal Controls page content with subpages."""
    # Tab 1: ESG Data Coverage
    if selected_function == "ESG Data Coverage":
        st.subheader("ESG Data Coverage")
        # let the user select the ESG columns they want to check
        esg_columns = st.multiselect("Select ESG columns to check:", df_pd.columns)
        esg_coverage = esg_data_coverage(df_pd, esg_columns)
        fig = px.histogram(esg_coverage, x=esg_coverage["ESG Data Coverage"], nbins=100, title=f'Distribution of ESG Data Coverage for selected columns')
        st.plotly_chart(fig, theme="streamlit")
        st.write(f"ESG Data Coverage:")
        st.write(esg_coverage)

    # Tab 2: filter company by score and flag
    if selected_function == "Filter Company by Score and Flag":
        st.subheader("Filter Company by Score and Flag")
        required_columns = ["COMPANY_NAME_MNS", "SCORE_SUMMARY", "FLAG_SUMMARY"]
        if all(col in df_pd.columns for col in required_columns):
            # let the user write the score and flag they want to filter by
            score_summary = st.text_input("Enter the score summary to filter by:")
            flag_summary = st.text_input("Enter the flag summary to filter by:")
            if not score_summary.isdigit() or not flag_summary.isalpha():
                st.error("Please enter valid score and flag summaries.")
            else:
                filtered_df = filter_company_by_score_and_flag(df_pd,required_columns, int(score_summary), flag_summary)
                st.write(filtered_df)
        else:
            st.error("Required columns not found in the DataFrame.")

    # Tab 3: Check Column Names
    if selected_function == "Check Column Names":
        st.subheader("Check Column Names")
        columns_to_check = st.multiselect("Select columns to check:", df_pd.columns)
        invalid_columns = check_column_names(df_pd, columns_to_check)
        if len(invalid_columns) > 0:
            st.write(f"Invalid column names: {', '.join(invalid_columns)}")
        else:
            st.write("All column names are valid.")

def cleaning_page(df_pd: pd.DataFrame):
    """Display the Cleaning page content."""
    st.write("This feature is under development.")

def machine_learning_page(df_pd: pd.DataFrame):
    """Display the Machine Learning page content."""
    st.write("This feature is under development.")

def personalize_controls_page(df_pd: pd.DataFrame, selected_function: str):
    """Display the Personalize Controls page content."""
    if selected_function == "custom regex":
        st.subheader("Custom Regex")
        regex = st.text_input("Enter the regex pattern:")
        col = st.selectbox("select the column",df_pd.columns)
        if check_regex(df_pd,regex,col):
            st.write("regex is valid for values in column ",col)
        else:
            st.write("regex invalid for values in column ",col)


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

    # Sidebar page navigation
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["External Controls", "Internal Controls", "Cleaning", "Machine Learning", "Personalize Controls"])

    with tab1:
        # Display dataset preview
        st.write("Dataset Preview:")
        st.dataframe(df_pd.head(100),height=300)
        st.write("Dataset shape:", df_pd.shape)

        st.markdown("<h1 style='text-align: center;'>External Controls</h1>", True)
        st.subheader("External Controls Settings")  # Sidebar options specific to External Controls
        selected_function = st.selectbox("Choose function for External Controls", [
            "Missing Data",
            "Verify Date Format",
            "Data Quality Score",
            "Basic Information for Numeric Columns",
            "Detect Outliers",
            "Duplicates Columns",
            "Deviation by Country",
            "Show Distribution"
        ])
        external_controls_page(df_pd, selected_function)
    with tab2:
        # Display dataset preview
        st.write("Dataset Preview:")
        st.dataframe(df_pd.head(100),height=300)
        st.write("Dataset shape:", df_pd.shape)

        st.markdown("<h1 style='text-align: center;'>Internal Controls</h1>", True)
        st.subheader("Internal Controls Settings")  # Sidebar options specific to Internal Controls
        selected_function = st.selectbox("Choose function for Internal Controls", [
            "ESG Data Coverage",
            "Filter Company by Score and Flag",
            "Check Column Names"
        ])
        internal_controls_page(df_pd, selected_function)
    with tab3:
        st.markdown("<h1 style='text-align: center;'>Cleaning</h1>", True)
        cleaning_page(df_pd)
    with tab4:
        st.markdown("<h1 style='text-align: center;'>Machine Learning</h1>", True)
        st.write("This feature is under development.")
    with tab5:
        st.markdown("<h1 style='text-align: center;'>Personalize Controls</h1>", True)
        selected_function = st.selectbox("Choose function for Personalize Controls", [
            "custom regex"
        ])
        personalize_controls_page(df_pd, selected_function)


if __name__ == "__main__":
    main()
