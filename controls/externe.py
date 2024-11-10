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
def verify_date_format(df_pd: pd.DataFrame, col: str, date_format: str) -> pd.DataFrame:
    """
    Identifies columns in the DataFrame that mostly match one of the common date formats and
    returns a DataFrame with the percentage of values matching the format.
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
        return None

    # Get the regex pattern for the specified date format
    date_regex = date_patterns[date_format]
    
    # List to store columns and match percentages
    results = []

    # Calculate the proportion of non-null values that match the date format
    match_ratio = df_pd[col].dropna().apply(lambda x: bool(date_regex.match(str(x)))).mean()
    # Append the column and match percentage to the results list
    results.append({'Column': col, 'Match Percentage': round(match_ratio * 100, 2),'Null Values': False})

    match_ratio = df_pd[col].apply(lambda x: bool(date_regex.match(str(x)))).mean()
    # Append the column and match percentage to the results list
    results.append({'Column': col, 'Match Percentage': round(match_ratio * 100, 2),'Null Values': True})


    # Convert results to DataFrame
    match_df = pd.DataFrame(results, columns=['Column', 'Match Percentage','Null Values'])

    return match_df

@st.cache_data
def data_quality_score(df: pd.DataFrame):
    """Calculate and display the data quality score."""
    dqs = 100 - round(df.isnull().mean() * 100, 2).mean()
    return dqs

@st.cache_data
def detect_outliers_by_sector(df: pd.DataFrame, sector_column: str, selected_sector: str, numeric_column: str) -> pd.DataFrame:
    """
    Detects outliers in a specified numeric column by sector and returns a DataFrame 
    with unique outlier values and their occurrences.
    """
    
    # Filter the DataFrame for the selected sector
    df_sector = df[df[sector_column] == selected_sector]
    col_data = df_sector[numeric_column]

    # Compute Q1, Q3, and IQR for outlier detection
    Q1 = col_data.quantile(0.25)
    Q3 = col_data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Identify outliers
    outliers = df_sector[
        (df_sector[numeric_column] < lower_bound) |
        (df_sector[numeric_column] > upper_bound)
    ]

    # Count occurrences of unique outlier values
    if not outliers.empty:
        unique_outliers = (
            outliers[numeric_column]
            .value_counts()
            .reset_index()
            .rename(columns={'index': numeric_column, numeric_column: 'extreme values'})
        )
        return unique_outliers
    else:
        return None

@st.cache_data
def duplicates_columns(df: pd.DataFrame):
    """Identify and display duplicate columns in a DataFrame."""
    duplicates = df.columns[df.columns.duplicated()]
    return duplicates

@st.cache_data
def calculate_deviation_by_country(df: pd.DataFrame, country_column: str, numeric_column: str):  
    """ 
    Calculate the mean for each country and compute deviation from it 
    """
    country_means = df.groupby(country_column)[numeric_column].transform('mean')
    deviations = df[numeric_column] - country_means

    result_df = pd.DataFrame({
        country_column: df[country_column],
        numeric_column: df[numeric_column],
        'Country Mean': country_means,
        'Deviation': deviations
    })

    return result_df

@st.cache_data
def show_distribution(df: pd.DataFrame, numeric_column: str):
    """Plot a histogram of the specified numeric column."""
    fig = px.histogram(df, x=numeric_column, nbins=500, title=f'Distribution of {numeric_column}')
    return fig

@st.cache_data
def esg_data_coverage(df: pd.DataFrame, esg_columns: list) -> pd.Series:
    """
    Calculate the coverage of ESG data in each specified column as a percentage.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing ESG data.
        esg_columns (list): A list of column names related to ESG metrics.
    
    Returns:
        pd.Series: Series with ESG column names as index and coverage percentage as values.
    """
    coverage = {}
    for col in esg_columns:
        coverage[col] = 100 * df[col].notna().mean()  # Calculate non-missing values as a percentage
    return pd.Series(coverage)

def update_frequency(df: pd.DataFrame, date_column):
    """Calculate and display the average update frequency."""
    st.subheader("Update Frequency")
    df[date_column] = pd.to_datetime(df[date_column])
    update_frequency = df[date_column].diff().mean()
    st.write(f"Average Update Frequency: {update_frequency}")
    return update_frequency
