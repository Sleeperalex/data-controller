# app.py

import streamlit as st
import pandas as pd
import os
from collections import Counter
import io
from pygwalker.api.streamlit import StreamlitRenderer
from transformers import pipeline

# Import for machine_learning_page
import spacy
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from transformers import pipeline, AutoTokenizer
import torch
from collections import OrderedDict
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import nltk
from nltk.tokenize import word_tokenize
from textblob import TextBlob

# Import external and internal control functions
from controls.externe import *
from controls.interne import *
from controls.personalize import *
from MA.machine_learning import *

# Import cleaning functions
from clean.clean import *

# Import modules for Financial Analysis Tool
from other_project import fta

# Configure Streamlit page
st.set_page_config(page_title="Data Analysis Tool", layout="wide")

@st.cache_resource
def get_pyg_renderer(df) -> StreamlitRenderer:
    return StreamlitRenderer(df,kernel_computation=True)

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

def load_file_options(folder_path):
    """Get available CSV file options from the dataset folder."""
    return [f for f in os.listdir(folder_path) if f.endswith('.csv')]

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

    # Tab 9: Apply Custom Threshold
    if selected_function == "Apply Custom Threshold":
        st.subheader("Apply Custom Threshold")
        column = st.selectbox("Select a numeric column:", df_pd.select_dtypes(include=['int64', 'float64']).columns, key="threshold_column")
        operator = st.selectbox("Select Operator:", ["<", ">", "<=", ">=", "="], key="threshold_operator")
        threshold = st.number_input(f"Enter threshold for {column}:", key="threshold_value")

        if st.button("Apply Threshold"):
            filtered_df = apply_custom_threshold(df_pd, column, threshold, operator)
            st.write(f"Filtered Data: `{column} {operator} {threshold}`")
            st.dataframe(filtered_df, height=400)  # Optimize Data Display

def internal_controls_page(df_pd: pd.DataFrame, selected_function: str):
    """Display the Internal Controls page content with subpages."""

    # Tab 1: filter company by score and flag
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

    # Tab 2: Check Column Names
    if selected_function == "Check Column Names":
        st.subheader("Check Column Names")
        invalid_columns = check_column_names(df_pd)
        if len(invalid_columns) > 0:
            st.write(f"Invalid column names: {', '.join(invalid_columns)}")
        else:
            st.write("All column names are valid.")

    # Tab 3: Check dimensions of datasets
    if selected_function == "Check dimensions of datasets":
        st.subheader("Check dimensions of datasets")
        st.write("This feature is under development.")

def cleaning_page(df_pd: pd.DataFrame):
    """Display the Cleaning page content."""
    st.subheader("Data Cleaning Operations")

    # Create checkboxes for cleaning options
    rd = st.checkbox("Remove Duplicates Rows")
    col1, col2 = st.columns(2)
    cd = col1.checkbox("Convert Date to Datetime")
    col = col2.selectbox("Select a column to convert to datetime", df_pd.columns)

    # Create an "Apply" button to execute selected operations
    if st.button("Apply Cleaning"):
        modified_df = df_pd.copy()  # Ensure the original DataFrame is not altered

        if rd:
            modified_df = remove_duplicates(modified_df)
            st.write("Duplicates removed successfully.")

        if cd:
            modified_df = convert_to_datetime(modified_df,col)
            st.write("Date converted to datetime successfully.")


        st.write("Cleaned Data:")
        st.write(modified_df)

# Définition du modèle ESGify
class ESGify(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.classifier = torch.nn.Sequential(
            OrderedDict([
                ('linear', torch.nn.Linear(768, 47)),
                ('act', torch.nn.ReLU()),
                ('drop', torch.nn.Dropout(0.2))
            ])
        )
    def forward(self, input_ids, attention_mask):
        logits = self.classifier(torch.randn(len(input_ids), 768))
        logits = torch.sigmoid(logits)
        return logits

# Charger le modèle ESGify
model = ESGify(None)
tokenizer = AutoTokenizer.from_pretrained('ai-lab/ESGify')

nltk.download('punkt')
nlp = spacy.load("en_core_web_sm")

def machine_learning_page(df_pd: pd.DataFrame):
    """Display the Machine Learning page content."""
    st.write("This feature is under development.")

    # PDF extraction
    st.subheader("PDF Text Extraction")
    uploaded_pdf = st.file_uploader("Upload PDF File", type="pdf")

    if uploaded_pdf is not None:
        # Extract text from the uploaded PDF
        extracted_text = extract_pdf_text(uploaded_pdf)
        ner_pipeline = pipeline("ner", grouped_entities=True)
        text = extracted_text
        entities = ner_pipeline(text)
        df_entities = pd.DataFrame(entities)
        st.dataframe(df_entities)

    # Option to upload a PDF from local file path
    file_path = st.text_input("Enter file path for local PDF extraction")
    if file_path:
        if os.path.exists(file_path) and file_path.endswith(".pdf"):
            extracted_text_local = extract_pdf_from_local(file_path)
            st.text_area("Extracted Text from Local File", extracted_text_local, height=300)
        else:
            st.error("Invalid file path or file is not a PDF.")

    if uploaded_pdf is not None:
        st.subheader("📜 Texte extrait")
        st.text_area("Contenu du fichier", text[:1000] + "...", height=250)

        # Extraction des mots-clés
        def extract_keywords(text):
            doc = nlp(text)
            return [token.text for token in doc if token.is_alpha and not token.is_stop]

        keywords = extract_keywords(text)
        st.subheader("🔑 Mots-clés")
        st.write(", ".join(keywords[:50]))

        # Génération du WordCloud
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(keywords))
        st.subheader("🌥️ Nuage de mots-clés")
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)

        # Résumé du texte avec T5
        summarizer = pipeline("summarization", model="t5-small")
        summary = summarizer(text[:1024], max_length=150, min_length=30, do_sample=False)[0]['summary_text']
        st.subheader("📌 Résumé du rapport")
        st.write(summary)

        # Analyse de sentiment
        sentiment = TextBlob(text).sentiment
        st.subheader("📊 Analyse de sentiment")
        st.write(f"Score de sentiment : {sentiment.polarity}")

        # Classification ESG
        tokens = tokenizer(text[:512], return_tensors='pt', truncation=True, padding="max_length", max_length=512)
        results = model(**tokens)
        top_indices = torch.topk(results[0], k=3).indices.tolist()
        st.subheader("🏢 Classification ESG")
        st.write(f"Top catégories ESG : {top_indices}")

def personalize_controls_page(df_pd: pd.DataFrame, selected_function: str):
    """Display the Personalize Controls page content."""
    if selected_function == "custom regex values":
        st.subheader("Custom Regex")
        regex = st.text_input("Enter the regex pattern:")
        col = st.selectbox("select the column",df_pd.columns)
        if check_regex(df_pd,regex,col):
            st.write("regex is valid for values in column ",col)
        else:
            st.write("regex invalid for values in column ",col)

    if selected_function == "custom regex column":
        st.subheader("Custom Regex")
        regex = st.text_input("Enter the regex pattern:")
        col = st.selectbox("select the column",df_pd.columns)
        if check_regex(df_pd,regex,col):
            st.write("regex is valid for column name ",col)
        else:
            st.write("regex invalid for column name ",col)


def esg_data_controller():
    st.markdown("<h1 style='text-align: center;'>ESG Data Controller</h1>", True)

    # Specify dataset folder
    csv_files = load_file_options('datasets') if os.path.exists('datasets') else []

    # File selection
    uploaded_file = st.sidebar.file_uploader("Or Upload a CSV file", type="csv")
    selected_file = st.sidebar.selectbox("Or Select a CSV file from the folder", csv_files)

    # Load data from either the uploaded file or the selected file
    if uploaded_file is not None:
        df_pd = load_data(uploaded_file=uploaded_file)
        st.success("CSV file uploaded successfully!")
    elif selected_file:
        file_path = os.path.join('datasets', selected_file)
        df_pd = load_data(file_path=file_path)
    else:
        st.error("Please upload a file or select one from the folder.")
        return

    renderer = get_pyg_renderer(df_pd)

    # Sidebar page navigation
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Data Summary","External Controls", "Internal Controls", "Cleaning", "Machine Learning", "Personalize Controls"])

    with tab1:
        renderer.explorer()
    with tab2:
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
            "Show Distribution",
            "Apply Custom Threshold"
        ])
        external_controls_page(df_pd, selected_function)
    with tab3:
        # Display dataset preview
        st.write("Dataset Preview:")
        st.dataframe(df_pd.head(100),height=300)
        st.write("Dataset shape:", df_pd.shape)

        st.markdown("<h1 style='text-align: center;'>Internal Controls</h1>", True)
        st.subheader("Internal Controls Settings")  # Sidebar options specific to Internal Controls
        selected_function = st.selectbox("Choose function for Internal Controls", [
            "Filter Company by Score and Flag",
            "Check Column Names",
            "Check dimensions of datasets"
        ])
        internal_controls_page(df_pd, selected_function)
    with tab4:
        st.markdown("<h1 style='text-align: center;'>Cleaning</h1>", True)
        cleaning_page(df_pd)
    with tab5:
        st.markdown("<h1 style='text-align: center;'>Machine Learning</h1>", True)
        machine_learning_page(df_pd)
    with tab6:
        st.markdown("<h1 style='text-align: center;'>Personalize Controls</h1>", True)
        selected_function = st.selectbox("Choose function for Personalize Controls", [
            "custom regex values",
            "custom regex column"
        ])
        personalize_controls_page(df_pd, selected_function)



def financial_analysis_tool():
    """Financial Analysis functionalities."""
    st.markdown("<h1 style='text-align: center;'>Financial Analysis Tool</h1>", True)
    fta.main()

def main():
    # Sidebar: Project Selection
    project_choice = st.sidebar.radio("Select a Project:", ["ESG Data Controller", "Financial Analysis Tool"])
    if project_choice == "ESG Data Controller":
        esg_data_controller()
    elif project_choice == "Financial Analysis Tool":
        financial_analysis_tool()

if __name__ == "__main__":
    st.write("Select a project from the sidebar.")
    main()
