# ESG Report to DataFrame Converter with Sentiment Analysis
import streamlit as st
import pandas as pd
import re
import io
import PyPDF2
from docx import Document
from collections import defaultdict
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

def extract_text_from_docx(docx_file):
    doc = Document(docx_file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def extract_text_from_txt(txt_file):
    return txt_file.getvalue().decode('utf-8')

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,;:%()-]', '', text)
    return text.strip()

def split_into_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip() and not re.match(r'^\d+\s+[A-Za-z ]+\.+$', s)]

def identify_esg_categories(text):
    sentences = split_into_sentences(text)
    e_keywords = ['environment', 'climate', 'carbon', 'emission', 'energy', 'water', 'waste', 'recycling', 'biodiversity', 'renewable', 'pollution', 'sustainable']
    s_keywords = ['social', 'employee', 'diversity', 'inclusion', 'health', 'safety', 'community', 'human rights', 'labor', 'training', 'development', 'wellbeing', 'gender']
    g_keywords = ['governance', 'board', 'compliance', 'ethics', 'transparency', 'audit', 'risk', 'management', 'executive', 'compensation', 'shareholder', 'policy']
    categorized = defaultdict(list)
    for sentence in sentences:
        lower_sentence = sentence.lower()
        if any(keyword in lower_sentence for keyword in e_keywords):
            categorized['Environmental'].append(sentence)
        if any(keyword in lower_sentence for keyword in s_keywords):
            categorized['Social'].append(sentence)
        if any(keyword in lower_sentence for keyword in g_keywords):
            categorized['Governance'].append(sentence)
    return categorized

def extract_years_and_numbers(text):
    years = re.findall(r'\b(20[0-2][0-9])\b', text)
    percentages = re.findall(r'(\d+(?:\.\d+)?)\s*%', text)
    numbers_with_units = re.findall(r'(\d+(?:\.\d+)?)\s*(tons?|t|MT|CO2e?|MWh|GWh|kWh|m3|gallons?)', text)
    return {
        'years': years,
        'percentages': percentages,
        'numbers_with_units': numbers_with_units
    }

def text_to_dataframe(text):
    cleaned_text = clean_text(text)
    categorized = identify_esg_categories(cleaned_text)
    category_data = []
    for category, sentences_list in categorized.items():
        for sentence in sentences_list:
            extracted = extract_years_and_numbers(sentence)
            years = ', '.join(extracted['years']) if extracted['years'] else 'N/A'
            values = ', '.join(extracted['percentages']) if extracted['percentages'] else 'N/A'
            sentiment = sia.polarity_scores(sentence)['compound']
            category_data.append({
                'Category': category,
                'Statement': sentence,
                'Years': years,
                'Values': values,
                'Sentiment Score': sentiment
            })
    category_df = pd.DataFrame(category_data) if category_data else pd.DataFrame(columns=['Category', 'Statement', 'Years', 'Values', 'Sentiment Score'])
    return category_df

def main():
    st.title("ESG Report Analyzer")
    st.write("""
    Upload your ESG report file (PDF, DOCX, or TXT) to extract structured data.
    This tool will attempt to identify ESG metrics, categorize statements, and extract relevant data points.
    """)
    uploaded_file = st.file_uploader("Upload ESG Report", type=['pdf', 'docx', 'txt'])
    if uploaded_file is not None:
        st.info(f"Processing {uploaded_file.name}...")
        file_extension = uploaded_file.name.split('.')[-1].lower()
        try:
            if file_extension == 'pdf':
                text = extract_text_from_pdf(uploaded_file)
            elif file_extension == 'docx':
                text = extract_text_from_docx(uploaded_file)
            elif file_extension == 'txt':
                text = extract_text_from_txt(uploaded_file)
            else:
                st.error("Unsupported file format.")
                return
            st.subheader("Sample of Extracted Text")
            st.text_area("Extracted Text", text, height=200)
            category_df = text_to_dataframe(text)
            st.subheader("ESG Categories and Statements")
            if len(category_df) > 0:
                st.dataframe(category_df)
            else:
                st.warning("No ESG categories detected in the text.")
            if len(category_df) > 0:
                st.subheader("ESG Sentiment Scores")
                sentiment_summary = category_df.groupby("Category")["Sentiment Score"].mean().reset_index()
                sentiment_summary.columns = ["ESG Pillar", "Average Sentiment"]
                st.dataframe(sentiment_summary)
                st.bar_chart(sentiment_summary.set_index("ESG Pillar"))
            if len(category_df) > 0:
                st.subheader("Download Results")
                csv_category = category_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download ESG Categories CSV",
                    data=csv_category,
                    file_name="esg_categories.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")