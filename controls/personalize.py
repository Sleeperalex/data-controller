import streamlit as st

@st.cache_data
def check_regex(df, regex_pattern, column_name):
    if column_name in df.columns:
        if not df[column_name].str.match(regex_pattern).all():
            return False
    return True
