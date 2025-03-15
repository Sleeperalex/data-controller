import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__)))

import load_data

def main():

    # Streamlit App
    st.title("Finance Alerting System")

    # File Upload
    st.header("Upload Data Files")
    ticker_file = st.file_uploader("Upload Ticker CSV", type=["csv"])
    controversy_file = st.file_uploader("Upload Controversy CSV", type=["csv"])

    tickers, controverses = None, None
    if ticker_file:
        tickers = pd.read_csv(ticker_file, sep=";")
        st.write("### Ticker Data", tickers.head())

    if controversy_file:
        controverses = pd.read_csv(controversy_file, sep=";")
        st.write("### Controversy Data", controverses.head())

    ######################### Transformation ###############################

    transformed_datasets_folder = "transformed_datasets"

    # Check if the folder exists, if not create it
    if not os.path.exists(transformed_datasets_folder):
        os.makedirs(transformed_datasets_folder)
        print(f"Created directory: {transformed_datasets_folder}")

    # Check if transformed datasets already exist
    if os.path.isfile(f"{transformed_datasets_folder}/tickers.csv") and os.path.isfile(f"{transformed_datasets_folder}/controverses.csv"):
        tickers = pd.read_csv(f"{transformed_datasets_folder}/tickers.csv", sep=";")
        controverses = pd.read_csv(f"{transformed_datasets_folder}/controverses.csv", sep=";")
        print("Loaded existing transformed datasets")
    else:
        # Transform the datasets
        tickers, controverses = load_data.transformer(tickers, controverses)
        
        # Save the transformed datasets
        tickers.to_csv(f"{transformed_datasets_folder}/tickers.csv", sep=";")
        controverses.to_csv(f"{transformed_datasets_folder}/controverses.csv", sep=";")

    ####################### Default threshold values #######################
    thresholds = {
        "price_variation": 5.0,
        "volume_multiplier": 2.0,
        "pe_threshold": 40.0,
        "volatility_threshold": 0.1,
        "market_cap_threshold": 0.1,
        "environmental_threshold": 5.0,
        "governance_threshold": 3.0,
        "dividend_threshold": 0.2,
        "eps_threshold": -0.1
    }

    st.title("Threshold Settings")

    # Create sliders and number inputs for thresholds
    for key, value in thresholds.items():
        thresholds[key] = st.number_input(f"{key.replace('_', ' ').title()}", value=value, step=0.1, format="%.2f")

    # Display updated values
    st.write("### Updated Thresholds")
    st.json(thresholds)

    ############################# User Stock Input #############################
    user_ticker = st.text_input("Enter a stock ticker (e.g., AAPL, TSLA):")
    if user_ticker:
        stock = yf.Ticker(user_ticker)
        stock_hist = stock.history(period="1y")
        if not stock_hist.empty:
            st.write(f"### Historical Data for {user_ticker}", stock_hist.tail(10))
            st.line_chart(stock_hist['Close'])
            
            recent_close, previous_close = stock_hist['Close'].iloc[-1], stock_hist['Close'].iloc[-2]
            if (previous_close - recent_close) / previous_close > thresholds['price_variation'] / 100:
                st.error(f"Alert: {user_ticker} dropped more than {thresholds['price_variation']}% today!")
            else:
                st.success("No major alerts detected.")
        else:
            st.error("Invalid ticker or no data available.")

    # Anomaly Detection Functions
    def detect_price_anomalies(df):
        df['price_change'] = df['previous_close'].pct_change() * 100
        return df[abs(df['price_change']) > thresholds['price_variation']]

    def detect_volume_anomalies(df, avg_volume):
        return df[df['volume'] > thresholds['volume_multiplier'] * avg_volume]

    def detect_pe_anomalies(df):
        df['trailing_pe'] = pd.to_numeric(df['trailing_pe'], errors='coerce')
        return df[df['trailing_pe'] > thresholds['pe_threshold']]

    # Process Tickers & Fetch Data if uploaded
    if controverses is not None:
        valid_tickers = controverses['Tickers'].dropna().tolist()
        controverses['price_change'] = None
        controverses['volume'] = None
        controverses['trailing_pe'] = None
        for i, ticker in enumerate(valid_tickers[:50]):
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                controverses.loc[i, 'price_change'] = info.get('regularMarketPreviousClose', 0)
                controverses.loc[i, 'volume'] = info.get('volume', 0)
                controverses.loc[i, 'trailing_pe'] = info.get('trailingPE', 0)
            except:
                continue
        
        avg_volume = controverses['volume'].mean()
        anomalies_price = detect_price_anomalies(controverses)
        anomalies_volume = detect_volume_anomalies(controverses, avg_volume)
        anomalies_pe = detect_pe_anomalies(controverses)
        
        st.write("### Anomalies Detected")
        st.write("#### Price Anomalies", anomalies_price)
        st.write("#### Volume Anomalies", anomalies_volume)
        st.write("#### P/E Ratio Anomalies", anomalies_pe)

        # Charts
        st.subheader("Visualization of Anomalies")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=anomalies_price['Tickers'], y=anomalies_price['price_change'], name="Price Anomalies"))
        fig.add_trace(go.Bar(x=anomalies_volume['Tickers'], y=anomalies_volume['volume'], name="Volume Anomalies"))
        fig.add_trace(go.Bar(x=anomalies_pe['Tickers'], y=anomalies_pe['trailing_pe'], name="P/E Anomalies"))
        st.plotly_chart(fig)

        # Save Filtered Data
        controverses.to_csv('controverses_filtered.csv', index=False, sep=";", encoding="utf-8")
        st.success("Filtered data saved as controverses_filtered.csv")