import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from plyer import notification
import logging
from logging.handlers import RotatingFileHandler
import threading
import streamlit as st
from newsapi import NewsApiClient
from newsapi.newsapi_exception import NewsAPIException
from sector_stocks import sectors
from stock_names import stock_names
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the NewsAPI client
newsapi = NewsApiClient(api_key=os.getenv('NEWS_API_KEY'))

# Set up logging with rotating file handler
handler = RotatingFileHandler('stock_alerts.log', maxBytes=10000, backupCount=3)
logging.basicConfig(handlers=[handler], level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to fetch news articles for a given stock ticker
@st.cache_data(ttl=3600)
def fetch_news(ticker):
    try:
        articles = newsapi.get_everything(q=ticker, language='en', sort_by='publishedAt', page_size=5)
        if articles['status'] == 'ok':
            return articles['articles']
        else:
            logging.error(f"Error fetching news for {ticker}: {articles['status']}")
            return []
    except NewsAPIException as e:
        logging.error(f"NewsAPI error for {ticker}: {e}")
        return []
    except Exception as e:
        logging.error(f"Unexpected error fetching news for {ticker}: {e}")
        return []

# Function to calculate RSI
def calculate_rsi(data, window):
    try:
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        avg_gain = gain.rolling(window=window, min_periods=1).mean()
        avg_loss = loss.rolling(window=window, min_periods=1).mean()
        rs = avg_gain / (avg_loss + 1e-10)  # Added small epsilon to avoid division by zero
        rsi = 100 - (100 / (1 + rs))
        return rsi
    except Exception as e:
        logging.error(f"Error calculating RSI: {e}")
        return pd.Series()

# Function to calculate LWTI
def calculate_lwti(prices, period):
    try:
        weights = np.arange(1, period + 1)
        wma = prices.rolling(window=period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
        return wma
    except Exception as e:
        logging.error(f"Error calculating LWTI: {e}")
        return np.array([])

# Function to calculate indicators
@st.cache_data(ttl=3600)
def calculate_indicators(data):
    try:
        data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()
        data['EMA_200'] = data['Close'].ewm(span=200, adjust=False).mean()

        window = 20
        data['SMA_20'] = data['Close'].rolling(window=window).mean()
        data['STD_20'] = data['Close'].rolling(window=window).std()
        data['Upper Band'] = data['SMA_20'] + (data['STD_20'] * 2)
        data['Lower Band'] = data['SMA_20'] - (data['STD_20'] * 2)

        ema_12 = data['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = ema_12 - ema_26
        data['Signal Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
        data['MACD Histogram'] = data['MACD'] - data['Signal Line']

        data['RSI'] = calculate_rsi(data, window=14)

        donchian_period = 20
        data['Upper Channel'] = data['High'].rolling(window=donchian_period).max()
        data['Lower Channel'] = data['Low'].rolling(window=donchian_period).min()
        data['Middle Channel'] = (data['Upper Channel'] + data['Lower Channel']) / 2

        # Calculate log returns
        data['Log Returns'] = np.log(data['Close'] / data['Close'].shift(1))

        # Calculate 14-day moving volatility
        data['14-Day Volatility'] = data['Log Returns'].rolling(window=14).std() * np.sqrt(14)
        vma_period = 20
        data['VMA_SMA'] = data['Volume'].rolling(window=vma_period).mean()
        data['VMA_EMA'] = data['Volume'].ewm(span=vma_period, adjust=False).mean()

        return data
    except Exception as e:
        logging.error(f"Error calculating indicators: {e}")
        return pd.DataFrame()

# Function to check for buy/sell signals
def check_signals(data):
    try:
        last_row = data.iloc[-1]
        previous_row = data.iloc[-2]
        buy_signal = (last_row['RSI'] > 30 and previous_row['RSI'] <= 30) and \
                     (last_row['MACD'] > last_row['Signal Line'] and previous_row['MACD'] <= previous_row['Signal Line']) and \
                     (last_row['Close'] > last_row['Lower Band'])
        
        sell_signal = (last_row['RSI'] < 70 and previous_row['RSI'] >= 70) and \
                      (last_row['MACD'] < last_row['Signal Line'] and previous_row['MACD'] >= previous_row['Signal Line']) and \
                      (last_row['Close'] < last_row['Upper Band'])
        if buy_signal:
            return "Buy"
        elif sell_signal:
            return "Sell"
        else:
            return "None"
        
    except Exception as e:
        logging.error(f"Error checking signals: {e}")
        return "None"

@st.cache_data(ttl=3600)
def update_daily_data(ticker):
    try:
        end_time = datetime.now()
        start_time = end_time - timedelta(days=365)  # Fetch one year of daily data
        
        daily_data = yf.download(ticker, start=start_time, end=end_time, interval="1d")
        if daily_data.empty or 'Close' not in daily_data.columns:
            logging.error(f"Failed to fetch data for {ticker}")
            return pd.DataFrame(columns=['Close'])  # Return empty DataFrame with 'Close' column
        
        daily_data = calculate_indicators(daily_data)
        
        return daily_data
    except Exception as e:
        logging.error(f"Error fetching or updating data for {ticker}: {e}")
        return pd.DataFrame(columns=['Close'])  # Return empty DataFrame with 'Close' column
    
# Function to process a single stock and send notifications
def process_stock(stock):
    try:
        data = update_daily_data(stock)
        if data.empty:
            return "None"
        return check_signals(data)
        
    except Exception as e:
        logging.error(f"Error processing stock {stock}: {e}")
        return "None"

# Function to send desktop notification
def send_notification(stock, sector, signal):
    notification.notify(
        title=f"Stock Alert: {stock} in {sector}",
        message=f"{signal} signal detected for {stock}",
        timeout=10
    )

# Function to check all stocks for signals
def check_sector_stocks(sector):
    logging.info(f"Trying update for {sector}")
    for stock in sectors[sector]:
        signal = process_stock(stock)
        if signal in ["Buy", "Sell"]:
            send_notification(stock, sector, signal)

# Main Streamlit app
def run_signal_check_in_background(sector):
    with st.spinner(f"Running signal check for {sector} sector..."):
        check_sector_stocks(sector)
        print(f'Run signal completed for {sector}')
        st.success(f"Signal check completed for {sector} sector!")

# Initialize session state
if 'selected_sector' not in st.session_state:
    st.session_state.selected_sector = list(sectors.keys())[0]

if 'selected_stock' not in st.session_state:
    st.session_state.selected_stock = None

# Streamlit app
st.title("Stock Analysis and News Dashboard")

# Sidebar for selecting sector and stocks
with st.sidebar:
    st.header("Sector and Stocks")
    st.session_state.selected_sector = st.selectbox("Select Sector", list(sectors.keys()), index=list(sectors.keys()).index(st.session_state.selected_sector))
    if st.button("Run Signal Check"):
        # Run the signal check for the selected sector in a separate thread
        thread = threading.Thread(target=run_signal_check_in_background, args=(st.session_state.selected_sector,))
        thread.start()

    st.header(f"Stocks in {st.session_state.selected_sector} Sector")

    # Display stock buttons in the sidebar
    for stock in sectors[st.session_state.selected_sector]:
        if st.button(stock):
            st.session_state.selected_stock = stock

# Get the selected stock
stock = st.session_state.selected_stock

if stock:
    st.subheader(f"Stock Data for {stock}")

    # Tabs for data, news, and signals
    tab1, tab2 = st.tabs(["Indicators", "News"])

    # Indicators tab
    with tab1:
        data = update_daily_data(stock)
        if data.empty or 'Close' not in data.columns:
            st.error(f"Unable to fetch or process data for {stock}. Please try again later.")
        else:
            # Plot the indicators
            fig, ax = plt.subplots(3, 1, figsize=(15, 10))

            # Plot EMA and Bollinger Bands
            ax[0].plot(data.index, data['Close'], label='Close Price')
            if 'EMA_50' in data.columns:
                ax[0].plot(data.index, data['EMA_50'], label='EMA 50')
            if 'EMA_200' in data.columns:
                ax[0].plot(data.index, data['EMA_200'], label='EMA 200')
            if 'Upper Band' in data.columns and 'Lower Band' in data.columns:
                ax[0].plot(data.index, data['Upper Band'], label='Upper Bollinger Band', linestyle='--')
                ax[0].plot(data.index, data['Lower Band'], label='Lower Bollinger Band', linestyle='--')
            ax[0].set_title('Close Price with EMA and Bollinger Bands')
            ax[0].legend()

            # Plot MACD and RSI
            if 'MACD' in data.columns and 'Signal Line' in data.columns:
                ax[1].plot(data.index, data['MACD'], label='MACD')
                ax[1].plot(data.index, data['Signal Line'], label='Signal Line')
                if 'MACD Histogram' in data.columns:
                    ax[1].bar(data.index, data['MACD Histogram'], label='MACD Histogram', color='gray')
                ax[1].set_title('MACD')
                ax[1].legend()

            if 'RSI' in data.columns:
                ax[2].plot(data.index, data['RSI'], label='RSI')
                ax[2].axhline(y=70, color='red', linestyle='--')
                ax[2].axhline(y=30, color='green', linestyle='--')
                ax[2].set_title('RSI')
                ax[2].legend()

            st.pyplot(fig)
            
            # Plot Donchian Channel, LWTI, and VMA
            fig2, ax2 = plt.subplots(3, 1, figsize=(15, 10))

            # Plot Donchian Channel
            ax2[0].plot(data.index, data['Close'], label='Close Price')
            if all(col in data.columns for col in ['Upper Channel', 'Lower Channel', 'Middle Channel']):
                ax2[0].plot(data.index, data['Upper Channel'], label='Upper Channel', linestyle='--')
                ax2[0].plot(data.index, data['Lower Channel'], label='Lower Channel', linestyle='--')
                ax2[0].plot(data.index, data['Middle Channel'], label='Middle Channel', linestyle='--')
            ax2[0].set_title('Donchian Channel')
            ax2[0].legend()

            # Plot LWTI
            #ax2[1].plot(data.index, data['Close'], label='Close Price')
            if '14-Day Volatility' in data.columns:
                ax2[1].plot(data.index, data['14-Day Volatility'], label='14-Day Volatility')
            ax2[1].set_title('Volatility')
            ax2[1].legend()

            # Plot VMA
            if 'Volume' in data.columns:
                ax2[2].plot(data.index, data['Volume'], label='Volume')
                if 'VMA_SMA' in data.columns:
                    ax2[2].plot(data.index, data['VMA_SMA'], label='VMA SMA')
                if 'VMA_EMA' in data.columns:
                    ax2[2].plot(data.index, data['VMA_EMA'], label='VMA EMA')
            ax2[2].set_title('Volume Moving Averages')
            ax2[2].legend()

            st.pyplot(fig2)
    
    with tab2:
        news_list = fetch_news(stock_names[stock])
        if news_list:
            for news in news_list:
                st.subheader(news['title'])
                st.write(news['publishedAt'])
                st.write(news['description'])
                st.write(f"[Read more]({news['url']})")
        else:
            st.write("No news available.")