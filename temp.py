import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from newsapi import NewsApiClient
import logging
from plyer import notification
import threading

# Set up logging to file
logging.basicConfig(filename='stock_alerts.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the NewsAPI client
newsapi = NewsApiClient(api_key='YOUR_NEWSAPI_KEY')

# Function to fetch news articles for a given stock ticker
def fetch_news(ticker):
    try:
        articles = newsapi.get_everything(q=ticker, language='en', sort_by='publishedAt', page_size=5)
        if articles['status'] == 'ok':
            news_list = [article for article in articles['articles'] if ticker.lower() in article['title'].lower() or ticker.lower() in article['description'].lower()]
            return news_list
        else:
            logging.error(f"Error fetching news for {ticker}: {articles['status']}")
            return []
    except Exception as e:
        logging.error(f"Error fetching news for {ticker}: {e}")
        return []

# Function to fetch and update data for indicators on a daily basis
def update_daily_data(ticker):
    try:
        end_time = datetime.now()
        start_time = end_time - timedelta(days=365)
        daily_data = yf.download(ticker, start=start_time, end=end_time, interval="1d")
        return daily_data
    except Exception as e:
        logging.error(f"Error fetching or updating data for {ticker}: {e}")
        return pd.DataFrame()

# Function to send desktop notification
def send_notification(stock, signal):
    notification.notify(
        title=f"Stock Alert: {stock}",
        message=f"{signal} signal detected for {stock}",
        timeout=10
    )

# Function to check for buy/sell signals for a given stock
def check_signals(stock):
    data = update_daily_data(stock)
    if data.empty:
        return None

    # Add any additional indicators you want to check here
    data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()
    data['EMA_200'] = data['Close'].ewm(span=200, adjust=False).mean()
    
    latest_close = data['Close'].iloc[-1]
    latest_ema50 = data['EMA_50'].iloc[-1]
    latest_ema200 = data['EMA_200'].iloc[-1]

    if latest_close > latest_ema50 > latest_ema200:
        return "Buy"
    elif latest_close < latest_ema50 < latest_ema200:
        return "Sell"
    else:
        return None

# Function to check stocks for signals in the selected sector
def check_sector_stocks(sector):
    stocks = sectors.get(sector, [])
    for stock in stocks:
        signal = "Buy"
        if signal:
            send_notification(stock, signal)

# Define a list of stock tickers for each sector
sectors = {
    "Technology": ["AAPL", "MSFT", "GOOGL"],
    "Finance": ["JPM", "BAC", "WFC"],
    "Healthcare": ["JNJ", "PFE", "MRK"]
}

# Main Streamlit app
def run_signal_check_in_background(sector):
    check_sector_stocks(sector)

# Initialize session state
if 'selected_sector' not in st.session_state:
    st.session_state.selected_sector = list(sectors.keys())[0]

if 'selected_stock' not in st.session_state:
    st.session_state.selected_stock = None

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
        if not data.empty:
            st.line_chart(data['Close'])

            data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()
            data['EMA_200'] = data['Close'].ewm(span=200, adjust=False).mean()

            fig, ax = plt.subplots()
            data['Close'].plot(ax=ax, label='Close')
            data['EMA_50'].plot(ax=ax, label='EMA 50')
            data['EMA_200'].plot(ax=ax, label='EMA 200')
            ax.legend()
            st.pyplot(fig)

    # News tab
    with tab2:
        news_list = fetch_news(stock)
        if news_list:
            for news in news_list:
                st.subheader(news['title'])
                st.write(news['description'])
                st.write(f"[Read more]({news['url']})")
        else:
            st.write("No relevant news available.")
