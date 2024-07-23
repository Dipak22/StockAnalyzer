import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Function to calculate RSI
def calculate_rsi(data, window):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

# Streamlit app
st.title("Stock Analysis App")
st.sidebar.header("User Input")

# Sidebar input for stock ticker
ticker = st.sidebar.text_input("Stock Ticker", "RELIANCE.NS")

# Fetch historical data for the input stock
data = yf.download(ticker, period="1y")

# Calculate percentage change in price
data['Percentage Price Change'] = data['Close'].pct_change() * 100

# Calculate percentage change in volume
data['Percentage Volume Change'] = data['Volume'].pct_change() * 100

# Drop NaN values resulting from pct_change
data.dropna(inplace=True)

# Normalize percentage changes
data['Normalized Price Change'] = (data['Percentage Price Change'] - data['Percentage Price Change'].mean()) / data['Percentage Price Change'].std()
data['Normalized Volume Change'] = (data['Percentage Volume Change'] - data['Percentage Volume Change'].mean()) / data['Percentage Volume Change'].std()

# Combine the normalized metrics into a single feature
data['Combined Feature'] = data['Normalized Price Change'] + data['Normalized Volume Change']

# Calculate RSI
data['RSI'] = calculate_rsi(data, window=14)

# Plotting
st.subheader("Close Price")
st.line_chart(data['Close'])

st.subheader("Percentage Changes")
fig, ax = plt.subplots()
ax.plot(data.index, data['Percentage Price Change'], label='Percentage Price Change', color='g')
ax.plot(data.index, data['Percentage Volume Change'], label='Percentage Volume Change', color='b')
ax.legend()
st.pyplot(fig)

st.subheader("Combined Feature")
st.line_chart(data['Combined Feature'])

st.subheader("RSI")
fig, ax = plt.subplots()
ax.plot(data.index, data['RSI'], label='RSI', color='purple')
ax.axhline(70, linestyle='--', alpha=0.5, color='red')
ax.axhline(30, linestyle='--', alpha=0.5, color='green')
ax.legend()
st.pyplot(fig)
