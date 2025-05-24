import yfinance as yf
import pandas as pd
import numpy as np

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def fetch_data(ticker="AAPL", start="2022-01-01", end="2023-01-01"):
    df = yf.download(ticker, start=start, end=end, auto_adjust=True)

    # Moving Averages
    df["MA10"] = df["Close"].rolling(window=10).mean()
    df["MA50"] = df["Close"].rolling(window=50).mean()

    # RSI manually calculated
    df["RSI"] = calculate_rsi(df["Close"], period=14)

    df.dropna(inplace=True)

    print(df[["Open", "High", "Low", "Close", "Volume", "MA10", "MA50", "RSI"]].tail())
    return df

if __name__ == "__main__":
    symbol = input("Enter the stock ticker (e.g., AAPL, TSLA, GOOGL): ").upper()
    fetch_data(ticker=symbol)