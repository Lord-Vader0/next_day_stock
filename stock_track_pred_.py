import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plot
from sklearn.linear_model import LinearRegression as LinReg

def main():
    while True:

        stock_name = input("Enter stock tickers seperated by commas (no spaces): ").upper()

        tickers = []
        raw_list = stock_name.split(',')
        for stock in raw_list:
            tickers.append(stock)
        for ticker in tickers:
            try:
                current, next_day_pred = pred_price(ticker)
                print(f"The current price for {ticker} is ${current:.2f} and the estimated price tomorrow at closing is ${next_day_pred:.2f}")
            except Exception as e:
                print(f"Error processing {ticker}: {e}")

def pred_price(ticker):
    hist = yf.download(ticker, period = '120d')
    hist['past'] = hist['Close'].shift(1)
    hist.dropna(inplace=True)

    stock_model = LinReg()
    stock_model.fit(hist[['past']], hist['Close'])

    current = float(hist["Close"].iloc[-1])
    next_day_pred = stock_model.predict([[current]]).item()
    return current, next_day_pred

if __name__ == "__main__":
    main()


