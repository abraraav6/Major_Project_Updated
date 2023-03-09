import streamlit as st
import yfinance as yf
from sklearn.linear_model import LinearRegression
from datetime import date, timedelta
import plotly.graph_objects as go
'''
# Set up the app title
# Define a function to retrieve the historical stock prices
def get_stock_data(ticker):
    stock_data = yf.download(ticker, start="2015-01-01", end=date.today())
    return stock_data

# Define a function to train the linear regression model
def train_model(stock_data):
    X = [[i] for i in range(len(stock_data))]
    y = stock_data["Close"].values.reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, y)
    return model

# Define a function to predict the future stoc'k' pric''es
def predict_prices(model, days):
    last_date = date.today()
    future_dates = [last_date + timedelta(days=x) for x in range(1, days+1)]
    future_dates = [d.strftime("%Y-%m-%d") for d in future_dates]
    future_prices = model.predict([[i] for i in range(len(future_dates))])
    return future_dates, future_prices


if ticker:
    st.write(f"Stock prices for {ticker.upper()}:")
    stock_data = stock_data = yf.download(ticker, start="2015-01-01", end=date.today())
    st.line_chart(stock_data["Close"])
    X = [[i] for i in range(len(stock_data))]
    y = stock_data["Close"].values.reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, y)
    last_date = date.today()
    future_dates = [last_date + timedelta(days=x) for x in range(1, days+1)]
    future_dates = [d.strftime("%Y-%m-%d") for d in future_dates]
    future_prices = model.predict([[i] for i in range(len(future_dates))])
    st.write(f"Predicted stock prices for the next {days} days:")
    for i in range(len(future_dates)):
        st.write(future_dates[i], future_prices[i])



'''

# Main code
class stock_prediction:
                def start():
                        st.title("Stock Market Prediction App")
                    
                        # Define the user interface
                        ticker = st.text_input("Enter a stock ticker symbol (e.g. AAPL for Apple):")
                        days = st.slider("Select the number of days for the prediction:", 1, 365, 30)
                        if ticker:
                            st.write(f"Stock prices for {ticker.upper()}:")
                            stock_data = yf.download(ticker, start="2015-01-01", end=date.today())
                        
                            # Add candlestick chart
                            fig = go.Figure(data=[go.Candlestick(x=stock_data.index,
                                                                 open=stock_data['Open'],
                                                                 high=stock_data['High'],
                                                                 low=stock_data['Low'],
                                                                 close=stock_data['Close'])])
                            fig.update_layout(title=f"{ticker.upper()} Stock Prices", xaxis_title="Date", yaxis_title="Price")
                            st.plotly_chart(fig)
                        
                            X = [[i] for i in range(len(stock_data))]
                            y = stock_data["Close"].values.reshape(-1, 1)
                            model = LinearRegression()
                            model.fit(X, y)
                            last_date = date.today()
                            future_dates = [last_date + timedelta(days=x) for x in range(1, days+1)]
                            future_dates = [d.strftime("%Y-%m-%d") for d in future_dates]
                            future_prices = model.predict([[i] for i in range(len(future_dates))])
                            st.write(f"Predicted stock prices for the next {days} days:")
                            for i in range(len(future_dates)):
                                st.write(future_dates[i], future_prices[i])