import math
import pandas as pd
import numpy as np
import scipy as sp
import datetime

import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os

def fill_missing_values(df_data):
    """Fill missing values in data frame, in place."""
    df_data.fillna(method="ffill", inplace=True)
    df_data.fillna(method="bfill", inplace=True)


def symbol_to_path(symbol, base_dir="Data"):
    """Return CSV file path given ticker symbol."""
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))


def get_data(symbols, dates):
    """Read stock data (adjusted close) for given symbols from CSV files."""
    df_final = pd.DataFrame(index=dates)

    if "SPY" not in symbols:  # add SPY for reference, if absent
        symbols.insert(0, "SPY")
    
    for symbol in symbols:
        file_path = symbol_to_path(symbol)
        df_temp = pd.read_csv(file_path, parse_dates=True, index_col="Date",
            usecols=["Date", "Adj Close"], na_values=["nan"])
        df_temp = df_temp.rename(columns={"Adj Close": symbol})
        df_final = df_final.join(df_temp)
        if symbol == "SPY":  # drop dates SPY did not trade
            df_final = df_final.dropna(subset=["SPY"])

    return df_final

def compute_daily_returns(df):
    """Compute and return the daily return values."""
    daily_returns = (df / df.shift(1)) - 1
    daily_returns.ix[0, :] = 0
    return daily_returns

def print_stats(df):
    print("Mean = ", df.mean())
    print("Sdev = ", df.std())
    print("Sharpe Ratio = ", (df.mean()/df.std() * math.sqrt(252)))

def get_portfolio_performance(allocation, daily_returns, display=False):
    alloced = daily_returns * allocation
    port_return = alloced.sum(axis=1)

    mean = port_return.mean()
    std = port_return.std()
    sharpe = (mean / std) * math.sqrt(252)

    if (display==True):
        print ("allocation = ", allocation)
        print("Port return mean = ", mean)  # on a daily basis
        print("Port return std = ", std)    # on a daily basis
        print("Avg Annual return = ", mean * 252 * 100)
        print("Sharpe Ratio = ", sharpe)
    return(-sharpe)

def constraint1(x):
    return (1 - sum(x))

def test_run():
    """Function called by Test Run."""
    # Read data
    symbol_list = ["GOOG", "AAPL", "GLD", "XOM"]  # list of symbols
    allocs = [0.15, 0.15, 0.35, 0.35] # starting allocation for optimization
    start_date = "2008-01-01"
    end_date = "2011-01-01"
    dates = pd.date_range(start_date, end_date)  # date range as index
    df_data = get_data(symbol_list, dates)  # get data for each symbol
    df_data = df_data / df_data.ix[0, :]    # normalize closing prices
    # Fill missing values
    fill_missing_values(df_data)

    # drop SPY which is just used for trading dates
    del df_data['SPY']
    daily_returns = compute_daily_returns(df_data)

    bnds = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0))

    weights = minimize(get_portfolio_performance,
                       x0=allocs,
                       args=(daily_returns),
                       method='SLSQP',
                       options={'disp': False},
                       constraints=({'type': 'eq', 'fun': constraint1}),
                       bounds=bnds)
    print (weights)
    get_portfolio_performance(weights.x, daily_returns, True)

if __name__ == "__main__":
    test_run()
