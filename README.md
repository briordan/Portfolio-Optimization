# Portfolio Optimization

Optimize.py is a script I wrote to optimize asset allocation of a stock portfolio based on risk adjusted return (Sharpe Ratio).

## Motivation
This was a homework assignment for Georgia Tech's Machine Learning for Trading class.  

Sharpe ratio is a measure of the return of an investment relative to its risk or volatility.  For this
homework we were given three stocks and asked to find the optimal asset allocation based on
the Sharpe ratio.

## Overview of Code
The .csv files in the Data directory have daily price data for each stock or index.  SPY (the S&P 500 index) is used to 
determine days the market traded.  

Missing price data is forward filled, if there is no next data point back filling is used.

scipy.minimize is used to determine the optimal asset allocation.  Since larger Sharpe ratios are better, I use
I use negative Sharpe ratio.  And I bound the asset allocations between 0 and 1 for each asset.

