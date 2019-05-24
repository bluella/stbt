#!/usr/bin/env python3
"""Module with technical operators/indicators"""
import numpy as np

def acceleration(df_ohlc, periods=24):
    """Self explanatory"""
    slopes = df_ohlc.pct_change()
    df_ohlc = regr(slopes, periods)

    return df_ohlc

def argmax(df_ohlc, periods=24):
    """Self explanatory"""

    return df_ohlc.rolling(periods).apply(np.argmax)

def argmin(df_ohlc, periods=24):
    """Self explanatory"""

    return df_ohlc.rolling(periods).apply(np.argmin)

def delay(df_ohlc, periods=1):
    """Self explanatory"""
    df_ohlc = df_ohlc.shift(periods)

    return df_ohlc

def delta(df_ohlc, periods=5):
    """Self explanatory"""
    df_ohlc = df_ohlc.diff(periods)

    return df_ohlc

def kurtosis(df_ohlc, periods=24):
    """https://en.wikipedia.org/wiki/Kurtosis_risk"""

    return df_ohlc.rolling(window=periods, center=False).kurt()

def neutralize(df_ohlc, axis=1):
    """https://realmoney.thestreet.com/articles/06/28/2016/portfolios-should-be-neutralized"""
    df_ohlc.fillna(0, inplace=True)
    df_ohlc_neu = df_ohlc - df_ohlc.mean(axis=axis)

    return df_ohlc_neu

def regr(df_ohlc, periods=24):
    """https://en.wikipedia.org/wiki/Linear_regression"""
    df_ohlc = df_ohlc.rolling(periods).apply(lambda x: np.cov(x, range(periods))[0][1]\
                                                                 / np.std(range(periods))**2)

    return df_ohlc

def sign(df_ohlc, low_zero, high_zero):
    """Assigns 0, -1, 1 depending on sign of value"""
    zero_labels = (df_ohlc > low_zero) & (df_ohlc < high_zero)
    minus_labels = df_ohlc <= low_zero
    plus_labels = df_ohlc >= high_zero
    df_ohlc[zero_labels] = 0
    df_ohlc[minus_labels] = -1
    df_ohlc[plus_labels] = 1

    return df_ohlc

def skewness(df_ohlc, periods=24):
    """https://en.wikipedia.org/wiki/Skewness_risk"""

    return df_ohlc.rolling(periods).skew()

def sma(df_ohlc, periods=5):
    """Simple moving average"""
    sma_df = df_ohlc.rolling(periods).mean()

    return sma_df

def standard_deviation(df_ohlc, periods=24):
    """Self explanatory"""
    df_ohlc = df_ohlc.rolling(periods, min_periods=periods).std()

    return df_ohlc

def trix(df_ohlc, periods=10):
    """https://www.investopedia.com/articles/technical/02/092402.asp"""
    ex1 = df_ohlc.ewm(span=periods, min_periods=periods).mean()
    ex2 = ex1.ewm(span=periods, min_periods=periods).mean()
    ex3 = ex2.ewm(span=periods, min_periods=periods).mean()
    trix_df = ex3.pct_change()

    return trix_df
