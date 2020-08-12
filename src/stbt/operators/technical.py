#!/usr/bin/env python3
"""Module with technical operators/indicators"""
import numpy as np
import pandas as pd
from lib.helpers import string_to_time


def vwap(close, volume, vwaptimeframe=60):
    vol60_df = volume.rolling(vwaptimeframe).sum()
    vol60_df.fillna(0, inplace=True)
    close_vol_mul_df = volume*close
    close_vol_cumsum_df = close_vol_mul_df.rolling(vwaptimeframe).sum()
    close_vol_cumsum_df.fillna(0, inplace=True)
    vwap_df = close_vol_cumsum_df/vol60_df

    return vwap_df


def simple_sign(ohlc_df):
    df = ohlc_df.copy()
    df[df > 0] = 1
    df[df < 0] = -1
    df.fillna(0)
    return df


def custom_rank(array):
    s = pd.Series(array)
    return s.rank(ascending=False)[len(s)-1]


def normalize(df_ohlc, timeframe=60):
    return df_ohlc/(df_ohlc.rolling(timeframe).max())


def neutralize(weights, axis=1):
    """https://realmoney.thestreet.com/articles/06/28/2016/portfolios-should-be-neutralized"""

    weights = weights.sub(weights.mean(axis=1), axis=0)

    return weights


def delay(weights, delay=1):
    """https://realmoney.thestreet.com/articles/06/28/2016/portfolios-should-be-neutralized"""

    weights = weights.shift(delay)

    return weights


def delta(df_ohlc, periods=5):
    """Self explanatory"""
    df_ohlc = df_ohlc.diff(periods)

    return df_ohlc


def acc(closes):
    """Self explanatory"""
    acc = (closes - 2 * delay(closes, 1) + delay(closes, 2))/delay(closes, 2)
    # acc = close_minus_open.pct_change()

    return acc


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


def regr(df_ohlc, periods=24):
    """https://en.wikipedia.org/wiki/Linear_regression"""
    df_ohlc = df_ohlc.rolling(periods).apply(lambda x: np.cov(x, range(periods))[0][1]
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


def standard_deviation(df_ohlc, periods=24):
    """Self explanatory"""
    df_ohlc = df_ohlc.rolling(periods, min_periods=periods).std()

    return df_ohlc


def ts_rank(df_ohlc, periods=60):
    """Self explanatory"""

    return df_ohlc.rolling(periods).apply(custom_rank)


def scale(weights):
    """Self explanatory"""

    return weights.div(weights.abs().sum(axis=1), axis=0)


def rank(weights):
    """Self explanatory"""

    return weights.rank(axis=1).div(weights.rank(axis=1).max(axis=1), axis=0)


def hump(weights, change_threshold=1):
    """Self explanatory"""
    previous_row = None
    previous_index = None
    for index, row in weights.iterrows():
        if previous_index:
            if (row - previous_row).abs().sum()/max(row.abs().sum(), 1) < change_threshold:
                weights.loc[index, :] = previous_row
        previous_row = row
        previous_index = index

    return weights


def null_noise(df_ohlc, low_zero=-0.01, high_zero=0.01):
    """Assigns 0, -1, 1 depending on sign of value"""
    zero_labels = (df_ohlc > low_zero) & (df_ohlc < high_zero)
    df_ohlc[zero_labels] = 0

    return df_ohlc


def null_daytime_timeperiod(df_weights, start="10:00", end="10:30"):
    """Assigns 0 if df.index is in e.g. from 10:00 to 10:30"""
    indexes = (df_weights.index.time > string_to_time(start)) &\
        (df_weights.index.time < string_to_time(end))
    df_weights[indexes] = 0

    return df_weights
