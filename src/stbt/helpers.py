#!/usr/bin/env python3
"""Module with Strategy class to all
   backtest related manipulations"""
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
matplotlib.use('TkAgg')

def plot_charts(df_ohlc):
    """Function to plot simple Close/Volume graph

    Args:
        df_ohlc (DataFrame):
            Close and Volume columns are necessary

    Returns:
        figure (figure):
            Matplotlib graph
    """
    if str(type(df_ohlc.index[0])) == "<class 'pandas._libs.tslib.Timestamp'>":
        pass
    else:
        df_ohlc.index = pd.to_datetime(df_ohlc.index)

    figure = plt.figure()

    ax1 = plt.subplot2grid((8, 1), (0, 0), rowspan=5, colspan=1)
    ax1.plot(df_ohlc.index, df_ohlc['Close'], label='Close')

    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Сharts')

    ax2 = plt.subplot2grid((8, 1), (6, 0), rowspan=2, colspan=1, sharex=ax1)
    ax2.plot(df_ohlc.index, df_ohlc['Volume'], label='Volume')

    plt.xticks(rotation=45)
    ax1.xaxis.set_major_locator(MaxNLocator(5))

    plt.ylabel('Volume')

    return figure

def resample(df_ohlc, frequency='H'):
    """Function to change frequency and fill the gaps"""
    df_ohlc = df_ohlc.resample(frequency).ffill()
    df_ohlc = df_ohlc.fillna(0)

    return df_ohlc

def get_sharpe(df_ohlc):
    """Function to calculate sharpe ratio"""
    if isinstance(df_ohlc, pd.DataFrame):
        return round((np.sqrt(len(df_ohlc)) * df_ohlc.mean() / df_ohlc.std().values).values[0], 2)
    else:
        return round((np.sqrt(len(df_ohlc)) * df_ohlc.mean() / df_ohlc.std()), 2)


def get_max_drawdown(returns):
    """Assumes returns is a pandas Series"""
    ret = returns.add(1).cumprod()
    daily_drawdown = ret.div(ret.cummax()).sub(1)
    max_drawdown = round(daily_drawdown.min(), 4)
    end = daily_drawdown.idxmin()
    start = ret.loc[:end].idxmax()
    return max_drawdown, start, end

def get_label_from_dict(settings_dict):
    """Function to get name from dict"""
    label = ''
    if 'delay' in settings_dict.keys():
        label += 'd' + str(settings_dict['delay'])
    if 'commissions_const' in settings_dict.keys():
        label += '_' + str(settings_dict['commissions_const'] * 100) + '%' + 'coms'

    return label
