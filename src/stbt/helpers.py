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
    max_drawdown = daily_drawdown.min()
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


#########################################################################
# Fuctions to repair bad data
#########################################################################



def correct_ohlc_df(df_ohlc, frequency=None, cols_to_drop=None):
    """Function to modify df to the required format, checking for wrong entries
    and filling nans

    Args:
        df_ohlc (DataFrame):
            Close, Open, Low, High and Volume columns are necessary
        frequency (str):
            Resample frequency 'D', 'W', 'M', if None - do not resample data
        cols_to_drop (list):
            names of unnecessary columns in df

    Returns:
        df (DataFrame):
            Repaired df
    """
    if cols_to_drop is None:
        cols_to_drop = []

    if str(type(df_ohlc.index[0])) == "<class 'pandas._libs.tslib.Timestamp'>":
        pass
    else:
        df_ohlc.index = pd.to_datetime(df_ohlc.index)

    # resampling data if needed
    if frequency is not None:
        df_ohlc = df_ohlc.resample(frequency).agg({
            'Close': 'last',
            'High': 'max',
            'Low': 'min',
            'Open': 'first',
            'Volume': 'sum',
        })
        df_ohlc.index = df_ohlc.index.strftime('%Y-%m-%d %H:%m:%s')

    df_before_correction = df_ohlc

    # make ohlc right
    count_of_ohlc_mistakes = 0
    for index, row in df_ohlc.iterrows():
        if row['Low'] > min(row['Close'], row['Open'], row['High']):
            df_ohlc.loc[index, 'Low'] = min(row['Close'], row['Open'], row['High']) * 0.999
            count_of_ohlc_mistakes += 1
        if row['High'] < max(row['Close'], row['Open'], row['Low']):
            df_ohlc.loc[index, 'High'] = max(row['Close'], row['Open'], row['Low']) * 1.001
            count_of_ohlc_mistakes += 1
        if row['Volume'] < 0:
            df_ohlc.loc[index, 'Volume'] = abs(row['Volume'])
            count_of_ohlc_mistakes += 1

    # delete duplicates
    print('Duplicates found:', len(df_ohlc[df_ohlc.index.duplicated()]))
    df_ohlc = df_ohlc[~df_ohlc.index.duplicated()]

    df_ohlc.fillna(method='ffill', inplace=True)
    print('Missed candles added:', len(df_ohlc) - len(df_before_correction))

    df_ohlc.index.name = 'Date'

    return df_ohlc
