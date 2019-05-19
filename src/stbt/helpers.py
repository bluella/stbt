#!/usr/bin/env python3
"""Module with Strategy class to all
   backtest related manipulations"""
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
matplotlib.use('TkAgg')

def plot_charts(df):
    """Function to plot simple Close/Volume graph

    Args:
        df (DataFrame):
            Close and Volume columns are necessary

    Returns:
        figure (figure):
            Matplotlib graph
    """
    if str(type(df.index[0])) == "<class 'pandas._libs.tslib.Timestamp'>":
        pass
    else:
        df.index = pd.to_datetime(df.index)

    figure = plt.figure()

    ax1 = plt.subplot2grid((8, 1), (0, 0), rowspan=5, colspan=1)
    ax1.plot(df.index, df['Close'], label='Close')

    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Сharts')

    ax2 = plt.subplot2grid((8, 1), (6, 0), rowspan=2, colspan=1, sharex=ax1)
    ax2.plot(df.index, df['Volume'], label='Volume')

    plt.xticks(rotation=45)
    ax1.xaxis.set_major_locator(MaxNLocator(5))

    plt.ylabel('Volume')

    return figure

def resample(df, frequency='H'):
    """Function to change frequency and fill the gaps"""
    df = df.resample(frequency).ffill()
    df = df.fillna(0)

    return df



def get_sharpe(df):
    """Function to calculate sharpe ratio"""
    if isinstance(df, pd.DataFrame):
        return round((np.sqrt(len(df)) * df.mean() / df.std().values).values[0], 2)
    else:
        return round((np.sqrt(len(df)) * df.mean() / df.std()), 2)


def get_max_Drawdown(returns):
    """Assumes returns is a pandas Series"""
    r = returns.add(1).cumprod()
    dd = r.div(r.cummax()).sub(1)
    mdd = dd.min()
    end = dd.idxmin()
    start = r.loc[:end].idxmax()
    return mdd, start, end

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



def correct_ohlc_df(df, frequency=None, cols_to_drop=None):
    """Function to modify df to the required format, checking for wrong entries
    and filling nans

    Args:
        df (DataFrame):
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

    if str(type(df.index[0])) == "<class 'pandas._libs.tslib.Timestamp'>":
        pass
    else:
        df.index = pd.to_datetime(df.index)

    # resampling data if needed
    if frequency is not None:
        df = df.resample(frequency).agg({
            'Close': 'last',
            'High': 'max',
            'Low': 'min',
            'Open': 'first',
            'Volume': 'sum',
        })
        df.index = df.index.strftime('%Y-%m-%d %H:%m:%s')

    df_before_correction = df

    # make ohlc right
    count_of_ohlc_mistakes = 0
    for index, row in df.iterrows():
        if row['Low'] > min(row['Close'], row['Open'], row['High']):
            df.loc[index, 'Low'] = min(row['Close'], row['Open'], row['High']) * 0.999
            count_of_ohlc_mistakes += 1
        if row['High'] < max(row['Close'], row['Open'], row['Low']):
            df.loc[index, 'High'] = max(row['Close'], row['Open'], row['Low']) * 1.001
            count_of_ohlc_mistakes += 1
        if row['Volume'] < 0:
            df.loc[index, 'Volume'] = abs(row['Volume'])
            count_of_ohlc_mistakes += 1

    # delete duplicates
    print('Duplicates found:', len(df[df.index.duplicated()]))
    df = df[~df.index.duplicated()]

    df.fillna(method='ffill', inplace=True)
    print('Missed candles added:', len(df) - len(df_before_correction))

    df.index.name = 'Date'

    return df
