#!/usr/bin/env python3
"""Module with Strategy class to all
   backtest related manipulations"""
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

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

def initialize_universe(ohlc_df):
    '''func to create technical data dfs and add them to dict'''
    # add moex_stocks Universe
    uni_dict = {}
    # ohlc_df = pd.read_csv(DATA_DIR + csv_file, index_col='Date')
    ohlc_df.index = pd.to_datetime(ohlc_df.index)
    ohlc_df.sort_index(inplace=True)
    ohlc_df = ohlc_df.loc[~ohlc_df.index.duplicated(keep='first')]
    # print(ohlc_df.info(max_cols=1000, null_counts=True))
    High = [column for column in ohlc_df.columns if 'High' in column]
    Low = [column for column in ohlc_df.columns if 'Low' in column]
    Close = [column for column in ohlc_df.columns if 'Close' in column]
    Open = [column for column in ohlc_df.columns if 'Open' in column]
    Volume = [column for column in ohlc_df.columns if 'Volume' in column]

    uni_dict['close'] = ohlc_df[Close].rename(columns=lambda col: col.replace('Close_', ''))
    uni_dict['volume'] = ohlc_df[Volume].rename(columns=lambda col: col.replace('Volume_', ''))
    uni_dict['open'] = ohlc_df[Open].rename(columns=lambda col: col.replace('Open_', ''))
    uni_dict['low'] = ohlc_df[Low].rename(columns=lambda col: col.replace('Low_', ''))
    uni_dict['high'] = ohlc_df[High].rename(columns=lambda col: col.replace('High_', ''))
    uni_dict['all'] = ohlc_df

    return uni_dict

# def prolong(series):
#     '''func to prolong weight
#     weights_df = events.rolling(prolong_timeframe).apply(prolong)'''
#     new_ser = series.copy()
#     # if new_ser[0] == 0 and new_ser.sum() != 0:
#     if new_ser.sum() > 0 and new_ser.sum() != len(new_ser):
#         return 1
#     elif new_ser.sum() < 0 and new_ser.sum() != -len(new_ser):
#         return -1
#     else:
#         return 0

def prolong(series):
    '''func to prolong weight
    weights_df = events.rolling(prolong_timeframe).apply(prolong)'''
    new_ser = series.copy()
    # if new_ser[0] == 0 and new_ser.sum() != 0:
    non_zero = new_ser[new_ser != 0]
    if not non_zero.empty:
        if non_zero[-1] > 0 and non_zero.sum() != len(new_ser):
            return 1
        elif non_zero[-1] < 0 and non_zero.sum() != -len(new_ser):
            return -1
        else:
            return 0
    else:
        return 0

def unpack_ohlc(ohlc_df):
    '''Just helper to create dict with all tech vars'''

    High = [column for column in ohlc_df.columns if 'High' in column]
    Low = [column for column in ohlc_df.columns if 'Low' in column]
    Close = [column for column in ohlc_df.columns if 'Close' in column]
    Open = [column for column in ohlc_df.columns if 'Open' in column]
    Volume = [column for column in ohlc_df.columns if 'Volume' in column]

    close = ohlc_df[Close].rename(columns=lambda col: col.replace('Close_', ''))
    volume = ohlc_df[Volume].rename(columns=lambda col: col.replace('Volume_', ''))
    opens = ohlc_df[Open].rename(columns=lambda col: col.replace('Open_', ''))
    low = ohlc_df[Low].rename(columns=lambda col: col.replace('Low_', ''))
    high = ohlc_df[High].rename(columns=lambda col: col.replace('High_', ''))

    return volume, opens, high, low, close
