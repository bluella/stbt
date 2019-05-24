#!/usr/bin/env python3
"""Module to download cryptocurrency ohlc data"""

import time
import datetime as dt
import logging
import json
import requests
import pandas as pd

def from_datetime_to_unix(date):
    '''in: datetime, out: unix_timestamp'''
    return int(time.mktime(date.timetuple()))

def from_unix_to_date(date):
    '''in: unix_timestamp, out: datetime'''
    value = dt.datetime.fromtimestamp(date)
    return value.date()

def str_to_datetime(time_str):
    '''in: str, out: datetime'''
    return dt.datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')


def columns_to_upper_case(df_ohlc):
    '''in : df, out : df, Makes all columns of df start with capital letter'''

    columns = list(df_ohlc.columns)
    for column in columns:
        if column[0].isupper():
            pass
        else:
            tmp_column_name = column[0].upper() + column[1:]
            df_ohlc.rename(index=str, columns={column: tmp_column_name}, inplace=True)
    return df_ohlc


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
    logging.debug('Duplicates found: %s', len(df_ohlc[df_ohlc.index.duplicated()]))
    df_ohlc = df_ohlc[~df_ohlc.index.duplicated()]

    df_ohlc.fillna(method='ffill', inplace=True)
    logging.debug('Missed candles added: %s', len(df_ohlc) - len(df_before_correction))

    return df_ohlc



def get_ohlc_cryptocompare_once(first_ticker, second_ticker, end_date=dt.datetime.now(),
                                aggregate=1, interval_key='day'):
    """ Retrieve limited bulk of ohlc cryptocurrency data from Cryptocompare.

    Args:

        first_ticker (str):
            Crypto symbol(BTC).
        second_ticker (str):
            Crypto symbol(USD).
        aggregate (int):
            How many points should be made into one
        interval_key (str):
            Time interval of data points
        end_date (datetime):
            Last moment in ohlc data

    Returns:
        df_ohlc (pandas.DataFrame):
            DF containing the opening price, high price, low price,
            closing price, and volume.

    Note:
        Data is limited(only 2000 point of data will be given)
    """
    limit = 2000
    df_ohlc = pd.DataFrame()
    interval_dict = {'minute': 'histominute', 'hour': 'histohour', 'day': 'histoday'}
    freq_dict = {'minute': '1M', 'hour': '1H', 'day': '1D'}
    end_date_unix = from_datetime_to_unix(end_date)

    url = 'https://min-api.cryptocompare.com/data/{}'.format(interval_dict[interval_key]) +\
        '?fsym={}'.format(first_ticker) +\
        '&tsym={}'.format(second_ticker) +\
        '&limit={}'.format(limit) +\
        '&aggregate={}'.format(aggregate) +\
        '&toTs={}'.format(str(end_date_unix))
    response = requests.get(url)
    resp_dict = json.loads(response.text)


    # parsing response dict to pieces
    if resp_dict["Response"] == "Success":
        data = resp_dict['Data']
        df_ohlc = pd.DataFrame(data)
        df_ohlc = columns_to_upper_case(df_ohlc)
        df_ohlc['Date'] = [dt.datetime.fromtimestamp(d) for d in df_ohlc.Time]
        df_ohlc['Volume'] = [v for v in df_ohlc.Volumeto]
        df_ohlc.set_index('Date', inplace=True)
        df_ohlc.index.name = 'Date'
        df_ohlc = correct_ohlc_df(df_ohlc, freq_dict[interval_key])

    elif resp_dict["Response"] == "Error":
        logging.error("There was an error in response from cryptocompare: %s", resp_dict)
    else:
        logging.error("Unknown response from cryptocompare: %s", resp_dict)

    return df_ohlc

def get_ohlc_cryptocompare(first_ticker, second_ticker, start_date,
                           end_date=dt.datetime.now(), **kwargs):
    """ Retrieves ohlc cryptocurrency data from Cryptocompare.

    Args:

        first_ticker (str):
            Crypto symbol(BTC).
        second_ticker (str):
            Crypto symbol(USD).
        start_date (datetime):
            First moment in ohlc data
        end_date (datetime):
            Optional.Last moment in ohlc data
        aggregate (int):
            Optional.How many points should be made into one
        interval_key (str):
            Optional.Time interval of data points


    Returns:
        df_total (pandas.DataFrame):
            DF containing the opening price, high price, low price,
            closing price, and volume.

    Note:
        This this loop for get_ohlc_cryptocompare_once
    """
    freq_dict = {'minute': '1M', 'hour': '1H', 'day': '1D'}
    df_total = get_ohlc_cryptocompare_once(first_ticker, second_ticker,
                                           end_date=end_date, **kwargs)
    new_start_date = df_total.index.min()
    while new_start_date > start_date:
        df_tmp = get_ohlc_cryptocompare_once(first_ticker, second_ticker,
                                             end_date=new_start_date, **kwargs)
        new_start_date = df_tmp.index.min()
        frames = [df_tmp, df_total]
        df_total = pd.concat(frames)
        df_total.drop_duplicates(inplace=True)
        time.sleep(10) # sort of gentle timeout for cryptocompare
    df_total = df_total[df_total.index >= start_date]
    if 'interval_key' in kwargs:
        df_total = correct_ohlc_df(df_total, freq_dict[kwargs['interval_key']])

    return df_total


# tryouts
if __name__ == "__main__":
    # vars for cryptocompare
    F_TICKER = 'BTC'
    S_TICKER = 'USD'

    END_DATE = dt.datetime(2018, 7, 1, 0, 0, 0)
    START_DATE = dt.datetime(2018, 3, 1, 0, 0, 0)

    OHLC_DF = get_ohlc_cryptocompare(F_TICKER, S_TICKER, START_DATE,
                                     end_date=END_DATE, interval_key='hour')

    print(OHLC_DF)
