#!/usr/bin/env python3
"""Module with events"""
import numpy as np
import pandas as pd
import stbt.operators.technical as ti


def get_acc_events(close, lookback=60, threshold=0.1):
    '''event of acceleration being positive or negative for a long time
    use afterwards as:
    result_df = events.rolling(prolong_timeframe).apply(prolong)
    threshold is prevailance of positive acc against negative 0.1 = 10% more

    NOTE: try doing acc sum without sign
    '''
    acc_df = ti.acc(close)
    acc_sign_sum = acc_df.applymap(np.sign).rolling(window=lookback).sum()
    events = ti.sign(acc_sign_sum, -1*threshold*lookback, threshold*lookback)

    return events


###############################################################################
# volume events
###############################################################################

def get_vwap_events(close, volume, threshold=.01, vwap_timeframe=60):
    '''event of close being to far from vwap
    use afterwards as:
    result_df = events.rolling(prolong_timeframe).apply(prolong)'''

    vwap = ti.vwap(close, volume, vwap_timeframe)
    events = ti.sign((vwap - close)/(close+vwap)*2, -1*threshold, threshold)

    return events

def get_high_volume_events(volume, lookback=60, threshold=10):
    '''event of volume being too high
    use afterwards as:
    result_df = events.rolling(prolong_timeframe).apply(prolong)'''
    vol60_df = volume.rolling(lookback).mean()
    vol60_df.fillna(0, inplace=True)
    events = ti.sign((volume - vol60_df)/(vol60_df+1), -1*threshold, threshold)

    return events

def get_low_volume_events(volume, lookback=60, threshold=0):
    '''event of volume being lesser than average,
    threshold=0 stands for equal to average, -1 - 2*average
    use afterwards as:
    result_df = events.rolling(prolong_timeframe).apply(prolong)'''
    vol60_df = volume.rolling(lookback).mean()
    vol60_df.fillna(0, inplace=True)
    events = ti.sign((vol60_df - volume)/(vol60_df+1), -9999999999, threshold)

    return events

###############################################################################
# candle type events
###############################################################################
def get_big_candle_body_events(close, opens, threshold=.01):
    '''event of large abs(close-opens)
    use afterwards as:
    result_df = events.rolling(prolong_timeframe).apply(prolong)'''

    events = ti.sign((close-opens)/(close+opens)*2, -1*threshold, threshold)

    return events

def get_small_candle_body_events(close, opens, threshold=1.5, lookback=60):
    '''event of candle being no larger than ave_candle * threshold
    use afterwards as:
    result_df = events.rolling(prolong_timeframe).apply(prolong)'''
    average_candle = (close - opens).abs().rolling(lookback).mean()
    pos_ind = ((close-opens) >= 0) & ((close-opens) < average_candle * threshold)
    neg_ind = ((close-opens) < 0) & ((close-opens) < average_candle * threshold)
    # zero_ind = (close-opens) >= average_candle * threshold
    events = close * 0
    events[pos_ind] = 1
    events[neg_ind] = -1

    return events

def get_small_candle_events(high, low, threshold=1.5, lookback=60):
    '''event of candle being no larger than ave_candle * threshold
    use afterwards as:
    result_df = events.rolling(prolong_timeframe).apply(prolong)'''
    average_candle = (high - low).abs().rolling(lookback).mean()
    pos_ind = ((high-low) >= 0) & ((high-low) < average_candle * threshold)

    events = high * 0
    events[pos_ind] = 1

    return events

def get_big_tail_events(close, high, low, open_prices, threshold=1.005):
    '''event of large tail
    use afterwards as:
    result_df = events.rolling(prolong_timeframe).apply(prolong)'''
    high_tail_df = high - pd.concat([close, open_prices]).max(level=0)
    low_tail_df = - low + pd.concat([close, open_prices]).min(level=0)

    pos_events = ti.sign(high_tail_df/high + 1, -1*threshold, threshold)
    neg_events = ti.sign(low_tail_df/high - 1, -1*threshold, threshold)
    events = pos_events + neg_events

    return events

###############################################################################
# other events
###############################################################################

def get_event_count_events(events_df, count, timeframe):
    '''event of sum of events over timeframe being bigger than count
    or lesser than -count
    use afterwards as:
    result_df = events.rolling(prolong_timeframe).apply(prolong)'''
    events_sum = events_df.rolling(timeframe).sum()
    pos_ind = events_sum >= count
    neg_ind = events_sum <= - count
    events = events_sum * 0
    events[pos_ind] = 1
    events[neg_ind] = -1

    return events

def get_trend_events(volume, threshold=1.01, lookback=60):
    '''event of slope being bigger than threshold
    use afterwards as:
    result_df = events.rolling(prolong_timeframe).apply(prolong)'''

    slopes = volume.rolling(lookback).apply(lambda x: \
        np.polyfit(volume.index, volume/volume[-1], 1)[0])
    pos_ind = slopes >= threshold
    neg_ind = slopes <= - threshold
    events = volume * 0
    events[pos_ind] = 1
    events[neg_ind] = -1

    return events
