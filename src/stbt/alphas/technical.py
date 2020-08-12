#!/usr/bin/env python3
"""Module with technical alphas"""
import stbt.operators.technical as ti
import stbt.alphas.events as ev
from stbt.helpers import prolong, unpack_ohlc


def acceleration_event(ohlc_df, lookback=60, prolong_timeframe=240, threshold=0.1):
    '''alpha based of long term acceleration
    '''
    volume, opens, high, low, close = unpack_ohlc(ohlc_df)
    del volume, opens, high, low
    events = ev.get_acc_events(close, lookback, threshold)
    weights = events.rolling(prolong_timeframe).apply(prolong)
    weights.fillna(0, inplace=True)
    return weights


def vwap_event(ohlc_df,
               threshold=.01, vwap_timeframe=60,
               prolong_timeframe=240,
               multiplier=1):
    '''alpha based on close being too far from vwap
    '''
    volume, opens, high, low, close = unpack_ohlc(ohlc_df)
    del opens, high, low

    events = ev.get_vwap_events(close, volume, threshold, vwap_timeframe)
    weights = events.rolling(prolong_timeframe).apply(prolong)

    return weights * multiplier


def high_volume_small_candle_event(ohlc_df,
                                   lookback=60,
                                   threshold_volume=10,
                                   threshold_candle=1.5,
                                   prolong_timeframe=240,
                                   multiplier=1):
    '''alpha based on high volume with in the small candle
    '''
    volume, opens, high, low, close = unpack_ohlc(ohlc_df)
    del high, low

    volume_events = ev.get_high_volume_events(
        volume, lookback, threshold_volume)
    candle_events = ev.get_small_candle_body_events(
        close, opens, threshold_candle, lookback)
    events = volume_events.multiply(candle_events)
    weights = events.rolling(prolong_timeframe).apply(prolong)

    return weights * multiplier


def high_volume_big_candle_event(ohlc_df,
                                 lookback=60,
                                 threshold_volume=10,
                                 threshold_candle=.01,
                                 prolong_timeframe=240,
                                 multiplier=1,
                                 delay=None):
    '''alpha based on high volume with in the big candle
    '''
    volume, opens, high, low, close = unpack_ohlc(ohlc_df)
    del high, low

    volume_events = ev.get_high_volume_events(
        volume, lookback, threshold_volume)
    candle_events = ev.get_big_candle_body_events(
        close, opens, threshold_candle)
    events = volume_events.multiply(candle_events)
    weights = events.rolling(prolong_timeframe).apply(prolong)

    if delay:
        weights = ti.delay(weights, delay)

    return weights * multiplier


def high_volume_big_tail_event(ohlc_df,
                               lookback=60,
                               threshold_volume=10,
                               threshold_tail=1.005,
                               prolong_timeframe=240,
                               multiplier=1,
                               delay=None):
    '''alpha based on high volume with in the big candle
    '''
    volume, opens, high, low, close = unpack_ohlc(ohlc_df)

    volume_events = ev.get_high_volume_events(
        volume, lookback, threshold_volume)
    candle_events = ev.get_big_tail_events(
        close, high, low, opens, threshold_tail)
    events = volume_events.multiply(candle_events)
    weights = events.rolling(prolong_timeframe).apply(prolong)
    if delay:
        weights = ti.delay(weights, delay)

    return weights * multiplier


def high_volume_vwap_event(ohlc_df,
                           lookback_volume=1000,
                           threshold_volume=10,
                           threshold_vwap=1.005,
                           vwap_timeframe=60,
                           prolong_timeframe=240,
                           multiplier=1,
                           delay=None):
    '''alpha based on high volume and price being far from vwap
    '''
    volume, opens, high, low, close = unpack_ohlc(ohlc_df)
    del high, low, opens
    volume_events = ev.get_high_volume_events(
        volume, lookback_volume, threshold_volume)
    vwap_events = ev.get_vwap_events(
        close, volume, threshold_vwap, vwap_timeframe)
    events = volume_events.multiply(vwap_events)
    weights = events.rolling(prolong_timeframe).apply(prolong)
    if delay:
        weights = ti.delay(weights, delay)

    return weights * multiplier


def multiple_tails_event(ohlc_df,
                         threshold_tail=1.005,
                         count_timeframe=60,
                         tail_count=2,
                         prolong_timeframe=240,
                         multiplier=1,
                         delay=None):
    '''alpha based on multiple big tails in small count_timeframe
    '''
    volume, opens, high, low, close = unpack_ohlc(ohlc_df)
    del volume
    candle_events = ev.get_big_tail_events(
        close, high, low, opens, threshold_tail)
    events = ev.get_event_count_events(
        candle_events, tail_count, count_timeframe)
    weights = events.rolling(prolong_timeframe).apply(prolong)
    if delay:
        weights = ti.delay(weights, delay)

    return weights * multiplier


def multiple_high_volume_small_candle_event(ohlc_df,
                                            lookback_volume=60,
                                            threshold_volume=10,
                                            threshold_candle=1.5,
                                            count_timeframe=60,
                                            event_count=2,
                                            prolong_timeframe=240,
                                            multiplier=1,
                                            delay=None):
    '''alpha based on multiple big tails in small count_timeframe
    '''
    volume, opens, high, low, close = unpack_ohlc(ohlc_df)
    del high, low

    volume_events = ev.get_high_volume_events(
        volume, lookback_volume, threshold_volume)
    candle_events = ev.get_small_candle_body_events(
        close, opens, threshold_candle, lookback_volume)
    events_hvsc = volume_events.multiply(candle_events)
    events = ev.get_event_count_events(
        events_hvsc, event_count, count_timeframe)
    weights = events.rolling(prolong_timeframe).apply(prolong)
    if delay:
        weights = ti.delay(weights, delay)

    return weights * multiplier


def multiple_vwap_event(ohlc_df,
                        threshold=.01,
                        vwap_timeframe=60,
                        count_timeframe=60,
                        event_count=2,
                        prolong_timeframe=240,
                        multiplier=1,
                        delay=None):
    '''alpha based on multiple vwap events
    '''

    volume, opens, high, low, close = unpack_ohlc(ohlc_df)
    del opens, high, low

    vwap_events = ev.get_vwap_events(close, volume, threshold, vwap_timeframe)
    events = ev.get_event_count_events(
        vwap_events, event_count, count_timeframe)
    weights = events.rolling(prolong_timeframe).apply(prolong)
    if delay:
        weights = ti.delay(weights, delay)

    return weights * multiplier


def multiple_high_volume_small_candle_event_time(ohlc_df,
                                                 lookback_volume=60,
                                                 threshold_volume=10,
                                                 threshold_candle=1.5,
                                                 count_timeframe=60,
                                                 event_count=2,
                                                 prolong_timeframe=240,
                                                 multiplier=1,
                                                 start_time="10:00",
                                                 end_time="10:30",
                                                 delay=None):
    '''alpha based on multiple big tails in small count_timeframe
    '''
    volume, opens, high, low, close = unpack_ohlc(ohlc_df)
    del high, low

    volume_events = ev.get_high_volume_events(
        volume, lookback_volume, threshold_volume)
    candle_events = ev.get_small_candle_body_events(
        close, opens, threshold_candle, lookback_volume)
    events_hvsc = volume_events.multiply(candle_events)
    events = ev.get_event_count_events(
        events_hvsc, event_count, count_timeframe)
    events = ti.null_daytime_timeperiod(events, start_time, end_time)
    weights = events.rolling(prolong_timeframe).apply(prolong)
    if delay:
        weights = ti.delay(weights, delay)

    return weights * multiplier

def low_volume_big_candle_event(ohlc_df,
                                lookback=60,
                                threshold_volume=0,
                                threshold_candle=.01,
                                prolong_timeframe=240,
                                multiplier=1,
                                delay=None):
    '''alpha based on high volume with in the big candle
    '''
    volume, opens, high, low, close = unpack_ohlc(ohlc_df)
    del high, low

    volume_events = ev.get_low_volume_events(
        volume, lookback, threshold_volume)
    candle_events = ev.get_big_candle_body_events(
        close, opens, threshold_candle)
    events = volume_events.multiply(candle_events)
    weights = events.rolling(prolong_timeframe).apply(prolong)

    if delay:
        weights = ti.delay(weights, delay)

    return weights * multiplier
