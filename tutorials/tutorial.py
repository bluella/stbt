#!/usr/bin/env python3
"""Module show package usage"""

import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from stbt.simulator import Strategy
from stbt.download_ohlc.cryptocurrency import get_ohlc_cryptocompare
from stbt.indicators.technical import skewness

# get trading data from cryptocompare
BTC_TICKER = 'BTC'
ETH_TICKER = 'ETH'
USD_TICKER = 'USD'

END_DATE = dt.datetime(2018, 7, 1, 0, 0, 0)
START_DATE = dt.datetime(2018, 3, 1, 0, 0, 0)

OHLC_BTC = get_ohlc_cryptocompare(BTC_TICKER, USD_TICKER, START_DATE,
                                  end_date=END_DATE, interval_key='day')
OHLC_ETH = get_ohlc_cryptocompare(ETH_TICKER, USD_TICKER, START_DATE,
                                  end_date=END_DATE, interval_key='day')

# create dfs in format that Strategy requires
closes_df = pd.concat([OHLC_BTC['Close'], OHLC_ETH['Close']],
                      axis=1, keys=['BTC', 'ETH'])

# use imported indicator to create weights
weights_df = skewness(closes_df)

# create strategy
s = Strategy(closes_df, weights_df, cash=100)

# run backtest, robust tests, calculate stats
s.run_all(delay=2, verify_data_integrity=True, instruments_drop=None,
          commissions_const=0, capitalization=False)

# check strategy stats
print(s.stats_dict)

# save strategy to futher comparison
s.add_to_pnls_pool()

# plot pool correlation heatmap
heatmap_fig, corr_matrix = s.get_pool_heatmap()

plt.show()
