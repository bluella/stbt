#!/usr/bin/env python3
"""Module to do custom testing"""

import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from stbt.simulator import Strategy
# import Simple_trading_backtest.helpers as hlp

# creating fake trading data

# prepare datetime index
some_date = dt.datetime(2017, 1, 1)
days = pd.date_range(some_date, some_date + dt.timedelta(30), freq='D')

# initialize close prices
data_values = list(range(1, 32))
closes_df = pd.DataFrame({'Date': days, 'inst1': data_values})
closes_df.set_index('Date', inplace=True)

# initialize weights
weights_values = [1 for i in range(31)]
weights_df = pd.DataFrame({'Date': days, 'inst1': weights_values})
weights_df.set_index('Date', inplace=True)

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
