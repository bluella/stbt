=========================
Simple trading backtester
=========================

Overview
========

Quantitative approach to trading is done via applying mathematical models to
various financial instruments. In order to get money for you strategy,
mathematical model beneath it should be sound. And to prove that this model
worth money one should do proper backtesting.
This project aims to provide easy and straitforward backtesting solution.

Relevance
=========

There are number of python projects for backtesting: `backtrader <https://github.com/backtrader/backtrader>`_,
`pyalgotrade <https://github.com/gbeced/pyalgotrade>`_, `zipline <https://github.com/quantopian/zipline>`_,
`rqalpha <https://github.com/ricequant/rqalpha>`_, etc.. When i was trying out them,
i was dissatisfied with one or more of the following: event driven,
unnecessary complex architecture, no support for trading multiple instruments
in convinient way, no proper performance evaluation, etc..
This project solves those issues at cost of not so wide functionality
compared to mentioned ones above.
Project is designed to be easily build on top of it.

Features
========

* Data manipulations are made with pandas.

* Backtesting operations are vector( no loops, not event driven).

* Extensive statistical evaluation of strategies.

* Number of visualizations embedded.

* Strategy robustness tests.

* API to work with OHLC data( download, prepare).

* Clean and straitforward project structure.

* PEP8 compliant code.

Installation
============

* Install via setup.py:

.. code-block:: bash

    git clone git@github.com:bluella/stbt.git
    cd stbt
    python setup.py install

* Install via pip:

.. code-block:: bash

    pip install stbt

* Run tests:

.. code-block:: bash

    pip install pytest
    pytest

Usage
=====

.. code-block:: python

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

Links
=====

    * `Pypi <https://pypi.org/project/stbt/>`_

    * `readthedocs <https://stbt.rtfd.io>`_

    * `GitHub <https://github.com/bluella/stbt>`_


Futher development
==================

    * Improve test coverage.

    * More API for data download.

    * More technical indicators.

    * Portfolio optimization tools.

Releases
========

See `CHANGELOG <https://github.com/bluella/stbt/blob/master/CHANGELOG.rst>`_.

License
=======

This project is licensed under the MIT License -
see the `LICENSE <https://github.com/bluella/stbt/blob/master/LICENSE.txt>`_ for details.
