=========================
Simple trading backtester
=========================

Overview
========

Quantitative approach to trading is done via applying mathematical models to
various financial instruments. In order to get money for you strategy,
mathematical model beneath it should be sound. And to prove that this model
worth money one should do proper backtesting.
This project aims to provide easy and straitforward backtesting oportunities.

Relevance
=========

There are number of python projects for backtesting: `backtrader <https://github.com/backtrader/backtrader>`_,
`pyalgotrade <https://github.com/gbeced/pyalgotrade>`_, `zipline <https://github.com/quantopian/zipline>`_,
`rqalpha <https://github.com/ricequant/rqalpha>`_, etc.. When i was trying out them,
i was dissatisfied with one or more of the following: event driven,
unnecessary complex architecture, no support for trading multiple instruments
in convinient way, no proper performance evaluation, bugs, etc..
This project solves those issues at cost of not so wide functionality
compared to mentioned ones above.
Project is designed to be easily build on top of it.

Features
========

* Data manipulations are made with pandas

* Backtesting operations are vector(no loops, no event driven)

* Extensive statistical evaluation of strategies

* Number of visualizations embedded

* Clean and straitforward project structure

* Strategy robustness tests

* OHLC data checking and preparation tools

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
    from Simple_trading_backtest.simulator import Strategy
    import Simple_trading_backtest.helpers as hlp

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


Links
=====

    * `Pypi <https://pypi.org/project/stbt/>`_

    * `readthedocs <stbt.rtfd.io>`_

    * `GitHub <https://github.com/bluella/stbt>`_


Futher development
==================

    * Improve test coverage

    * API for data download

    * Technical indicators support

    * Portfolio optimization tools

Releases
========

See `CHANGELOG <https://github.com/bluella/stbt/blob/master/CHANGELOG.rst>`_

License
=======

This project is licensed under the MIT License -
see the `LICENSE <https://github.com/bluella/stbt/blob/master/LICENSE.txt>`_ for details.
