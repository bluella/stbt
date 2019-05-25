#!/usr/bin/env python3
"""Module with all the tests"""

import unittest
import datetime as dt
import pandas as pd
from stbt.helpers import get_sharpe, get_max_drawdown
from Simple_trading_backtest.simulator import Strategy

class TestStrategy(unittest.TestCase):
    """Class to test all Strategy methods"""
    @classmethod
    def setUpClass(cls):
        """Prepare data for futher tests"""
        cls.sim_results = {}

        # prepare datetime index
        some_date = dt.datetime(2017, 1, 1)
        days = pd.date_range(some_date, some_date + dt.timedelta(30), freq='D')

        # initialize close prices
        data_values = list(range(1, 32))
        cls.closes_df = pd.DataFrame({'Date': days, 'inst1': data_values})
        cls.closes_df.set_index('Date', inplace=True)

        # initialize weights
        weights_values = [1 for i in range(31)]
        cls.weights_df = pd.DataFrame({'Date': days, 'inst1': weights_values})
        cls.weights_df.set_index('Date', inplace=True)

        # create strategy
        cls.test_strategy = Strategy(cls.closes_df, cls.weights_df, cash=100)

        cls.sim_results = cls.test_strategy.backtest()

    def test_constructor(self):
        """Test to check Strategy __init__ method"""
        assert len(self.test_strategy.weights) == len(self.weights_df)

    def test_verify_data_integrity(self):
        """Test to check Strategy verify_data_integrity method
        P.S. method will raise error if data is not okay"""
        self.test_strategy.verify_data_integrity()

    def test_backtest(self):
        """Test to check Strategy backtest method"""
        assert self.sim_results['pnl'].values[2] == 250

    def test_plot_sim_results(self):
        """Test to check Strategy verify_data_integrity method
        P.S. method will raise error if data is not okay"""
        self.test_strategy.plot_sim_results(self.sim_results['pnl'])
        assert str(self.test_strategy.strategy_figure) == "Figure(640x480)"

    def test_calculate_sim_stats(self):
        """Test to check Strategy calculate_sim_stats method"""
        sim_stats_dict = self.test_strategy.calculate_sim_stats(self.sim_results['pnl'],
                                                                self.sim_results['returns'])
        assert sim_stats_dict['Avg_returns'] == "12.89%"

    def test_run_tests(self):
        """Test to check Strategy run_tests method"""
        self.test_strategy.run_tests()
        assert str(self.test_strategy.tests_figure) == "Figure(640x480)"

    def test_get_sharpe(self):
        """Test to check get_sharpe function"""
        test_df = pd.DataFrame({'pnl':[1, 2, 3]})
        assert get_sharpe(test_df) == 3.46

    def test_get_max_drawdown(self):
        """Test to check get_max_drawdown function"""
        test_df = pd.Series([0.01, -0.02, 0.03])
        drawdown_dict = get_max_drawdown(test_df)
        assert round(drawdown_dict[0], 2) == -0.02
