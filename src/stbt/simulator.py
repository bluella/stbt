#!/usr/bin/env python3
"""Module with Strategy class to all
   backtest related manipulations"""
import logging
import pickle
import operator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import seaborn as sns
import stb.helpers as hf

# logging
###########################################################################
log_format = '%(asctime)s: %(filename)s: %(funcName)s: %(message)s'
formatter = logging.Formatter(log_format, datefmt='%b %d %H:%M:%S')
syslog = logging.StreamHandler()
syslog.setFormatter(formatter)

logger = logging.getLogger('simulator')
logger.addHandler(syslog)
logger.setLevel(logging.INFO)
###########################################################################


class Strategy(object):
    ''' Class for various backtesting'''

    def __init__(self, data_df, weights_df, pool_file='strategy_pool.pickle', cash=1.0):
        """
Initialize data, weights and cash, they are same for all methods.
Parameters
----------
data_df : df
    Close prices of instruments should be columns.
weights_df : df
    Money distribution for every day, same form with data_df
cash : float64
    Starting capital and returns multiplier
Returns
-------
None

"""
        self.data = data_df
        # scaling weights, sum of absolute values is one for every row
        self.weights = weights_df.div(
            weights_df.abs().sum(axis=1), axis=0).fillna(0)
        self.instruments = []  # not in use
        self.cash = cash  # not in use, because reasons
        self.pool_file = pool_file
        self.pnl = None
        self.stats_dict = {}
        self.data_mistakes_dict = {
            'shape': 0,
            'index_type': 0,
            'duplicates': 0,
            'Nans': 0,
            'missed': 0,
            'dates_values': 0,
            'column_names': 0}
        self.stats_figure = None
        self.strategy_figure = None
        self.tests_figure = None

    def verify_data_integrity(self, frequency=None):
        """
Check data for mistakes
Parameters
----------
frequency - e.g. 'D', 'W', 'M', if None - do not resample data

Returns
-------
None

"""
        # lens
        if len(self.data) != len(self.weights):
            self.data_mistakes_dict['shape'] += 1
        if len(self.data.columns) != len(self.weights.columns):
            self.data_mistakes_dict['shape'] += 1

        # index type
        if not isinstance(self.data.index, pd.DatetimeIndex):
            self.data_mistakes_dict['index_type'] += 1
        if not isinstance(self.weights.index, pd.DatetimeIndex):
            self.data_mistakes_dict['index_type'] += 1

        # duplicates
        if len(self.weights[self.weights.index.duplicated()]) >= 1:
            self.data_mistakes_dict['duplicates'] += 1
        if len(self.data[self.data.index.duplicated()]) >= 1:
            self.data_mistakes_dict['duplicates'] += 1

        # NaNs
        for i in (len(self.data) - self.data.count()).values:
            if i != 0:
                self.data_mistakes_dict['Nans'] += 1
        for i in (self.weights.count() - len(self.weights)).values:
            if i != 0:
                self.data_mistakes_dict['Nans'] += 1
                inds = pd.isnull(self.weights).any(1).nonzero()[0]
                logger.debug('wrong indexes: {}'.format(inds))

        # missed
        if frequency:
            # fill gaps in data
            df_reindexed = self.data.reindex(self.data.date_range(start=self.data.index.min(),
                                                                  end=self.data.index.max(),
                                                                  freq=frequency))

            df_reindexed.fillna(method='ffill', inplace=True)
            if len(df_reindexed) - len(self.data) >= 1:
                self.data_mistakes_dict['missed'] += 1

        # dates values
        if self.data.index[0] != self.weights.index[0]:
            self.data_mistakes_dict['dates_values'] += 1
        if self.data.index[-1] != self.weights.index[-1]:
            self.data_mistakes_dict['dates_values'] += 1

        # same columns
        if list(self.data.columns) != list(self.weights.columns):
            self.data_mistakes_dict['column_names'] += 1

        # mistakes assesment
        data_is_okay = True
        for key in self.data_mistakes_dict:
            if self.data_mistakes_dict[key] != 0:
                logger.error('There are mistakes in data!, self.data_mistakes_dict: {}'
                             .format(self.data_mistakes_dict))
                data_is_okay = False
        if not data_is_okay:
            raise ValueError('Take a look at data passed to Strategy')
        else:
            logger.debug('Data in Strategy is okay, good to go')

    def backtest(self, delay=1, instruments_drop=None, commissions_const=0.0, capitalization=False,
                 start_date=None, end_date=None):
        """
Method to calculate returns and pnl
Parameters
----------
instruments_drop : list of strings
    Columns with such names will be droped from data and weights
commissions_const : float64
    Fee paid for every transaction: 0.01 is 1% fee for every trade
capitalization : Boolean
    If money should be reinvested every time
Returns
-------
dict with pnl, returns, commissions dataframes

"""

        # initialize local data for simulate
        #########################################################
        simulate_data = self.data.copy()
        simulate_weights = self.weights.copy()
        #########################################################

        # filter instruments
        #########################################################
        if instruments_drop is None:
            pass
        else:
            simulate_data.drop(columns=instruments_drop, inplace=True)
            simulate_weights.drop(columns=instruments_drop, inplace=True)
        #########################################################

        # filter time
        #########################################################
        if (start_date is None) or (end_date is None):
            pass
        else:
            simulate_data = simulate_data[(simulate_data.index > start_date) & (
                simulate_data.index < end_date)]
            simulate_weights = simulate_weights[(simulate_weights.index > start_date) & (
                simulate_weights.index < end_date)]

        #########################################################

        # delay
        #########################################################
        if delay:
            simulate_weights = simulate_weights.shift(delay)
        #########################################################

        # initialize everything
        #########################################################
        commissions = pd.DataFrame(
            np.zeros(len(simulate_data)), index=simulate_data.index, columns=['coms'])
        returns_df = pd.DataFrame()
        pnl = pd.DataFrame(np.zeros(len(simulate_data)), index=simulate_data.index,
                           columns=['_'.join(simulate_data.columns)])
        inst_sum_returns = pd.DataFrame(
            np.zeros(len(simulate_data)), index=simulate_data.index)
        #########################################################

        # calculate commissions
        #########################################################
        weights_diff_df = simulate_weights.diff()
        if commissions_const > 0.0:
            commissions = commissions.add(weights_diff_df.sum(axis=1), axis=0)
            commissions = abs(commissions) * commissions_const
            commissions.fillna(0, inplace=True)
        #########################################################

        # calculate returns
        #########################################################
        returns_df = simulate_data.pct_change()
        returns_df.fillna(0, inplace=True)
        #########################################################

        # calculate returns for all instruments with respect to weights and commissions
        #########################################################
        daily_returns_dist = ((returns_df) * simulate_weights)
        inst_sum_returns = inst_sum_returns.add(
            daily_returns_dist.sum(axis=1), axis=0)
        inst_sum_returns.columns = ['_'.join(simulate_data.columns)]
        inst_sum_returns = inst_sum_returns.subtract(
            commissions[['coms']].values)
        #########################################################

        # calculate pnl with respect to capitalization
        #########################################################
        if not capitalization:
            pnl = (inst_sum_returns * self.cash).cumsum() + self.cash
        else:
            fake_returns = inst_sum_returns.copy()
            fake_returns.iloc[0][0] = self.cash
            pnl[pnl.columns[0]] = ((fake_returns + 1).cumprod())

        #########################################################
        logger.debug('Strategy was backtested')

        return {
            'pnl': pnl,
            'returns': inst_sum_returns,
            'coms': commissions,
            'capitalization': capitalization,
            'delay': delay,
            'commissions_const': commissions_const
        }

    def calculate_sim_stats(self, pnl, returns):
        """
Method to calculate vatious statistics of simulation
Parameters
----------
pnl : df
with one column
returns : df
with one column

Returns
-------
dict sim_stats_dict with great deal of stats

"""

        sim_stats_dict = {
            'start_date': str(returns.index[0]),
            'end_date': str(returns.index[-1]),
            'Sharpe': 0,
            'Sharpe_1d': 0,
            'Sharpe_30d': 0,
            'Sharpe_90d': 0
        }

        # sharpe calculation:
        sim_stats_dict['Sharpe'] = hf.get_sharpe(returns)

        # correlation with data
        sim_stats_dict['Correlation'] = pnl.corrwith(self.data)

        # 1day sharpe
        df_resampled_sharpe_1d = returns.resample('1d').apply(hf.get_sharpe)
        sim_stats_dict['Sharpe_1d'] = round(
            df_resampled_sharpe_1d.mean()[0], 1)

        # 30days sharpe
        df_resampled_sharpe_30d = returns.resample('30d').apply(hf.get_sharpe)
        sim_stats_dict['Sharpe_30d'] = round(
            df_resampled_sharpe_30d.mean()[0], 1)

        # 30days sharpe
        df_resampled_sharpe_90d = returns.resample('90d').apply(hf.get_sharpe)
        sim_stats_dict['Sharpe_90d'] = round(
            df_resampled_sharpe_90d.mean()[0], 1)

        # total returns
        sim_stats_dict['Total_returns'] = str(
            round(returns.sum()[0] * 100, 2)) + '%'

        # avg_returns per period
        sim_stats_dict['Avg_returns'] = str(
            round(returns.mean()[0] * 100, 2)) + '%'

        # avg_returns per day
        returns_resampled_1d = returns.resample('1d').sum()
        sim_stats_dict['Avg_returns_1d'] = str(
            round(returns_resampled_1d.mean()[0] * 100, 2)) + '%'

        # avg_returns per month
        returns_resampled_30d = returns.resample('30d').sum()
        sim_stats_dict['Avg_returns_30d'] = str(
            round(returns_resampled_30d.mean()[0] * 100, 2)) + '%'

        # Max Drawdawn
        dd_tuple = hf.get_max_Drawdown(returns.iloc[:, 0])
        sim_stats_dict['Max_Drawdown'] = str(
            round(abs(dd_tuple[0]) * 100, 1)) + '%'

        # Daily Turnover
        weights_diff = abs(self.weights.diff())

        weights_diff_resampled = weights_diff.resample('1d').sum()
        turnover = str(round(weights_diff_resampled.mean()[0] * 100, 1)) + '%'

        sim_stats_dict['Turnover_1d'] = turnover

        turnover_resampled = weights_diff_resampled.resample(
            '30d').mean() * 100

        # MAX CORR
        sim_stats_dict['Max_corr'] = self.get_max_corr(pnl)

        # plot stats
        self.stats_figure = plt.figure(tight_layout=True)

        ax = plt.subplot2grid((12, 1), (0, 0), rowspan=2, colspan=1)
        ax.plot(returns_resampled_30d.index.values,
                returns_resampled_30d.values)
        ax.plot(returns_resampled_30d.index.values,
                np.zeros(len(returns_resampled_30d)), 'r--')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mticker.MaxNLocator(5))
        plt.title('Avg_returns_30d')

        ax2 = plt.subplot2grid((12, 1), (3, 0), rowspan=2, colspan=1)
        ax2.plot(df_resampled_sharpe_30d.index.values,
                 df_resampled_sharpe_30d.values)
        ax2.plot(df_resampled_sharpe_30d.index.values,
                 np.zeros(len(df_resampled_sharpe_30d)), 'r--')
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax2.xaxis.set_major_locator(mticker.MaxNLocator(5))
        plt.title('Sharpe_30d')

        ax3 = plt.subplot2grid((12, 1), (6, 0), rowspan=2, colspan=1)
        ax3.plot(df_resampled_sharpe_90d.index.values,
                 df_resampled_sharpe_90d.values)
        ax3.plot(df_resampled_sharpe_90d.index.values,
                 np.zeros(len(df_resampled_sharpe_90d)), 'r--')
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax3.xaxis.set_major_locator(mticker.MaxNLocator(5))
        plt.title('Sharpe_90d')

        ax4 = plt.subplot2grid((12, 1), (9, 0), rowspan=2, colspan=1)
        ax4.plot(turnover_resampled.index.values, turnover_resampled.values)
        ax4.plot(turnover_resampled.index.values,
                 np.zeros(len(turnover_resampled)), 'r--')
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax4.xaxis.set_major_locator(mticker.MaxNLocator(5))
        plt.title('Turnover_daily_30d_mean')

        logger.debug('Statistics for strategy were calculated:')

        return sim_stats_dict

    def plot_sim_results(self, pnl):
        """
Method to visualize simulation
Parameters
----------
pnl : df
with one column
returns : df
with one column

Returns
-------
figure with 4 plots:data, pnl, weights, returns

"""

        self.strategy_figure = plt.figure(tight_layout=True)

        ax1 = plt.subplot2grid((12, 1), (0, 0), rowspan=3, colspan=1)
        ax1.plot(self.data.index.values, self.data.values)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.xaxis.set_major_locator(mticker.MaxNLocator(5))
        plt.title('_'.join(self.data.columns))

        ax2 = plt.subplot2grid((12, 1), (3, 0), rowspan=6, colspan=1)
        ax2.plot(pnl.index.values, pnl.values)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax2.xaxis.set_major_locator(mticker.MaxNLocator(5))
        plt.title('PnL')

        ax3 = plt.subplot2grid((12, 1), (9, 0), rowspan=3, colspan=1)
        ax3.plot(self.weights.index.values, self.weights.values)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax3.xaxis.set_major_locator(mticker.MaxNLocator(5))
        plt.title('Weights')

        logger.debug('Graph with backtest results was created')

    def run_tests(self):
        """Method to check strategy robusness against time and comissions"""
        list_of_res_dicts = []
        tests = [
            {'delay': 1},
            {'delay': 2},
            {'delay': 3},
            {'delay': 2, 'commissions_const': 0.001},
        ]

        self.tests_figure = plt.figure(tight_layout=True)
        ax = plt.subplot2grid((12, 1), (0, 0), rowspan=12, colspan=1)

        test_number = 0
        for test in tests:
            list_of_res_dicts.append(self.backtest(**test))
            ax.plot(self.weights.index,
                    list_of_res_dicts[-1]['pnl'], label='{}'.format(hf.get_label_from_dict(test)))
            test_number += 1
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mticker.MaxNLocator(5))
        plt.title('Tests_pnls')
        plt.legend()

        return list_of_res_dicts

    def run_all(self, delay=1, verify_data_integrity=True, instruments_drop=None,
                commissions_const=0, capitalization=False, start_date=None, end_date=None):
        """
Method to get all info about strategy
Parameters
----------
instruments_drop : list of strings
    Columns with such names will be droped from data and weights
commissions_const : float64
    Fee paid for every transaction: 0.01 is 1% fee for every trade
capitalization : Boolean
    If money should be reinvested every time
Returns
-------
None, just prints info and draws graphs

"""
        if verify_data_integrity:
            self.verify_data_integrity()
        results_dict = self.backtest(instruments_drop=instruments_drop,
                                     commissions_const=commissions_const,
                                     capitalization=capitalization,
                                     delay=delay,
                                     start_date=start_date,
                                     end_date=end_date)

        self.plot_sim_results(results_dict['pnl'])
        self.pnl = results_dict['pnl']
        self.stats_dict = self.calculate_sim_stats(results_dict['pnl'], results_dict['returns'])
        logger.debug(str(self.stats_dict))
        self.run_tests()

    def get_pnls_pool(self):
        """Method to read all pnls from self.pool_file"""
        with open(self.pool_file, 'rb') as f:
            pnls_df = pickle.load(f)

        return pnls_df

    def add_to_pnls_pool(self, pnl_df=None):
        """Method to add pnls to self.pool_file"""
        if not pnl_df:
            pnl_df = self.pnl
        try:
            pnls_df = self.get_pnls_pool()
            if len(pnl_df) == len(pnls_df):
                pnls_df = pnls_df.join(pnl_df)
                with open(self.pool_file, 'wb') as f:
                    pickle.dump(pnls_df, f)
            else:
                logger.error('Length of dfs is inconsistent: cant save such pnls!')

        except FileNotFoundError:
            pnls_df = pnl_df
            with open(self.pool_file, 'wb') as f:
                pickle.dump(pnls_df, f)

        return pnls_df

    def get_pool_heatmap(self):
        """Method to visualize self.pool_file"""
        pnls_df = self.get_pnls_pool()
        corr = pnls_df.corr()

        figure = plt.figure()
        sns.heatmap(corr, annot=True)
        plt.title('Correlation heatmap')

        return figure, corr

    def get_max_corr(self, pnl):
        """Method to get highest correlation with pnl from self.pool_file"""
        corr_dict = {}
        try:
            pnls_df = self.get_pnls_pool()
            time_delta = pnl.index[1] - pnl.index[0]
            if time_delta != pd.Timedelta(1, 'h'):
                pnl = hf.resample(pnl, 'H')

            if len(pnl) > len(pnls_df):
                zero_df = pd.DataFrame(
                    np.zeros(len(pnls_df)), index=pnls_df.index, columns=pnl.columns)
                pnl = zero_df + pnl
                pnl = pnl.dropna()
            if len(pnl) < len(pnls_df):
                zero_df = pd.DataFrame(
                    np.zeros(len(pnls_df)), index=pnls_df.index, columns=pnl.columns)
                pnl = zero_df + pnl
                pnl = pnl.ffill()

            for column in pnls_df:
                corr_dict[column] = pnl.corrwith(pnls_df[column]).values[0]

            top_key = max(corr_dict.items(), key=operator.itemgetter(1))[0]
            res_list = [top_key, corr_dict[top_key]]
        except BaseException as e:
            logger.error(e)
            res_list = ['0', 0]

        return res_list


# Special functions:

def get_correlation(list_of_pnls, plot=True):
    """Function to get correlation heatmap"""
    pnl_df = pd.DataFrame()
    for i in range(len(list_of_pnls)):
        list_of_pnls[i].rename(
            columns={list_of_pnls[i].columns[0]: "{}".format(i)}, inplace=True)
        if pnl_df.empty:
            pnl_df = list_of_pnls[i]
        else:
            pnl_df = pnl_df.join(list_of_pnls[i])

    corr = pnl_df.corr()

    figure = None
    if plot:
        figure = plt.figure()
        sns.heatmap(corr, annot=True)
        plt.title('Correlation heatmap')

    return corr, figure
