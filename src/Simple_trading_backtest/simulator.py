import pandas as pd
import numpy as np
import scipy as ss
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import seaborn as sns
import pickle
import operator

import stb.helpers as hf


class Strategy(object):
    ''' Class for various alpha backtesting'''

    def __init__(self, data_df, weights_df, cash=1.0):
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
        self.weights = weights_df.div(weights_df.abs().sum(axis=1), axis=0).fillna(0)
        self.instruments = []  # not in use
        self.cash = cash  # not in use, because reasons
        self.pool_file = 'alphas_pool.pickle'
        self.data_mistakes_dict = {
            'shape': 0,
            'index_type': 0,
            'duplicates': 0,
            'Nans': 0,
            'missed': 0,
            'dates_values': 0,
            'column_names': 0}

    def verify_data_integrity(self, frequency=None, *args):
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
        for i in ((len(self.data) - self.data.count()).values):
            if i != 0:
                self.data_mistakes_dict['Nans'] += 1
                print(len(self.data), self.data.count(), 'data')
        for i in ((self.weights.count() - len(self.weights)).values):
            if i != 0:
                self.data_mistakes_dict['Nans'] += 1
                inds = pd.isnull(self.weights).any(1).nonzero()[0]
                print(inds)

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
        # print(self.data.columns, self.weights.columns)
        if list(self.data.columns) != list(self.weights.columns):
            self.data_mistakes_dict['column_names'] += 1

        # mistakes assesment
        data_is_okay = True
        for key in self.data_mistakes_dict:
            if self.data_mistakes_dict[key] != 0:
                print('There are mistakes in data!!!!Danger ALERT!')
                print(self.data_mistakes_dict)
                data_is_okay = False
        if not data_is_okay:
            raise ValueError('Take a look at data passed to Strategy')
        else:
            print('Data in Strategy is okay, good to go')

        pass

    def simulate(self, delay=1, instruments_drop=None, commissions_const=0.0, capitalization=False, start_date=None, end_date=None, *args):
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
            simulate_data = simulate_data[(simulate_data.index > start_date) & (simulate_data.index < end_date)]
            simulate_weights = simulate_weights[(simulate_weights.index > start_date) & (simulate_weights.index < end_date)]

        #########################################################

        # delay
        #########################################################
        if delay:
            simulate_weights = simulate_weights.shift(delay)
        #########################################################

        # initialize everything
        #########################################################
        commissions = pd.DataFrame(np.zeros(len(simulate_data)), index=simulate_data.index, columns=['coms'])
        returns_df = pd.DataFrame()
        pnl = pd.DataFrame(np.zeros(len(simulate_data)), index=simulate_data.index, columns=['_'.join(simulate_data.columns)])
        inst_sum_returns = pd.DataFrame(np.zeros(len(simulate_data)), index=simulate_data.index)
        #########################################################

        # calculate commissions
        #########################################################
        weights_diff_df = simulate_weights.diff()
        # for column in weights_diff_df:
        # commissions = commissions.add(weights_diff_df[column], axis=0)
        if commissions_const > 0.0:
            commissions = commissions.add(weights_diff_df.sum(axis=1), axis=0)
            commissions = abs(commissions) * commissions_const
            commissions.fillna(0, inplace=True)

        # print(commissions)
        # hf.print_df_info(inst_sum_returns)
        #########################################################

        # calculate returns
        #########################################################
        returns_df = simulate_data.pct_change()
        returns_df.fillna(0, inplace=True)
        # returns_df.index = pd.to_datetime(returns_df.index)
        # hf.print_df_info(simulate_weights)
        #########################################################

        # calculate returns for all instruments with respect to weights and commissions
        #########################################################
        daily_returns_dist = ((returns_df) * simulate_weights)
        # print(daily_returns_dist)
        # for column in daily_returns_dist:
        #     inst_sum_returns = inst_sum_returns.add(daily_returns_dist[column], axis=0)
        inst_sum_returns = inst_sum_returns.add(daily_returns_dist.sum(axis=1), axis=0)
        inst_sum_returns.columns = ['_'.join(simulate_data.columns)]
        # print(inst_sum_returns)
        # commissions:
        # hf.print_df_info(inst_sum_returns)
        inst_sum_returns = inst_sum_returns.subtract(commissions[['coms']].values)
        # hf.print_df_info(inst_sum_returns)

        #########################################################

        # calculate pnl with respect to capitalization
        #########################################################
        if not capitalization:
            pnl = (inst_sum_returns * self.cash).cumsum() + self.cash
        else:
            # make initial return as price of one stock to see real Closes:
            # self.cash = ((simulate_data * simulate_weights).iloc[0].sum() - 1)  # optional: comment to use cash as it was set
            # and here come worst lines of this ptoject:
            fake_returns = inst_sum_returns.copy()
            fake_returns.iloc[0][0] = self.cash
            # print(inst_sum_returns.iloc[0][0])
            pnl[pnl.columns[0]] = ((fake_returns + 1).cumprod())

        pass
        #########################################################
        print('Simulation is finished')

        return {
            'pnl': pnl,
            'returns': inst_sum_returns,
            'coms': commissions,
            'capitalization': capitalization,
            'delay': delay,
            'commissions_const': commissions_const}

    def calculate_sim_stats(self, pnl, returns, *args):
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
            'Sharpe_90d': 0,
            # 'Max_Drawdown': 0,
        }

        # sharpe calculation:
        # periods = len(returns)
        sim_stats_dict['Sharpe'] = hf.get_sharpe(returns)

        # correlation with data
        sim_stats_dict['Correlation'] = pnl.corrwith(self.data)

        # 1day sharpe
        df_resampled_sharpe_1d = returns.resample('1d').apply(hf.get_sharpe)
        sim_stats_dict['Sharpe_1d'] = round(df_resampled_sharpe_1d.mean()[0], 1)
        # hf.print_df_info(df_resampled_sharpe_1d)
        # print(df_resampled_sharpe_1d)
        # df_resampled_sharpe_1d.plot()

        # 30days sharpe
        df_resampled_sharpe_30d = returns.resample('30d').apply(hf.get_sharpe)
        sim_stats_dict['Sharpe_30d'] = round(df_resampled_sharpe_30d.mean()[0], 1)
        # hf.print_df_info(df_resampled_sharpe_30d)
        # print(df_resampled_sharpe_30d)
        # df_resampled_sharpe_30d.plot()

        # 30days sharpe
        df_resampled_sharpe_90d = returns.resample('90d').apply(hf.get_sharpe)
        sim_stats_dict['Sharpe_90d'] = round(df_resampled_sharpe_90d.mean()[0], 1)
        # hf.print_df_info(df_resampled_sharpe_90d)
        # print(df_resampled_sharpe_90d)
        # df_resampled_sharpe_90d.plot()

        # total returns
        sim_stats_dict['Total_returns'] = str(round(returns.sum()[0] * 100, 2)) + '%'

        # avg_returns per period
        sim_stats_dict['Avg_returns'] = str(round(returns.mean()[0] * 100, 2)) + '%'

        # avg_returns per day
        returns_resampled_1d = returns.resample('1d').sum()
        sim_stats_dict['Avg_returns_1d'] = str(round(returns_resampled_1d.mean()[0] * 100, 2)) + '%'

        # avg_returns per month
        returns_resampled_30d = returns.resample('30d').sum()
        sim_stats_dict['Avg_returns_30d'] = str(round(returns_resampled_30d.mean()[0] * 100, 2)) + '%'

        # Max Drawdawn
        # print(returns.iloc[:, 0], type(returns.iloc[:, 0]))
        dd_tuple = hf.get_max_Drawdown(returns.iloc[:, 0])
        sim_stats_dict['Max_Drawdown'] = str(round(abs(dd_tuple[0]) * 100, 1)) + '%'

        # Daily Turnover
        weights_diff = abs(self.weights.diff())

        weights_diff_resampled = weights_diff.resample('1d').sum()
        turnover = str(round(weights_diff_resampled.mean()[0] * 100, 1)) + '%'

        sim_stats_dict['Turnover_1d'] = turnover

        turnover_resampled = weights_diff_resampled.resample('30d').mean() * 100

        # MAX CORR
        sim_stats_dict['Max_corr'] = self.get_max_corr(pnl)

        # plot stats
        stats_figure = plt.figure(tight_layout=True)

        ax = plt.subplot2grid((12, 1), (0, 0), rowspan=2, colspan=1)
        ax.plot(returns_resampled_30d.index.values, returns_resampled_30d.values)
        ax.plot(returns_resampled_30d.index.values, np.zeros(len(returns_resampled_30d)), 'r--')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mticker.MaxNLocator(5))
        plt.title('Avg_returns_30d')

        ax2 = plt.subplot2grid((12, 1), (3, 0), rowspan=2, colspan=1)
        ax2.plot(df_resampled_sharpe_30d.index.values, df_resampled_sharpe_30d.values)
        ax2.plot(df_resampled_sharpe_30d.index.values, np.zeros(len(df_resampled_sharpe_30d)), 'r--')
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax2.xaxis.set_major_locator(mticker.MaxNLocator(5))
        plt.title('Sharpe_30d')

        ax3 = plt.subplot2grid((12, 1), (6, 0), rowspan=2, colspan=1)
        ax3.plot(df_resampled_sharpe_90d.index.values, df_resampled_sharpe_90d.values)
        ax3.plot(df_resampled_sharpe_90d.index.values, np.zeros(len(df_resampled_sharpe_90d)), 'r--')
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax3.xaxis.set_major_locator(mticker.MaxNLocator(5))
        plt.title('Sharpe_90d')

        ax4 = plt.subplot2grid((12, 1), (9, 0), rowspan=2, colspan=1)
        ax4.plot(turnover_resampled.index.values, turnover_resampled.values)
        ax4.plot(turnover_resampled.index.values, np.zeros(len(turnover_resampled)), 'r--')
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax4.xaxis.set_major_locator(mticker.MaxNLocator(5))
        plt.title('Turnover_daily_30d_mean')

        # ffn to get all stats
        # Another way to get a lot of stats
        # but results are a little bit different
        # res = GroupStats(pnl.iloc[:, 0])
        # print(res.stats)
        # res.display()

        print('Statistics for simulation were collected:')
        # print(sim_stats_dict)

        return sim_stats_dict, stats_figure

    def plot_sim_results(self, pnl, returns, *args):
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

        figure = plt.figure(tight_layout=True)
        # dates_list = [dt.datetime.strptime(date, "%Y-%m-%d %H:%M:%S") for date in self.data.index.values]
        ax1 = plt.subplot2grid((12, 1), (0, 0), rowspan=3, colspan=1)
        ax1.plot(self.data.index.values, self.data.values)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.xaxis.set_major_locator(mticker.MaxNLocator(5))
        plt.title('_'.join(self.data.columns))
        # print(self.data.index.values[0], type(self.data.index.values[0]))
        # print(self.data['btc'].values, type(self.data['btc'].values))
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

        # returns_resampled_30d = returns.resample('30d').sum()
        # ax4 = plt.subplot2grid((12, 1), (9, 0), rowspan=2, colspan=1)
        # ax4.plot(returns_resampled_30d.index.values, returns_resampled_30d.values)
        # ax4.plot(returns_resampled_30d.index.values, np.zeros(len(returns_resampled_30d)), 'r--')
        # ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        # ax4.xaxis.set_major_locator(mticker.MaxNLocator(5))
        # plt.title('Returns')

        print('Graph with sim results was created')

        return figure

    def run_tests(self, *args):
        list_of_res_dicts = []
        tests = [
            {'delay': 1},
            {'delay': 2},
            {'delay': 3},
            {'delay': 2, 'commissions_const': 0.001},
        ]

        tests_figure = plt.figure(tight_layout=True)
        ax = plt.subplot2grid((12, 1), (0, 0), rowspan=12, colspan=1)

        test_number = 0
        for test in tests:
            list_of_res_dicts.append(self.simulate(**test))
            ax.plot(self.weights.index, list_of_res_dicts[-1]['pnl'], label='{}'.format(hf.get_label_from_dict(test)))
            test_number += 1
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mticker.MaxNLocator(5))
        plt.title('Tests_pnls')
        plt.legend()

        return list_of_res_dicts, tests_figure

    def run_all(self, delay=1, verify_data_integrity=True, instruments_drop=None, commissions_const=0, capitalization=False, start_date=None, end_date=None, *args):
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
        results_dict = self.simulate(instruments_drop=instruments_drop,
                                     commissions_const=commissions_const,
                                     capitalization=capitalization,
                                     delay=delay,
                                     start_date=start_date,
                                     end_date=end_date)
        # hf.print_df_info(results_dict['pnl'])

        self.plot_sim_results(results_dict['pnl'], results_dict['returns'])
        sim_stats_dict, stats_figure = self.calculate_sim_stats(results_dict['pnl'], results_dict['returns'])
        print(sim_stats_dict)
        self.run_tests()

    def get_alphas_pool(self, *args):
        with open(self.pool_file, 'rb') as f:
            pnls_df = pickle.load(f)

        return pnls_df

    def add_to_alphas_pool(self, pnl_df, *args):

        pnls_df = self.get_alphas_pool()
        if len(pnl_df) == len(pnls_df):
            pnls_df = pnls_df.join(pnl_df)
            with open(self.pool_file, 'wb') as f:
                pickle.dump(pnls_df, f)

        else:
            print('Length of dfs is inconsistent: cant save such pnls!!!!!!!')

        return pnls_df

    def get_pool_heatmap(self, *args):
        pnls_df = self.get_alphas_pool()
        corr = pnls_df.corr()

        figure = plt.figure()
        sns.heatmap(corr, annot=True)
        plt.title('Correlation heatmap')

        return figure, corr

    def get_max_corr(self, pnl, *args):
        corr_dict = {}
        try:
            pnls_df = self.get_alphas_pool()
            time_delta = pnl.index[1] - pnl.index[0]
            if time_delta != pd.Timedelta(1, 'h'):
                pnl = hf.resample(pnl, 'H')

            if len(pnl) > len(pnls_df):
                zero_df = pd.DataFrame(np.zeros(len(pnls_df)), index=pnls_df.index, columns=pnl.columns)
                pnl = zero_df + pnl
                pnl = pnl.dropna()
            if len(pnl) < len(pnls_df):
                zero_df = pd.DataFrame(np.zeros(len(pnls_df)), index=pnls_df.index, columns=pnl.columns)
                pnl = zero_df + pnl
                pnl = pnl.ffill()

            for column in pnls_df:
                corr_dict[column] = pnl.corrwith(pnls_df[column]).values[0]

            top_key = max(corr_dict.items(), key=operator.itemgetter(1))[0]
            res_list = [top_key, corr_dict[top_key]]
        except Exception as e:
            print(e)
            res_list = ['0', 0]

        return res_list


# Special functions:

def get_correlation(list_of_pnls, plot=True):
    pnl_df = pd.DataFrame()
    for i in range(len(list_of_pnls)):
        list_of_pnls[i].rename(columns={list_of_pnls[i].columns[0]: "{}".format(i)}, inplace=True)
        if len(pnl_df) < 1:
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
