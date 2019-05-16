#!/usr/bin/env python3

import pytest
import pandas as pd
from Simple_trading_backtest.helpers import get_sharpe
from Simple_trading_backtest.simulator import Strategy

__author__ = "Igor Grigorev"
__copyright__ = "Igor Grigorev"
__license__ = "mit"

def test_get_sharpe():
    test_df = pd.DataFrame({'pnl':[1,2,3]})
    print(get_sharpe(test_df))
    assert get_sharpe(test_df) == 3.46

