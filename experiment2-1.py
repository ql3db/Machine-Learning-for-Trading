"""
Experiment 2
Name : Qi Li
UserID : qli385
GT ID: 903472876
"""


import datetime as dt
import pandas as pd
import util as ut
import random
import numpy as np
import StrategyLearner as st
import ManualStrategy as ms
from marketsimcode import compute_portvals2
from util import get_data, plot_data
import matplotlib.pyplot as plt
import indicators as ind

def author():
    return 'qli385'

def daily_return(port_prices):
    daily_returns = (port_prices/port_prices.shift(1)) - 1
    #get rid of first row
    daily_returns = daily_returns.iloc[1:] 
    #print(daily_returns)
    return daily_returns

def test_code():
    # setting the random seed
    np.random.seed(903472876)

    # input
    sd = dt.datetime(2008,1,1)
    ed = dt.datetime(2009, 12, 31)
    symbols = ['JPM']
    dates = pd.date_range(sd, ed)
    prices = ind.get_prices(symbols,dates)


    # Strategy Learner - impact = 0.0005
    learner = st.StrategyLearner(verbose = False, impact=0.0005)
    learner.addEvidence(symbol="JPM",sd=dt.datetime(2008,1,1),ed=dt.datetime(2009,12,31),sv=100000)
    test = learner.testPolicy(symbol="JPM",sd=dt.datetime(2008,1,1),ed=dt.datetime(2009,12,31),sv=100000)
    t = np.count_nonzero(test)
    #st_trades = trades_ST(test,'JPM')
    st_port_val = compute_portvals2(test,100000,0,0.0005)


    # Strategy Learner - impact = 0.005
    learner = st.StrategyLearner(verbose = False, impact=0.005)
    learner.addEvidence(symbol="JPM",sd=dt.datetime(2008,1,1),ed=dt.datetime(2009,12,31),sv=100000)
    test = learner.testPolicy(symbol="JPM",sd=dt.datetime(2008,1,1),ed=dt.datetime(2009,12,31),sv=100000)
    #st_trades = trades_ST(test,'JPM')
    t2 = np.count_nonzero(test)
    st_port_val2 = compute_portvals2(test,100000,0,0.005)


    # Strategy Learner - impact = 0.05
    learner = st.StrategyLearner(verbose = False, impact=0.05)
    learner.addEvidence(symbol="JPM",sd=dt.datetime(2008,1,1),ed=dt.datetime(2009,12,31),sv=100000)
    test = learner.testPolicy(symbol="JPM",sd=dt.datetime(2008,1,1),ed=dt.datetime(2009,12,31),sv=100000)
    #st_trades = trades_ST(test,'JPM')
    t3 = np.count_nonzero(test)
    st_port_val3 = compute_portvals2(test,100000,0,0.05)
    
    #normalize portfolio values
    st_port_val = st_port_val/st_port_val.iloc[0]
    st_port_val2 = st_port_val2/st_port_val2.iloc[0]
    st_port_val3 = st_port_val3/st_port_val3.iloc[0]


    # Plotting charts
    plt.figure(figsize=(14,8))
    plt.xlabel("Date",fontsize = 20)
    plt.ylabel("Normalized Port Value",fontsize = 20)
    start_date = dt.datetime(2008,1,1)
    end_date = dt.datetime(2009,12,31)
    plt.xlim(start_date,end_date)
    plt.title("Exp 2: Effect of market impact on portfolio value",fontsize=25)
    plt.plot(st_port_val, 'g', label ="impact=0.0005")
    plt.plot(st_port_val2,'r', label = "impact=0.005")
    plt.plot(st_port_val3, 'b', label = "impact=0.05")
    plt.legend(fontsize = 20)
    plt.savefig('exp2.png')
    
    '''
    st_cum_ret = st_port_val.iloc[-1]/st_port_val.iloc[0] - 1
    st_cum_ret2 = st_port_val2.iloc[-1]/st_port_val2.iloc[0] - 1
    st_cum_ret3 = st_port_val3.iloc[-1]/st_port_val3.iloc[0] - 1
    print("In-sample:")
    print("impact = 0.0005 Cumulative Return: ", st_cum_ret.item())
    print("impact = 0.005 Cumulative Return: ", st_cum_ret2.item())
    print("impact = 0.05 Cumulative Return: ", st_cum_ret3.item())
    #daily return std
    st_daily_ret = daily_return(st_port_val)
    st_daily_ret2 = daily_return(st_port_val2)
    st_daily_ret3 = daily_return(st_port_val3)
    print("impact = 0.0005 Daily Return Std:", st_daily_ret.std().item())
    print("impact = 0.005 Daily Return Std:", st_daily_ret2.std().item())
    print("impact = 0.05 Daily Return Std:", st_daily_ret3.std().item())
    #daily return mean
    print("impact = 0.0005 Daily Return Mean:", st_daily_ret.mean().item())
    print("impact = 0.005 Daily Return Mean:", st_daily_ret2.mean().item())
    print("impact = 0.05 Daily Return Mean:", st_daily_ret3.mean().item())
    #Sharpe
    print("impact = 0.0005 Sharpe Ratio:", (252**0.5)*st_daily_ret.mean().item()/st_daily_ret.std().item())
    print("impact = 0.005 Sharpe Ratio:", (252**0.5)*st_daily_ret2.mean().item()/st_daily_ret2.std().item())
    print("impact = 0.05 Sharpe Ratio:", (252**0.5)*st_daily_ret3.mean().item()/st_daily_ret3.std().item())
    #trades
    print("impact = 0.0005 trades:", t)
    print("impact = 0.005 trades:", t2)
    print("impact = 0.05 trades:", t3)
    '''
    
if __name__=="__main__":
    test_code()