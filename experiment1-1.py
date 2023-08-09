"""
Experiment 1
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
    #print(prices)

    # Strategy Learner
    learner = st.StrategyLearner(verbose = False, impact=0.0)
    learner.addEvidence(symbol="JPM",sd=dt.datetime(2008,1,1),ed=dt.datetime(2009,12,31),sv=100000)
    test = learner.testPolicy(symbol="JPM",sd=dt.datetime(2008,1,1),ed=dt.datetime(2009,12,31),sv=100000)
    test_df = test.copy()
    st_port_val = compute_portvals2(test_df,100000,9.95,0.005)
    #out of sample
    test_os = learner.testPolicy(symbol="JPM",sd=dt.datetime(2010,1,1),ed=dt.datetime(2011,12,31),sv=100000)
    test_os_df = test_os.copy()
    st_port_val_os = compute_portvals2(test_os_df,100000,9.95,0.005)
    
    
    # Benchmark
    #bench = pd.DataFrame(columns=['Date', 'Symbol', 'Order', 'Shares'])
    #bench.loc[0] = [prices_all.index[0].strftime('%Y-%m-%d'),'JPM','BUY',1000]
    #bench.loc[1] = [prices_all.index[-1].strftime('%Y-%m-%d'),'JPM','SELL',1000]
    bench_orders = test.copy()
    bench_orders.iloc[:,:] = 0
    bench_orders.iloc[0] = 1000
    #print("bench_orders",bench_orders)
    bench_port_val = compute_portvals2(bench_orders,100000,9.95,0.005)
    #out of sample
    #becn_orders_os = test_os.copy()
    #becn_orders_os.iloc[:,:] = 0
    #becn_orders_os.iloc[0] = 1000
    #bench_port_val_os = compute_portvals2(becn_orders_os,100000,9.95,0.005)

    # ManualStrategy
    ms_orders= ms.testPolicy(symbol='JPM', sd=sd ,ed=ed, sv=100000)
    #print("ms_orders",ms_orders)
    ms_orders_df = ms_orders.copy()
    ms_port_val = compute_portvals2(ms_orders_df,100000,9.95,0.005)
    #out of sample
    ms_orders_os = ms.testPolicy(symbol="JPM",sd=dt.datetime(2010,1,1),ed=dt.datetime(2011,12,31),sv=100000)
    ms_orders_os_df = ms_orders_os.copy()
    ms_port_val_os = compute_portvals2(ms_orders_os_df,100000,9.95,0.005)
    
    #normalize portfolio values
    bench_port_val = bench_port_val/bench_port_val.iloc[0]
    ms_port_val = ms_port_val/ms_port_val.iloc[0]
    st_port_val = st_port_val/st_port_val.iloc[0]
    #out of sample
    #bench_port_val_os = bench_port_val_os/bench_port_val_os.iloc[0]
    ms_port_val_os = ms_port_val_os/ms_port_val_os.iloc[0]
    st_port_val_os = st_port_val_os/st_port_val_os.iloc[0]    
    

    # Plotting charts
    #in sample benchmark vs. manual strategy LONG/SHORT
    plt.figure(figsize=(14,8))
    plt.xlabel("Date",fontsize = 20)
    plt.ylabel("Normalized Port Value",fontsize = 20)
    start_date = dt.datetime(2008,1,1)
    end_date = dt.datetime(2009,12,31)
    plt.xlim(start_date,end_date)
    plt.title("Exp 1: In-Sample Strategy Learner vs Manual Strategy vs Bench",fontsize=25)
    plt.plot(bench_port_val, 'g', label ="Benchmark",linewidth=1.5)
    plt.plot(ms_port_val,'r', label = "Manual Strategy", linewidth=1.5)
    plt.plot(st_port_val,'b', label = "Strategy Learner", linewidth=1.5)
    plt.legend(fontsize = 20)
    plt.savefig('exp1.png')
    
    
    '''
    #table
    port_cum_ret = ms_port_val.iloc[-1]/ms_port_val.iloc[0] - 1
    st_cum_ret = st_port_val.iloc[-1]/st_port_val.iloc[0] - 1
    bench_cum_ret = bench_port_val.iloc[-1]/bench_port_val.iloc[0] - 1
    print("In-sample:")
    print("Manual Strategy Cumulative Return: ", port_cum_ret.item())
    print("Strategy Learner Cumulative Return: ", st_cum_ret.item())
    print("Benchmark Cumulative Return: ", bench_cum_ret.item())
    #daily return std
    port_daily_ret = daily_return(ms_port_val)
    st_daily_ret = daily_return(st_port_val)
    bench_daily_ret = daily_return(bench_port_val)
    print("Manual Strategy Daily Return Std:", port_daily_ret.std().item())
    print("Strategy Learner Daily Return Std:", st_daily_ret.std().item())
    print("Benchmark Daily Return Std:", bench_daily_ret.std().item())
    #daily return mean
    print("Manual Strategy Daily Return Mean:", port_daily_ret.mean().item())
    print("Strategy Learner Daily Return Mean:", st_daily_ret.mean().item())
    print("Benchmark Daily Return Mean:", bench_daily_ret.mean().item())
    #Sharpe
    print("Manual Strategy Sharpe Ratio:", (252**0.5)*port_daily_ret.mean().item()/port_daily_ret.std().item())
    print("Strategy Learner Sharpe Ratio:", (252**0.5)*st_daily_ret.mean().item()/st_daily_ret.std().item())
    print("Benchmark Sharpe Ratio:", (252**0.5)*bench_daily_ret.mean().item()/bench_daily_ret.std().item())
    '''
    
if __name__=="__main__":
    test_code()