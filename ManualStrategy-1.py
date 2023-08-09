import numpy as np
import pandas as pd
import random as rand
import matplotlib.pyplot as plt
import indicators as ind
import marketsimcode as ms
from marketsimcode import compute_portvals2
import datetime as dt
import util as ut
import StrategyLearner as st
from util import get_data, plot_data

def author():
    return 'qli385'


def get_orders_from_holdings(holdings):
    orders = holdings - holdings.shift(1)
    orders.iloc[0] = holdings.iloc[0]
    return orders


def testPolicy(symbol='JPM', sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv=100000):
    #returns a trade dataframe
    
    #use indicators: price/SMA, BB%, stochastic oscillator
    window = 14
    start_window = 10
    sd_start = sd - dt.timedelta(days=window+start_window) #so real start dates have data
    dates = pd.date_range(sd_start, ed)
    symbols = [symbol]
    if (isinstance(sd, dt.datetime)):
        sd_str = sd.strftime("%Y-%m-%d")
    
    #price/SMA
    prices = ind.get_prices(symbols,dates)
    sma_df = ind.sma(prices, window)
    price_sma_df = ind.price_sma(prices, sma_df, window)
    price_sma_df = price_sma_df.rename(columns={"JPM":"Price/SMA"})
    final_dates = pd.date_range(sd, ed)
    #get only the time period interested
    price_sma_df = price_sma_df.iloc[price_sma_df.index >=sd_str]
    #print("price/SMA:\n",price_sma_df)
    
    #BB%
    top_band, bottom_band,bbp = ind.bbd(prices,sma_df,window)
    bbp = bbp.rename(columns={"JPM":"BBP"})
    #get only the time period interested
    bbp = bbp.iloc[bbp.index >=sd_str]
    #print("bbp:\n",bbp)
    
    #stochastic oscillator
    close = ind.get_close(symbols, dates)
    high = ind.get_high(symbols, dates)
    low = ind.get_low(symbols, dates)
    rolling_min, rolling_max, k,d = ind.so(prices, high, low, close, window, 3)
    d = d.rename(columns={"JPM":"%D"})
    #get only the time period interested
    d = d.iloc[d.index >=sd_str]
    #print("d:\n",d)
    
    #join 3 indicators
    ind3 = price_sma_df.join(bbp)
    ind3 = ind3.join(d)
    #print(ind3)
    
    #if indicators create buy signal, should hold +1000
    pos = 1000
    holdings = price_sma_df.copy()
    holdings[:]=0
    holdings = holdings.rename(columns={"Price/SMA":"JPM"})
    for i in range(0, holdings.shape[0]):
        if (ind3.iloc[i,0]<0.95 and ind3.iloc[i,1]<0 and ind3.iloc[i,2]<30):
            holdings.iloc[i,0] = pos
        if (ind3.iloc[i,0]>1.05 and ind3.iloc[i,1]>0.8 and ind3.iloc[i,2]>70):
            holdings.iloc[i,0] = -pos
    #print(holdings.groupby(['JPM']).size())
    
    #get orders from holdings
    orders = get_orders_from_holdings(holdings)
    #print("orders",orders)
    return orders

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
    becn_orders_os = test_os.copy()
    becn_orders_os.iloc[:,:] = 0
    becn_orders_os.iloc[0] = 1000
    bench_port_val_os = compute_portvals2(becn_orders_os,100000,9.95,0.005)

    # ManualStrategy
    ms_orders= testPolicy(symbol='JPM', sd=sd ,ed=ed, sv=100000)
    #print("ms_orders",ms_orders)
    ms_orders_df = ms_orders.copy()
    ms_port_val = compute_portvals2(ms_orders_df,100000,9.95,0.005)
    #out of sample
    ms_orders_os = testPolicy(symbol="JPM",sd=dt.datetime(2010,1,1),ed=dt.datetime(2011,12,31),sv=100000)
    ms_orders_os_df = ms_orders_os.copy()
    ms_port_val_os = compute_portvals2(ms_orders_os_df,100000,9.95,0.005)
    
    #normalize portfolio values
    bench_port_val = bench_port_val/bench_port_val.iloc[0]
    ms_port_val = ms_port_val/ms_port_val.iloc[0]
    st_port_val = st_port_val/st_port_val.iloc[0]
    #out of sample
    bench_port_val_os = bench_port_val_os/bench_port_val_os.iloc[0]
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
    plt.title("Fig 1: In-Sample Manual Strategy vs Bench",fontsize=25)
    plt.plot(bench_port_val, 'g', label ="Benchmark",linewidth=1.5)
    plt.plot(ms_port_val,'r', label = "Manual Strategy", linewidth=1.5)
    ms_holdings = ms_orders.cumsum(axis = 0)
    #print(ms_orders)
    #print("ms_holdings:",ms_holdings)
    for i, row in ms_holdings.iterrows():
        if row.values == 1000:
            plt.axvline(x = i,color='blue',linewidth=0.8)
        elif row.values == -1000:
            pass
            plt.axvline(x = i,color='black', linewidth=0.8)
    plt.legend(fontsize = 20)
    plt.savefig('fig1.png')
    
    #out of sample benchmark vs. manual strategy LONG/SHORT
    plt.figure(figsize=(14,8))
    plt.xlabel("Date",fontsize = 20)
    plt.ylabel("Normalized Port Value",fontsize = 20)
    start_date = dt.datetime(2010,1,1)
    end_date = dt.datetime(2011,12,31)
    plt.xlim(start_date,end_date)
    plt.title("Fig 2: Out-of-Sample Manual Strategy vs Bench",fontsize=25)
    plt.plot(bench_port_val_os, 'g', label ="Benchmark",linewidth=1.5)
    plt.plot(ms_port_val_os,'r', label = "Manual Strategy", linewidth=1.5)
    ms_holdings_os = ms_orders_os.cumsum(axis = 0)
    #print(ms_orders_os)
    #print("ms_holdings_os:",ms_holdings_os)
    for i, row in ms_holdings_os.iterrows():
        if row.values == 1000:
            plt.axvline(x = i,color='blue',linewidth=0.8)
        elif row.values == -1000:
            pass
            plt.axvline(x = i,color='black', linewidth=0.8)
    plt.legend(fontsize = 20)
    plt.savefig('fig2.png')
    
    #in-sample vs. out of sample manual strategy vs. benchmark
    fig, axs = plt.subplots(2,figsize=(14,14))
    fig.suptitle('Fig 3: In-Sample vs Out-of-Sample', fontsize=25)
    #axs[0].title("Figure 1: Price/{}-day Simple Moving Average".format(window),fontsize=25)
    axs[0].set_title("In-Sample Manual Strategy vs Bench", fontsize=22)
    axs[0].set_xlabel("Date", fontsize = 20)
    axs[0].set_ylabel("Normalized Portfolio Value", fontsize = 20)
    start_date = dt.datetime(2008,1,1)
    end_date = dt.datetime(2009,12,31)
    axs[0].set_xlim(start_date,end_date)
    axs[0].plot(bench_port_val, 'g', label ="Benchmark",linewidth=1.5)
    axs[0].plot(ms_port_val,'r', label = "Manual Strategy", linewidth=1.5)
    axs[0].legend(fontsize = 20)
    axs[1].set_title("Out-of-Sample Manual Strategy vs Bench", fontsize=22)
    axs[1].set_xlabel("Date", fontsize = 20)
    axs[1].set_ylabel("Normalized Portfolio Value", fontsize = 20)
    start_date = dt.datetime(2010,1,1)
    end_date = dt.datetime(2011,12,31)
    axs[1].set_xlim(start_date,end_date)
    axs[1].plot(bench_port_val_os, 'g', label ="Benchmark",linewidth=1.5)
    axs[1].plot(ms_port_val_os,'r', label = "Manual Strategy", linewidth=1.5)
    axs[1].legend(fontsize = 20)
    plt.savefig("fig3.png")
    
    '''
    #table
    port_cum_ret = ms_port_val.iloc[-1]/ms_port_val.iloc[0] - 1
    bench_cum_ret = bench_port_val.iloc[-1]/bench_port_val.iloc[0] - 1
    print("In-sample:")
    print("Portfolio Cumulative Return: ", port_cum_ret.item())
    print("Benchmark Cumulative Return: ", bench_cum_ret.item())
    #daily return std
    port_daily_ret = daily_return(ms_port_val)
    bench_daily_ret = daily_return(bench_port_val)
    print("Portfolio Daily Return Std:", port_daily_ret.std().item())
    print("Benchmark Daily Return Std:", bench_daily_ret.std().item())
    #daily return mean
    print("Portfolio Daily Return Mean:", port_daily_ret.mean().item())
    print("Benchmark Daily Return Mean:", bench_daily_ret.mean().item())
    print("Portfolio Sharpe Ratio:", (252**0.5)*port_daily_ret.mean().item()/port_daily_ret.std().item())
    print("Benchmark Sharpe Ratio:", (252**0.5)*bench_daily_ret.mean().item()/bench_daily_ret.std().item())
    
    print("out of sample:")
    port_cum_ret = ms_port_val_os.iloc[-1]/ms_port_val_os.iloc[0] - 1
    bench_cum_ret = bench_port_val_os.iloc[-1]/bench_port_val_os.iloc[0] - 1
    print("Portfolio Cumulative Return: ", port_cum_ret.item())
    print("Benchmark Cumulative Return: ", bench_cum_ret.item())
    #daily return std
    port_daily_ret = daily_return(ms_port_val_os)
    bench_daily_ret = daily_return(bench_port_val_os)
    print("Portfolio Daily Return Std:", port_daily_ret.std().item())
    print("Benchmark Daily Return Std:", bench_daily_ret.std().item())
    #daily return mean
    print("Portfolio Daily Return Mean:", port_daily_ret.mean().item())
    print("Benchmark Daily Return Mean:", bench_daily_ret.mean().item())
    print("Portfolio Sharpe Ratio:", (252**0.5)*port_daily_ret.mean().item()/port_daily_ret.std().item())
    print("Benchmark Sharpe Ratio:", (252**0.5)*bench_daily_ret.mean().item()/bench_daily_ret.std().item())
    '''
    
if __name__=="__main__":
    test_code()