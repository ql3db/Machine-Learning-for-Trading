from util import get_data, plot_data
import pandas as pd
import numpy as np
import datetime as dt #???
import os #???
import math #??
import matplotlib.pyplot as plt

def author():
    return 'qli385'

def get_prices(symbols, dates):
    prices = get_data(symbols, dates)
    prices = prices.drop(['SPY'], axis=1)
    return prices

def get_volume(symbols, dates):
    volume = get_data(symbols, dates, colname='Volume')
    volume = volume.drop(['SPY'], axis=1)
    return volume

def get_high(symbols, dates):
    high = get_data(symbols, dates, colname='High')
    high = high.drop(['SPY'], axis=1)
    return high

def get_low(symbols, dates):
    low = get_data(symbols, dates, colname='Low')
    low = low.drop(['SPY'], axis=1)
    return low

def get_close(symbols, dates):
    close = get_data(symbols, dates, colname='Close')
    close = close.drop(['SPY'], axis=1)
    return close

def adj_high_low(high_low, close, prices):
    adj = prices/close
    adj_high_low = adj * high_low
    #print("adj_high_low",adj_high_low)
    return adj_high_low

def sma(prices, window):
    #calculate cumsum
    #sma_cumsum = prices.cumsum(axis =0)
    #print(sma_cumsum)
    #0 - (window - 2) line: 0
    #sma = sma_cumsum.copy()
    #sma.iloc[0:window - 1,:] = 0
    #the (window - 1) line
    #sma.iloc[window - 1,:] = sma_cumsum.iloc[window - 1,:]/ window
    #window - end lines
    #sma.iloc[window:,:] = (sma_cumsum.iloc[window:,:] - sma_cumsum.iloc[0:-window,:].values) / window
    sma = prices.rolling(window = window, min_periods = window).mean()
    #print("sma():",sma)
    return sma

#1st indicator: price/sma
def price_sma(prices, sma, window):
    price_sma = sma.copy()
    price_sma.iloc[window - 1:,:] = prices.iloc[window - 1:, :] / sma.iloc[window - 1:, :].values
    #print("volume_sma",price_sma)
    return price_sma

#2nd indicator: bollinger band
def bbd_test(prices, sma, window):
    error = sma.copy()
    #0 - (window - 2) line: 0
    error.iloc[0:window - 1, :] = 0
    #calculate squared error
    error.iloc[window - 1:,:] = (prices.iloc[window - 1:,:] - sma.iloc[window - 1:,:])**2
    print("error:",error)
    #calculate cumsum of error
    error_cumsum = error.cumsum(axis = 0)
    print("error_cumsum", error_cumsum)
    #cumsum delta
    error_cumsum_delta = error_cumsum.copy()
    #error_cumsum_delta.iloc[window:,:] = error_cumsum.iloc[window:,:] - error_cumsum.iloc
    #rolling_std = prices.rolling(window = window, min_periods = window).std
    #print("rolling_std:",rolling_std)
    #top_band = sma + (2*rolling_std)
    #bottom_band = sma - (2*rolling_std)
    #bbp = (prices - bottom_band)/(top_band - bottom_band)
    return bbp

def bbd(prices, sma, window):
    rolling_std = prices.rolling(window = window, min_periods = window).std()
    #print("rolling_std:",rolling_std)
    top_band = sma + (2*rolling_std)
    bottom_band = sma - (2*rolling_std)
    bbp = (prices - bottom_band)/(top_band - bottom_band)
    return top_band, bottom_band,bbp

#no longer useful: 3rd indicator: On-balance Volumn 
#https://www.investopedia.com/terms/o/onbalancevolume.asp
def obv(prices, volume):
    #get the close price diff
    prices_diff = prices - prices.shift(1)
    #print("prices_diff", prices_diff)
    #get the delta of volumes: only when price_diff <0, volume is negative
    obv_delta = volume.copy()
    cond1 = prices_diff > 0
    obv_delta = obv_delta.where(cond1, -volume)
    cond2 = prices_diff != 0
    obv_delta = obv_delta.where(cond2, -volume)#????????????????location???/
    obv_delta.iloc[0,:] = 0
    #print("obv_delta",obv_delta)
    #get obv
    obv=obv_delta.cumsum(axis = 0)
    print("obv",obv)
    return obv

#no longer useful: 3rd indicator: momentum
def momentum(prices, window):
    m = prices / prices.shift(window) - 1
    return m

#3rd indicator: MACD
def macd(prices, window1 = 9, window2=12, window3=26):
    macd12 = ema(prices, window2)
    macd26 = ema(prices, window3)
    macd_delta = macd12 - macd26
    #print(macd_delta)
    macd9 = ema(macd_delta.iloc[window2+window3:,:], window=9)
    #print("macd9:", macd9)
    return macd9, macd12, macd26, macd_delta

#4th indicator: Force Index
#https://school.stockcharts.com/doku.php?id=technical_indicators:force_index
def fi(prices, volume, window):
    prices_diff = prices - prices.shift(1)
    #print(prices_diff)
    fi = prices_diff * volume
    #print("fi:",fi)
    fi_ema = ema(fi, window)
    #print(fi_ema)
    return fi, fi_ema

#exponential moving average
##https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/slow-stochastic#:~:text=A%20sell%20signal%20occurs%20when,confirmed%20by%20the%20Stochastic%20Oscillator.
def ema(df, window):
    multiplier = (2/(window + 1))
    sma_df = sma(df, window)
    #print("sma to ema:", sma_df)
    ema = sma_df.copy()
    #print("ema=sma:", ema)
    cur = window + 1
    #print("df",df)
    while cur < ema.shape[0]:
        ema.iloc[cur,:] = multiplier *df.iloc[cur,:] + (1 - multiplier) * ema.iloc[cur - 1, :]
        #print(cur)
        cur = cur + 1
    #print("ema:", ema)
    return ema

#5th indicator: stochastic oscillator
#https://school.stockcharts.com/doku.php?id=technical_indicators:stochastic_oscillator_fast_slow_and_full
def so(prices, high, low, close, window, window2):
    #%K = (Current Close - Lowest Low)/(Highest High - Lowest Low) * 100
    #D = 3-day SMA of %K
    adj_high = adj_high_low(high, close, prices)
    adj_low = adj_high_low(low, close, prices)
    rolling_min = adj_low.rolling(window = window, min_periods = window).min()
    #print("rolling_min",rolling_min)
    rolling_max = adj_high.rolling(window = window, min_periods = window).max()
    #print("rolling_max",rolling_max)
    deno = rolling_max - rolling_min
    #print("deno:", deno)
    k = (prices - rolling_min)/deno * 100
    #print("k", k)
    d = sma(k,window2)
    #print("d", d)
    return rolling_min, rolling_max, k, d 
    


def test_code():
    #test get_prices()
    #dates = pd.date_range("2008-01-01","2009-12-31")
    #symbols=['JPM']
    #prices = get_prices(symbols,dates)
    #print(prices)
    
    #test sma()
    #window = 2
    #sma_df = sma(prices, window)
    
    #test price_sma()
    #price_sma_df = price_sma(prices, sma_df, window)
    
    #print(get_rolling_std(prices))
    #test bbd()
    #print('bbd')
    #bbd(prices,sma_df,window)
    
    #volume and obv
    #volume = get_volume(symbols,dates)
    #print("volume:" , volume)
    #cond = prices>200
    #print("test",volume.where(cond,-volume))
    #obv_df = obv(prices, volume)
    
    #test force index
    #fi_df = fi(prices, volume, window)
    
    #test ema
    #ema_df = ema(prices, window)
    
    #test so
    #so_df = so(prices,window)
    
    #plots
    #1 price/SMA
    symbols=['JPM']
    window = 14
    dates = pd.date_range("2008-01-01","2009-12-31")
    
    prices = get_prices(symbols,dates)
    #print("prices",prices)
    sma_df = sma(prices, window)
    
    price_sma_df = price_sma(prices, sma_df, window)
    #normalize
    prices_norm = prices / prices.iloc[0]
    sma_df_norm = sma_df/sma_df.iloc[window - 1]
    #print("sma_df:", sma_df)
    #print(price_sma_df)
    
    plt.figure(figsize=(14,8))
    plt.xlabel("Date",fontsize = 20)
    plt.ylabel("Normalized Values",fontsize = 20)
    start_date = dt.datetime(2008,1,1)
    end_date = dt.datetime(2009,12,31)
    plt.xlim(start_date,end_date)
    plt.title("Fig 1: JPM Price/{}-day Simple Moving Average".format(window),fontsize=25)
    plt.plot(prices_norm, label ="Price")
    plt.plot(sma_df_norm, label = "{}-day SMA".format(window))
    plt.plot(price_sma_df, label = "Prices/{}day SMA".format(window))
    plt.axhline(y=0.95, color='r', linestyle='--')
    plt.axhline(y=1.05, color='r', linestyle='--', label = "1.05 and 0.95 line")
    plt.legend(fontsize = 20)
    plt.savefig('fig1.png')
    
    
    #2 bollinger bands
    top_band, bottom_band,bbp = bbd(prices,sma_df,window)
    
    fig, axs = plt.subplots(2,figsize=(14,14))
    fig.suptitle('Fig 2: JPM Bollinger Band %', fontsize=25)
    #axs[0].title("Figure 1: Price/{}-day Simple Moving Average".format(window),fontsize=25)
    axs[0].set_title("JPM Price, {}-day SMA and Bollinger Bands".format(window), fontsize=22)
    axs[0].set_xlabel("Date", fontsize = 20)
    axs[0].set_ylabel("Price", fontsize = 20)
    axs[0].plot(top_band, label = "Upper Band")
    axs[0].plot(bottom_band, label = "Lower Band")
    axs[0].plot(sma_df, label = "{}-day SMA".format(window))
    axs[0].plot(prices, label = "Price", color ='grey')
    axs[0].legend(fontsize = 20)
    axs[1].set_title("JPM Bollinger Band %".format(window), fontsize=22)
    axs[1].set_xlabel("Date", fontsize = 20)
    axs[1].set_ylabel("Bollinger Band %", fontsize = 20)
    axs[1].plot(bbp, label = "Bollinger Band %")
    axs[1].axhline(y=0, color='r', linestyle='--')
    axs[1].axhline(y=1, color='r', linestyle='--', label = "1 and 0 line")
    axs[1].legend(fontsize = 20, loc="upper right")
    plt.savefig("fig2.png")
    
    
    #3 MACD
    #https://commodity.com/technical-analysis/momentum/
    macd9, macd12, macd26, macd_delta = macd(prices)
    
    
    fig, axs = plt.subplots(2,figsize=(14,14))
    fig.suptitle('Fig 3: JPM MACD', fontsize=25)
    axs[0].set_title("JPM Price and MACD", fontsize=22)
    axs[0].set_xlabel("Date", fontsize = 20)
    axs[0].set_ylabel("Price, MACD", fontsize = 20)
    axs[0].plot(prices, label = "Price")
    axs[0].plot(macd12, label = "12-day EMA")
    axs[0].plot(macd26, label = "26-day EMA")
    axs[0].legend(fontsize = 20)
    axs[1].set_title("MACD", fontsize=22)
    axs[1].set_xlabel("Date", fontsize = 20)
    axs[1].set_ylabel("MACD", fontsize = 20)
    axs[1].plot(macd_delta, label = "MACD")
    axs[1].plot(macd9, label = "Signal")
    axs[1].plot(macd_delta-macd9, label = "MACD - Signal")
    #axs[1].hist(macd_delta-macd9, label = "?")
    #axs[1].axhline(y=80, color='r', linestyle='--')
    axs[1].axhline(y=0, color='r', linestyle='--', label = "0 line")
    axs[1].legend(fontsize = 20)
    plt.savefig("fig3.png")
        
    
    #4 Force Index
    volume = get_volume(symbols,dates)
    fi_df, fi_ema = fi(prices, volume, window)
    
    fi_df_norm =fi_df/(10**8)
    fi_ema_norm = fi_ema/((10**7)*2)
    #fi_ema_norm = fi_df/fi_df.iloc[window]
    #print(fi_df)
    #print(fi_ema)
    
    plt.figure(figsize=(14,8))
    plt.xlabel("Date",fontsize = 20)
    plt.ylabel("Force Index",fontsize = 20)
    start_date = dt.datetime(2008,1,1)
    end_date = dt.datetime(2009,12,31)
    plt.xlim(start_date,end_date)
    plt.title("Fig 4: JPM Force Index".format(window),fontsize=25)
    plt.plot(fi_df_norm, color='orange',label ="Raw Force Index")
    plt.plot(fi_ema_norm, 'b',label = "{}-day Force Index EMA".format(window))
    plt.axhline(y=0, color='r', linestyle='--', label = "0 line")
    plt.legend(fontsize = 20)
    plt.savefig('fig4.png')
    
    #plt.figure(figsize=(14,8))
    #plt.xlabel("Date",fontsize = 20)
    #plt.ylabel("Force Index EMA",fontsize = 20)
    #start_date = dt.datetime(2008,1,1)
    #end_date = dt.datetime(2009,12,31)
  #  plt.xlim(start_date,end_date)
   # plt.title("Fig 4.2: JPM Force Index EMA".format(window),fontsize=25)
  ##  plt.plot(fi_ema, 'b',label = "{}-day Force Index EMA".format(window))
   # plt.axhline(y=0, color='r', linestyle='--', label = "0, 0.5, -0.5 line")
   # plt.axhline(y=-0.5*100000000, color='r', linestyle='--')
  #  plt.axhline(y=0.5*100000000, color='r', linestyle='--')
  #  plt.legend(fontsize = 20)
  #  plt.savefig('fig42.png')
    
    #5 Stochastic Oscillator
    close = get_close(symbols, dates)
    high = get_high(symbols, dates)
    low = get_low(symbols, dates)
    #prices, high, low, close, window, window2
    rolling_min, rolling_max, k,d = so(prices, high, low, close, window, 3)
    
    fig, axs = plt.subplots(2,figsize=(14,14))
    fig.suptitle('Fig 5: JPM Stochastic Oscillator', fontsize=25)
    axs[0].set_title("JPM Price, Lowest Low and Highest High", fontsize=22)
    axs[0].set_xlabel("Date", fontsize = 20)
    axs[0].set_ylabel("Price", fontsize = 20)
    axs[0].plot(rolling_max, label = "Highest High")
    axs[0].plot(rolling_min, label = "Lowest Low")
    #axs[0].plot(sma_df, label = "{}-day SMA".format(window))
    axs[0].plot(prices, label = "Price")
    axs[0].legend(fontsize = 20)
    axs[1].set_title("JPM Stochastic Oscillator", fontsize=22)
    axs[1].set_xlabel("Date", fontsize = 20)
    axs[1].set_ylabel("%K, %D", fontsize = 20)
    axs[1].plot(k, label = "%K")
    axs[1].plot(d, label = "%D")
    axs[1].axhline(y=80, color='r', linestyle='--')
    axs[1].axhline(y=20, color='r', linestyle='--', label = "80 and 20 line")
    axs[1].legend(fontsize = 20, loc="upper right")
    plt.savefig("fig5.png")
    
    #plt.figure(figsize=(14,8))
    #plt.xlabel("Date",fontsize = 20)
   # plt.ylabel("Stochastic Oscillator",fontsize = 20)
   # start_date = dt.datetime(2008,1,1)
  #  end_date = dt.datetime(2009,12,31)
   # plt.xlim(start_date,end_date)
   # plt.title("Fig 5: JPM Stochastic Oscillator".format(window),fontsize=25)
   # plt.plot(k,'g',label = "Stochastic Oscillator".format(window))
   # plt.plot(d,'b',label = "{}-day Stochastic Oscillator SMA".format(window))
   # plt.axhline(y=80, color='r', linestyle='--', label = "20, 80 line")
   # plt.axhline(y=20, color='r', linestyle='--')
   # plt.legend(fontsize = 20)
   # plt.savefig('fig5.png')

if __name__ == "__main__":
    test_code()