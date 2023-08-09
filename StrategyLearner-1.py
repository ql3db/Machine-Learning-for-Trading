"""  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
Atlanta, Georgia 30332  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
All Rights Reserved  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
Template code for CS 4646/7646  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
works, including solutions to the projects assigned in this course. Students  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
and other users of this template code are advised not to share it with others  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
or to make it available on publicly viewable websites including repositories  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
such as github and gitlab.  This copyright statement should not be removed  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
or edited.  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
We do grant permission to share solutions privately with non-students such  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
as potential employers. However, sharing with other current or future  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
GT honor code violation.  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
-----do not edit anything above this line---  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
Student Name: Tucker Balch (replace with your name)  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
GT User ID: qli385 (replace with your User ID)  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
GT ID: 903472876 (replace with your GT ID)  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
""" 

import datetime as dt
import pandas as pd
import util as ut
import random
import RTLearner as rt
import BagLearner as bl
import indicators as ind
import numpy as np

class StrategyLearner(object):

    def author(self):
        return 'qli385'

    # constructor
    def __init__(self, verbose = False, impact=0.0):
        self.verbose = verbose
        self.impact = impact
        self.learner = bl.BagLearner(learner = rt.RTLearner, kwargs = {"leaf_size":5}, bags = 20, boost = False, verbose = False)
        self.impact = impact

    def get_ind_df(self,symbol='JPM', sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31)):
        #use indicators: price/SMA, BB%, stochastic oscillator
        window = 14
        sd_start = sd - dt.timedelta(days=window*2) #so real start dates have data
        dates = pd.date_range(sd_start, ed)
        symbols = [symbol]
        if (isinstance(sd, dt.datetime)):
            sd_str = sd.strftime("%Y-%m-%d")

        #price/SMA
        prices = ind.get_prices(symbols,dates)
        sma_df = ind.sma(prices, window)
        price_sma_df = ind.price_sma(prices, sma_df, window)
        price_sma_df = price_sma_df.rename(columns={symbol:"SMA"})
        final_dates = pd.date_range(sd, ed)
        #get only the time period interested
        price_sma_df = price_sma_df.iloc[price_sma_df.index >=sd_str]
        #print("price/SMA:\n",price_sma_df)

        #BB%
        top_band, bottom_band,bbp = ind.bbd(prices,sma_df,window)
        bbp = bbp.rename(columns={symbol:"BBP"})
        #get only the time period interested
        bbp = bbp.iloc[bbp.index >=sd_str]
        #print("bbp:\n",bbp)

        #stochastic oscillator
        close = ind.get_close(symbols, dates)
        high = ind.get_high(symbols, dates)
        low = ind.get_low(symbols, dates)
        rolling_min, rolling_max, k,d = ind.so(prices, high, low, close, window, 3)
        d = d.rename(columns={symbol:"VOL"})
        #get only the time period interested
        d = d.iloc[d.index >=sd_str]
        #print("d:\n",d)

        #join 3 indicators
        #ind3 = price_sma_df.join(bbp)
        #ind3 = ind3.join(d)
        #print(ind3)

        return price_sma_df, bbp, d, symbol
    
    # this method should create a QLearner, and train it for trading
    def addEvidence(self, symbol = "IBM", \
        sd=dt.datetime(2008,1,1), \
        ed=dt.datetime(2009,1,1), \
        sv = 10000):

        # example usage of the old backward compatible util function
        #syms=[symbol]
        #dates = pd.date_range(sd, ed)
        #prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        #prices = prices_all[syms]  # only portfolio symbols
        # prices_SPY = prices_all['SPY']  # only SPY, for comparison later
        sma, bbp, d, symbol = self.get_ind_df(symbol=symbol)
        dates = pd.date_range(sd, ed)
        symbols = [symbol]
        prices = ind.get_prices(symbols,dates)
        
        # Getting the technical indicators
        #lookback = 2
        #Indicator no. 1 : SMA/price
        #sma = getSMA(prices,lookback,syms)
        #copysma = sma.copy()
        #Indicator no. 2 : Bollinger bands
        #bba = getBollinger(prices,syms,lookback,copysma)
        #Indicator no. 3 : Volatility
        #volatility = getVolatility(prices,lookback,syms)

        # Constructing trainX
        #df1=sma.rename(columns={symbol:'SMA'})
        #df2=bba.rename(columns={symbol:'BBP'})
        #df3=volatility.rename(columns={symbol:'VOL'})
        df1 = sma.copy()
        df2 = bbp.copy()
        df3=d.copy()

        indicators = pd.concat((df1,df2,df3),axis=1)
        indicators.fillna(0,inplace=True)
        indicators=indicators[:-5]
        trainX = indicators.values

        # Constructing trainY
        trainY=[]
        for i in range(prices.shape[0]-5):
            ratio = (prices.iloc[i+5,0]-prices.iloc[i,0])/prices.iloc[i,0]
            if ratio > (0.02 + self.impact):
                trainY.append(1)
            elif ratio < (-0.02 - self.impact):
                trainY.append(-1)
            else:
                trainY.append(0)
        trainY=np.array(trainY)

        # Training
        self.learner.addEvidence(trainX,trainY)


    # this method should use the existing policy and test it against new data
    def testPolicy(self, symbol = "IBM", \
        sd=dt.datetime(2009,1,1), \
        ed=dt.datetime(2010,1,1), \
        sv = 10000):

        #syms=[symbol]
        #dates = pd.date_range(sd, ed)
        #prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        #prices = prices_all[syms]  # only portfolio symbols
        # prices_SPY = prices_all['SPY']  # only SPY, for comparison later
        sma, bbp, d, symbol = self.get_ind_df(symbol=symbol)
        dates = pd.date_range(sd, ed)
        symbols = [symbol]
        prices = ind.get_prices(symbols,dates)


        # Getting the technical indicators
        #lookback = 2
        #Indicator no. 1 : SMA
        #sma = getSMA(prices,lookback,syms)
        #copysma = sma.copy()
        #Indicator no. 2 : Bollinger bands
        #bba = getBollinger(prices,syms,lookback,copysma)
        #Indicator no. 3 : Volatility
        #volatility = getVolatility(prices,lookback,syms)


        # Constructing testX
        #df1=sma.rename(columns={symbol:'SMA'})
        #df2=bba.rename(columns={symbol:'BBA'})
        #df3=volatility.rename(columns={symbol:'VOL'})
        df1 = sma.copy()
        df2 = bbp.copy()
        df3=d.copy()

        indicators = pd.concat((df1,df2,df3),axis=1)
        indicators.fillna(0,inplace=True)
        testX = indicators.values

        # Querying the learner for testY
        testY=self.learner.query(testX)

        # Constructing trades DataFrame
        trades = prices[symbols].copy()
        trades.loc[:]=0
        flag=0
        for i in range(0,prices.shape[0]-1):
            if flag==0:
                if testY[i]>0:
                    trades.values[i,:] = 1000
                    flag = 1
                elif testY[i]<0:
                    trades.values[i,:] = -1000
                    flag = -1

            elif flag==1:
                if testY[i]<0:
                    trades.values[i,:]=-2000
                    flag=-1
                elif testY[i]==0:
                    trades.values[i,:]=-1000
                    flag = 0

            else:
                if testY[i]>0:
                    trades.values[i,:]=2000
                    flag=1
                elif testY[i]==0:
                    trades.values[i,:]=1000
                    flag=0

        if flag==-1:
            trades.values[prices.shape[0]-1,:]=1000
        elif flag==1:
            trades.values[prices.shape[0]-1,:]=-1000

        return trades

if __name__=="__main__":
    print("One does not simply think up a strategy")
    st = StrategyLearner()
    