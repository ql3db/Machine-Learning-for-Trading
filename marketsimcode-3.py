import pandas as pd  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
import numpy as np  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
import datetime as dt  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
import os  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
from util import get_data, plot_data

def author():
    return 'qli385'

def compute_portvals2(orders_df, start_val = 100000, commission=0, impact=0):  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    # this is the function the autograder will call to test your code  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    # NOTE: orders_file may be a string, or it may be a file object. Your  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    # code should work correctly with either input  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    # TODO: Your code here

    
    #get the first date and last date of orders
    start_date = orders_df.index[0]
    end_date=orders_df.index[-1]
    #print(start_date, end_date)
    
    #get prices of all tickers inside orders within the data range
    dates = pd.date_range(start_date, end_date)
    #symbols = np.array(orders_df.columns[0])
    symbols=[orders_df.columns[0]]
    #print("symbols", type(symbols))
    prices_df = get_data(symbols, dates)
    #fill na
    prices_df.fillna(method='ffill',inplace=True)
    prices_df.fillna(method='bfill',inplace=True)
    #add a cash column
    prices_df = prices_df.drop(['SPY'], axis=1)
    #print("prices_df:", prices_df)
    
    stock = orders_df.columns[0]
    impact_fee = prices_df*orders_df*impact
    impact_fee = impact_fee.iloc[:,:].abs()
    impact_fee.loc[impact_fee.iloc[:,0] >0, symbols] = commission + impact_fee.iloc[:,0]
    #impact_fee['commission'] = 0
    #impact_fee.loc[impact_fee.iloc[:,0] >0, 'commission'] = commission
    #impact_fee_df = impact_fee[symbols]
    #comm_df = impact_fee.loc['commision'] 
    #print("if",impact_fee)
    orders_df['CASH'] = -prices_df * orders_df - impact_fee
    #impact_fee.loc[impact_fee.iloc[:,0] >0, symbols] = commission
    #orders_df['CASH'] -= impact_fee
    #print("orders_df",orders_df)
    #print("test",impact_fee.iloc[:,0]) 
    #- impact_fee.iloc[:,1] 
    
    #print("shares_cost",shares_cost)
    #print("orders_df", orders_df)
    #print("trades_df", trades_df)
    #print("test trades:\n",trades_df.loc['2009-08-01':'2009-08-05',:])
    
    #get holdings (cumulative)
    holdings_df = orders_df.cumsum(axis=0)
    holdings_df.loc[:,'CASH'] += start_val
    #print("holdings_df:\n", holdings_df)
    
    #get portfolios value
    #values of each asset
    prices_df['CASH'] = 1.0
    hold_val_df = holdings_df *prices_df
    #print("hold_val_df",type(hold_val_df))
    #print("holdings*prices:\n", hold_val_df)
    #add all asset values
    port_val_df = pd.DataFrame(hold_val_df.sum(axis=1),columns=['sum'])
    #print("port_val_df",type(port_val_df))
    #print("port_val:\n", port_val_df)
    
    
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    # In the template, instead of computing the value of the portfolio, we just  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    # read in the value of IBM over 6 months  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    #start_date = dt.datetime(2008,1,1)  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    #end_date = dt.datetime(2008,6,1)  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    #portvals = get_data(['IBM'], pd.date_range(start_date, end_date))  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    #portvals = portvals[['IBM']]  # remove SPY  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    #rv = pd.DataFrame(index=portvals.index, data=portvals.values)  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    #return rv  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    return port_val_df 