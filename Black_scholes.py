# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 19:33:05 2023

@author: amish
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math as m

#Extracting required data from NSE 
df1=pd.read_csv('Nifty5031Aug.csv')
df1call=df1[['IV','LTP','STRIKE','LTP_put']]

#Codinig Black-Scholes Model
from scipy.stats import norm
def black_scholes(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price


#Applying the model on a specific Dataset
df1call=df1call.iloc[47:59]
df1call.reset_index(inplace=True)
df1call.drop('index',axis=1,inplace=True)
l=[]
for i in range(0,12):
    a=black_scholes(19400,float((df1call.iloc[i,2]).replace(',','')),7/366,0.0675,float(df1call.iloc[i,0])*0.01)
    l.append(a)
df1call['pred_LTP']=''    
df1call['pred_LTP']=l

#Calculating errors
df1call['error']=''
for i in range(0,12):
    df1call['error'][i]=df1call['pred_LTP'][i]-float(df1call['LTP'][i])
    
    
#Plotting the curves
fige,ax=plt.subplots(nrows=1,ncols=2)

ax[0].plot(df1call['error'][0:6],'o--',color='blue')
ax[0].plot(df1call['error'][6:12],'o--',color='red')

#Backtesting on 2nd Expiry Date
df2=pd.read_csv('Nifty507SEPT.csv')
df2call=df2[['IV','LTP','STRIKE','LTP_put']]
df2call=df2call.iloc[27:39]
df2call.reset_index(inplace=True)
df2call.drop("index",axis=1,inplace=True)
l2=[]
for i in range(0,12):
    a=black_scholes(19400,float((df2call.iloc[i,2]).replace(',','')),7/366,0.0675,float(df2call.iloc[i,0])*0.01)
    l2.append(a)
df2call['pred_LTP']=''
df2call['pred_LTP']=l2
df2call['error']=''
for i in range(0,12):
    df2call['error'][i]=df2call['pred_LTP'][i]-float(df2call['LTP'][i])
ax[0].plot(df2call['error'][0:6],'o--',color='blue')
ax[0].plot(df2call['error'][6:12],'o--',color='red')
plt.title('Error plot')
ax[0].set_title("Call-Option error")
ax[0].legend(['OTM(31Aug)','ITM(31Aug)','OTM(7Sep)','ITM(7Sep)'])

#Using Put Call Parity for calculating Price of Put Options

df1call['put_p']=''

arr=[]
for i in range(0,12):
    a=float((df1call['STRIKE'][i]).replace(',',''))
    arr.append(a)
for i in range(0,12):
    df1call['put_p'][i]=float(df1call['LTP'][i])+arr[i]*(pow(m.e,-0.0675*7/366))-19368.
    

df1call['error_put']=df1call['put_p']-(df1call['LTP_put']).astype(float)
print(df1call)


df2call['put_p']=''

arr2=[]
for i in range(0,12):
    a=float((df2call['STRIKE'][i]).replace(',',''))
    arr2.append(a)
for i in range(0,12):
    df2call['put_p'][i]=float(df2call['LTP'][i])+arr2[i]*(pow(m.e,-0.0675*7/366))-19368.
    

df2call['error_put']=df2call['put_p']-(df2call['LTP_put']).astype(float)
print(df2call)


#Plotting errors for put options 
ax[1].plot(df1call['error_put'][0:6],'b--')
ax[1].plot(df1call['error_put'][6:12],'r--')
ax[1].plot(df2call['error_put'][0:6],'b--')
ax[1].plot(df2call['error_put'][6:12],'r--')
ax[1].legend(['OTM(31Aug)','ITM(31Aug)','OTM(7Sep)','ITM(7Sep)'])
ax[1].set_title("Put-Option error")

plt.show()
fige.suptitle("Deviation from actual premiums against Various Strikes")


#Plotting predicted LTP's and actual LTP's
fig,axes = plt.subplots(nrows=1,ncols=2)

axes[0].plot(df1call['LTP'].astype(float))
axes[0].plot(df1call['pred_LTP'],'o--')
axes[0].set_title('31Aug')

axes[0].plot(df1call['LTP_put'].astype(float))
axes[0].plot(df1call['put_p'],'o--')
axes[0].legend(['LTP','Pred_LTP','LTP put','Pred_LTP put'])
axes[0].set_title('31Aug')

axes[1].plot(df2call['LTP'].astype(float))
axes[1].plot(df2call['pred_LTP'],'o--')
axes[1].set_title('7Sept')

axes[1].plot(df2call['LTP_put'].astype(float))
axes[1].plot(df2call['put_p'],'o--')
axes[1].legend(['LTP','Pred_LTP','LTP put','Pred_LTP put'])
axes[1].set_title('7Sep')

fig.suptitle("Premium Prices against Various Strikes")

# Calculating RMSE Errors for the results and creating an RMSE dataframe
from sklearn.metrics import mean_squared_error
R=m.sqrt(mean_squared_error(df1call['pred_LTP'],df1call['LTP'].astype(float)))
b=m.sqrt(mean_squared_error(df2call['pred_LTP'],df2call['LTP'].astype(float)))
c=m.sqrt(mean_squared_error(df1call['put_p'],df1call['LTP_put'].astype(float)))
d=m.sqrt(mean_squared_error(df2call['put_p'],df2call['LTP_put'].astype(float)))

data={'31Aug':[R,c],'7Sep':[b,d]}
RMSE=pd.DataFrame(data,index=['call','put'])

print(RMSE)


## Monte Carlo Comparison

