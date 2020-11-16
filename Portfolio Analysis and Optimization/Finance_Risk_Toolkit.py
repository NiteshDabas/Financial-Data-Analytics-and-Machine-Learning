#!/usr/bin/env python
# coding: utf-8

# # Finance Risk Toolkit

# In this file I am going to write finance domain specific functions(financial tools) which performs specific risk evaluation tasks and we are going to import this file as a module named Finance_Risk_Toolkit in another file where I will use these tools.

# In[1]:


#import the libraries
import pandas as pd
import numpy as np
import scipy.stats
from scipy.stats import norm
from scipy.stats import skew
from scipy.stats import kurtosis


# In[2]:


#function to compute Wealth Index, Previous Peaks and Drawdowns from asset returns
def drawdown(return_series: pd.Series):
    wealth_index = 1000*(1+return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks)/previous_peaks
    return pd.DataFrame({"Wealth": wealth_index, 
                         "Previous Peak": previous_peaks, 
                         "Drawdown": drawdowns})


# In[3]:


#function to load the returns on top and bottom 10% stocks from our dataset
def get_stocks_returns():
    stocks_data = pd.read_csv("../Data/Portfolios_monthly_data.csv", header=0, index_col=0, na_values=-99.99)
    returns_topbottomdeciles = stocks_data[['Lo 10', 'Hi 10']]
    returns_topbottomdeciles.columns = ['SmallCap', 'LargeCap']
    returns_topbottomdeciles = returns_topbottomdeciles/100
    returns_topbottomdeciles.index = pd.to_datetime(returns_topbottomdeciles.index, format="%Y%m").to_period('M')
    return returns_topbottomdeciles


# In[4]:


#function to load and format the hedge fund's returns
def get_hedgefunds_returns():
    hedgefunds_data = pd.read_csv("../Data/HedgeFundsData.csv", header=0, index_col=0, parse_dates=True)
    hedgefunds_data = hedgefunds_data/100
    hedgefunds_data.index = hedgefunds_data.index.to_period('M')
    return hedgefunds_data


# In[5]:


#function to perform Jarque-Bera test of normailty
def is_normal(s, alpha=0.01):
    if isinstance(s, pd.DataFrame):
        return s.aggregate(is_normal)
    else:
        statistic, p_value = scipy.stats.jarque_bera(s)
        return p_value > alpha


# In[6]:


#function to return semideviation(negative)
def semideviation(s):
    is_negative = s < 0
    return s[is_negative].std(ddof=0)


# In[7]:


#function to return the historic VaR at a specified alpha
def VaR_historic(s, alpha=5):
    if isinstance(s, pd.DataFrame):
        return s.aggregate(VaR_historic, alpha=alpha)
    elif isinstance(s, pd.Series):
        return -np.percentile(s, alpha)
    else:
        raise TypeError("Incorrect data format")


# In[8]:


#function to return the conditional VaR at a specified alpha
def cVaR_historic(s, alpha=5):
    if isinstance(s, pd.Series):
        is_beyond = s <= -VaR_historic(s, alpha=alpha)
        return -s[is_beyond].mean()
    elif isinstance(s, pd.DataFrame):
        return s.aggregate(cVaR_historic, alpha=alpha)
    else:
        raise TypeError("Incorrect data format")
        

# In[9]:


#function to return the Parametric/Semi-Parametric Gauusian VaR at a specified alpha
def VaR_gaussian(s, alpha=5, modified=False):
    z = norm.ppf(alpha/100) #z-score
    if modified: #i.e. Cornish Fisher VaR
        s = skew(s)
        k = kurtosis(s)
        z = (z +
                (z**2 - 1)*s/6 +
                (z**3 -3*z)*(k-3)/24 -
                (2*z**3 - 5*z)*(s**2)/36
            )
    return -(s.mean() + z*s.std(ddof=0))

