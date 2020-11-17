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
from scipy.optimize import minimize


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
        

# In[10]:


#function to load and format the 30 Industry Portfolios Value Weighted Monthly Returns
def get_ind_returns():
    industy_returns_data = pd.read_csv("../Data/dataset_ind30_vw_returns.csv", header=0, index_col=0)/100
    industy_returns_data.index = pd.to_datetime(industy_returns_data.index, format="%Y%m").to_period('M')
    industy_returns_data.columns = industy_returns_data.columns.str.strip()
    return industy_returns_data


# In[11]:


#function to return annualized returns of his/her investment
def annualize_rets(s, periods_per_year):
    compounded_growth = (1+s).prod()
    n_periods = s.shape[0]
    return compounded_growth**(periods_per_year/n_periods)-1


# In[12]:


#function to return annualized volatility of his/her investment
def annualize_vol(s, periods_per_year):
    return s.std()*(periods_per_year**0.5)


# In[13]:


#function to calculate annualized sharpe ratio of a set of returns
def sharpe_ratio(s, riskfree_rate, periods_per_year):
    # convert the annual riskfree rate to per period
    rf_per_period = (1+riskfree_rate)**(1/periods_per_year)-1
    excess_ret = s - rf_per_period
    ann_ex_ret = annualize_rets(excess_ret, periods_per_year)
    ann_vol = annualize_vol(s, periods_per_year)
    return ann_ex_ret/ann_vol


# In[14]:


#function to compute the returns of a portfolio, given a set of weights, returns, and a covariance matrix.
def portfolio_return(weights, returns):
    return weights.T @ returns


# In[15]:


#function to compute the volatility of a portfolio, given a set of weights, returns, and a covariance matrix.
def portfolio_vol(weights, covmat):
    return (weights.T @ covmat @ weights)**0.5


# In[16]:


#function to plot 2-assets efficient frontier
def plot_ef2(n_points, er, cov):
    if er.shape[0] != 2 or er.shape[0] != 2:
        raise ValueError("plot_ef2 can only plot 2-asset frontiers")
    weights = [np.array([w, 1-w]) for w in np.linspace(0, 1, n_points)]
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({
        "Returns": rets, 
        "Volatility": vols
    })
    return ef.plot.line(x="Volatility", y="Returns", style=".-")


# In[17]:


#function to minimize the volatility for a given level of returns
def minimize_vol(target_return, er, cov):
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n # an N-tuple of 2-tuples!
    # construct the constraints
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
    }
    return_is_target = {'type': 'eq',
                        'args': (er,),
                        'fun': lambda weights, er: target_return - portfolio_return(weights,er)
    }
    weights = minimize(portfolio_vol, init_guess,
                       args=(cov,), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,return_is_target),
                       bounds=bounds)
    return weights.x


# In[18]:


# Optimizes the weights given a particular gridspace
def optimal_weights(n_points, er, cov):
    target_rs = np.linspace(er.min(), er.max(), n_points)
    weights = [minimize_vol(target_return, er, cov) for target_return in target_rs]
    return weights


# In[19]:


#function to plot multi-asset efficient frontier
def plot_ef(n_points, er, cov):
    weights = optimal_weights(n_points, er, cov)
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({
        "Returns": rets, 
        "Volatility": vols
    })
    return ef.plot.line(x="Volatility", y="Returns", style='.-', legend=False)