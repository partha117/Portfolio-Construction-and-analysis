import pandas as pd
import scipy
import numpy as np
from scipy.stats import norm

def drawdown(return_series:pd.Series):
    wealth_index = 1000* (1+ return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdown = (wealth_index - previous_peaks) / previous_peaks
    return pd.DataFrame({
        "Wealth": wealth_index,
        "Peaks": previous_peaks,
        "Drawdown": drawdown
    })


def get_ffme_return():
    me_m = pd.read_csv("data/Portfolios_Formed_on_ME_monthly_EW.csv",header=0,index_col=0,parse_dates=True,na_values=-99.99)
    columns = ['Lo 10','Hi 10']
    rets = me_m[columns]
    rets = rets / 100
    rets.columns = ['SmallCap','LargeCap']
    rets.index = pd.to_datetime(rets.index,format= "%Y%m")
    rets.index = rets.index.to_period('M')
    return rets

def get_hfi_return():
    hfi = pd.read_csv("data/edhec-hedgefundindices.csv",header=0,index_col=0,parse_dates=True,na_values=-99.99)
    hfi = hfi / 100
    hfi.index = hfi.index.to_period('M')
    return hfi

def get_ind_return():
    ind = pd.read_csv("data/ind30_m_vw_rets.csv",header=0,index_col=0,parse_dates=True)/100
    ind.index = pd.to_datetime(ind.index,format= "%Y%m").to_period('M')
    ind.columns = ind.columns.str.strip()
    return ind


def annualize_return(r, periods_per_year):
    compound_growth = (1+r).prod()
    n_periods = r.shape[0]
    return compound_growth**(periods_per_year/n_periods) -1



def annualize_volatility(r,periods_per_year):
    return r.std() * (periods_per_year**0.5)
 
    
def sharpe_ratio(r,risk_free_rate,periods_per_year):
    rf_per_period = ( 1+risk_free_rate)**(1/periods_per_year) - 1
    excess_rate = r - rf_per_period
    ann_ex_rate = annualize_return(excess_rate, periods_per_year)
    ann_vol = annualize_volatility(r,periods_per_year)
    return ann_ex_rate/ann_vol



def semideviation(r):
    is_negative = r < 0
    return r[is_negative].std(ddof=0)
    
    
def skewness(r):
    demeaned_r = r - r.mean()
    sigma_r = r.std(ddof=0) #volatility
    exp = (demeaned_r**3).mean()
    return exp/sigma_r**3

def kurtosis(r):
    demeaned_r = r - r.mean()
    sigma_r = r.std(ddof=0) #volatility
    exp = (demeaned_r**4).mean()
    return exp/sigma_r**4

def is_normal(r, level=0.01):
    statistic,p_value = scipy.stats.jarque_bera(r)
    return p_value > level

def var_historic(r,level=5):
    if isinstance(r,pd.DataFrame):
        return r.aggregate(var_historic,level=level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r,level)
    else:
        raise TypeError("Expected series or dataframe")
    
def var_gaussian(r, level=5,modified=False):
    z = norm.ppf(level/100)
    if modified:
        s = skewness(r)
        k = kurtosis(r)
        z = (z +
              (z**2 - 1)*s/6 +
              (z**3 - 3*z)*(k-3)/24 -
              (2*z**3 - 5*z)*(s**2)/36
            )
    return -(r.mean() + z * r.std(ddof=0))

def cvar_historic(r, level=5):
    if isinstance(r,pd.DataFrame):
        is_beyond =  r<= - var_historic(r,level=level)
        return -r[is_beyond].mean()
    elif isinstance(r, pd.Series):
        return r.aggregate(cvar_historic,level=level)
    else:
        raise TypeError("Expected series or dataframe")