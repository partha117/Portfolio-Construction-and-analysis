import pandas as pd
import scipy
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize


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

def get_ind_size():
    ind = pd.read_csv("data/ind30_m_size.csv",header=0,index_col=0,parse_dates=True)
    ind.index = pd.to_datetime(ind.index,format= "%Y%m").to_period('M')
    ind.columns = ind.columns.str.strip()
    return ind

def get_ind_nfirms():
    ind = pd.read_csv("data/ind30_m_nfirms.csv",header=0,index_col=0,parse_dates=True)
    ind.index = pd.to_datetime(ind.index,format= "%Y%m").to_period('M')
    ind.columns = ind.columns.str.strip()
    return ind

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
        
def portfolio_return(weights, returns):
    """
    weights -> returns
    """
    return weights.T @ returns

def portfolio_vol(weights, covmat):
    """
    weights -> vol
    """
    return (weights.T @ covmat @ weights)**0.5

def plot_ef2(n_points, er, cov):
    """
    Plot 2 asset efficient frontier
    """
    if er.shape[0] != 2:
        raise ValueError("Can only plot 2 asset frontier")
    weights = [np.array([w,1-w]) for w in np.linspace(0,1,n_points)]
    rets = [portfolio_return(w,er) for w in weights]
    vols = [portfolio_vol(w,cov) for w in weights]
    ef = pd.DataFrame({"Returns":rets,"Volatility":vols})
    return ef.plot.line(x="Volatility",y="Returns",style=".-")


def minimize_volatility(target_return, er, cov):
    """
    target return -> w
    """
    n = er.shape[0]
    init_guess = np.repeat(1/n,n)
    bounds = ((0.0,1.0),)*n
    return_is_target = {
        'type':'eq',
        'args': (er,),
        'fun': lambda weights, er : target_return - portfolio_return(weights,er)
    }
    weights_sum_to_one = {
        'type':'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }
    results = minimize(portfolio_vol,init_guess,
                       args=(cov,),
                       method='SLSQP',
                       options={'disp':False},
                       constraints=(return_is_target,weights_sum_to_one),
                       bounds=bounds
                      )
    return results.x


def optimal_weights(n_points,er,cov):
    """
    list of weights to run optimizer on to minimize volatility
    """
    target_rs = np.linspace(er.min(), er.max(), n_points)
    weights = [ minimize_volatility(target_return, er, cov) for target_return in target_rs]
    return weights

def maximize_sharpe_ratio(risk_free_rate, er, cov):
    """
    risk free rate + ER + Cov -> w
    """
    n = er.shape[0]
    init_guess = np.repeat(1/n,n)
    bounds = ((0.0,1.0),)*n
    weights_sum_to_one = {
        'type':'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }
    def neg_sharpe_ratio(weights, risk_free_rate, er, cov):
        """
        Returns the engative of sharpe ratio
        """
        returns = portfolio_return(weights,er)
        volatility = portfolio_vol(weights,cov)
        return -((returns - risk_free_rate)/volatility)
            
    results = minimize(neg_sharpe_ratio,init_guess,
                       args=(risk_free_rate, er, cov,),
                       method='SLSQP',
                       options={'disp':False},
                       constraints=(weights_sum_to_one),
                       bounds=bounds
                      )
    return results.x

def gmv(cov):
    """
    Returns the weight of global maximum  portfolio
    """
    n = cov.shape[0]
    return maximize_sharpe_ratio(0,np.repeat(1,n),cov)
    
    
    
def plot_efN(n_points, er, cov, show_cml=True, style='.-',risk_free_rate=0,show_ew=True, show_gmv=True):
    """
    Plot 2 asset efficient frontier
    """

    weights = optimal_weights(n_points,er,cov)
    rets = [portfolio_return(w,er) for w in weights]
    vols = [portfolio_vol(w,cov) for w in weights]
    ef = pd.DataFrame({"Returns":rets,"Volatility":vols})
    ax = ef.plot.line(x="Volatility",y="Returns",style=".-")
    
    if show_ew:
        n = er.shape[0]
        w_ew = np.repeat(1/n,n)
        r_ew = portfolio_return(w_ew,er)
        vol_ew = portfolio_vol(w_ew,cov)
        ax.plot([vol_ew],[r_ew],color='red',marker='o',markersize=12)
        
    if show_gmv:
        w_gmv = gmv(cov)
        r_gmv = portfolio_return(w_gmv,er)
        vol_gmv = portfolio_vol(w_gmv,cov)
        ax.plot([vol_gmv],[r_gmv],color='goldenrod',marker='o',markersize=12)
        
    
    if show_cml:
        ax.set_xlim(left=0)
        weight_max_sharpe_ratio = maximize_sharpe_ratio(risk_free_rate,er,cov)
        return_max_sharpe_ratio = portfolio_return(weight_max_sharpe_ratio,er)
        volatility_max_sharpe_ratio = portfolio_vol(weight_max_sharpe_ratio,cov)


        # Add CML
        cml_x = [0,volatility_max_sharpe_ratio]
        cml_y = [risk_free_rate, return_max_sharpe_ratio]
        ax.plot(cml_x,cml_y,color='green',marker='o',linestyle='dashed',markersize=12,linewidth=2)
    return ax