import pandas as pd
import numpy as np
from datetime import datetime
import re
from mftool import Mftool

def calculate_cagr(series, years):
    """
    Calculates the Compound Annual Growth Rate (CAGR) for a given series over 'years'.
    Returns float percentage (e.g., 15.4 for 15.4%) or None if insufficient data.
    """
    if series.empty or len(series) < 2:
        return None
        
    start_date = series.index[-1] - pd.DateOffset(years=years)
    available_history = series[series.index >= start_date]
    
    if available_history.empty:
        return None
        
    # Check if we actually have enough history (e.g. at least 90% of the requested period)
    actual_start_date = available_history.index[0]
    days_diff = (series.index[-1] - actual_start_date).days
    required_days = years * 365
    
    if days_diff < (required_days * 0.9): # A little buffer
        return None
        
    start_val = available_history.iloc[0]
    end_val = series.iloc[-1]
    
    if start_val == 0:
        return None
        
    # Standard CAGR formula: (End/Start)^(1/n) - 1
    # We use (days_diff / 365.25) for precise 'n'
    n = days_diff / 365.25
    cagr = ( (end_val / start_val) ** (1/n) ) - 1
    
    return cagr * 100

def calculate_volatility(series):
    """Annualized Volatility (Standard Deviation)"""
    if series.empty or len(series) < 2: return None
    daily_ret = series.pct_change().dropna()
    return daily_ret.std() * np.sqrt(252) * 100

def calculate_max_drawdown(series):
    """Maximum Drawdown in percentage (positive value)"""
    if series.empty or len(series) < 2: return None
    rollup = series.cummax()
    drawdown = (series - rollup) / rollup
    min_dd = drawdown.min()
    return abs(min_dd) * 100 if min_dd < 0 else 0.0

def calculate_sharpe(series, risk_free_rate=0.06):
    """Sharpe Ratio (Annualized)"""
    if series.empty or len(series) < 2: return None
    
    # CAGR
    days = (series.index[-1] - series.index[0]).days
    if days < 30: return None
    cagr = ((series.iloc[-1] / series.iloc[0]) ** (365.25/days)) - 1
    
    # Volatility
    daily_ret = series.pct_change().dropna()
    vol = daily_ret.std() * np.sqrt(252)
    
    if vol == 0: return 0.0
    return (cagr - risk_free_rate) / vol

def calculate_sortino(series, risk_free_rate=0.06):
    """Sortino Ratio (Annualized)"""
    if series.empty or len(series) < 2: return None
    
    days = (series.index[-1] - series.index[0]).days
    if days < 30: return None
    cagr = ((series.iloc[-1] / series.iloc[0]) ** (365.25/days)) - 1
    
    daily_ret = series.pct_change().dropna()
    negative_rets = daily_ret[daily_ret < 0]
    
    if negative_rets.empty: return 0.0 # No downside risk
    
    downside_std = negative_rets.std() * np.sqrt(252)
    if downside_std == 0: return 0.0
    
    return (cagr - risk_free_rate) / downside_std

def calculate_absolute_return(series, months):
    """
    Calculate absolute return over specified months
    Returns percentage (e.g., 15.4 for 15.4%) or None if insufficient data
    """
    if series.empty or len(series) < 2:
        return None
    
    start_date = series.index[-1] - pd.DateOffset(months=months)
    available_history = series[series.index >= start_date]
    
    if available_history.empty:
        return None
    
    # Check if we have at least 80% of the requested period
    actual_start_date = available_history.index[0]
    days_diff = (series.index[-1] - actual_start_date).days
    required_days = months * 30
    
    if days_diff < (required_days * 0.8):
        return None
    
    start_val = available_history.iloc[0]
    end_val = series.iloc[-1]
    
    if start_val == 0:
        return None
    
    absolute_return = ((end_val - start_val) / start_val) * 100
    return absolute_return

def calculate_rolling_return(series, years):
    """
    Calculate average annual rolling return over specified years
    Returns percentage or None if insufficient data
    """
    if series.empty or len(series) < 2:
        return None
    
    # Need at least 'years' of data
    min_date = series.index[-1] - pd.DateOffset(years=years)
    if series.index[0] > min_date:
        return None
    
    # Calculate rolling returns
    rolling_returns = []
    
    # Create 1-year windows
    for i in range(len(series) - 252):  # Assuming daily data, 252 trading days = 1 year
        if i + 252 < len(series):
            start_val = series.iloc[i]
            end_val = series.iloc[i + 252]
            if start_val > 0:
                annual_return = ((end_val / start_val) - 1) * 100
                rolling_returns.append(annual_return)
    
    if not rolling_returns:
        return None
    
    return np.mean(rolling_returns)

def calculate_simple_rolling_return(series, years):
    """
    Calculates the return over the last 'years' period (Point-to-Point).
    Equivalent to CAGR for the specified period ending today.
    """
    return calculate_cagr(series, years)

def get_benchmark_for_category(category, scheme_name=None):
    """
    Determine appropriate benchmark based on fund category and optionally scheme name
    Returns benchmark ticker name
    """
    category_lower = category.lower() if category else ""
    scheme_name_lower = scheme_name.lower() if scheme_name else ""
    
    # Check Scheme Name matches first (more specific)
    if 'gold' in scheme_name_lower:
        return "GOLD"
    if 'silver' in scheme_name_lower:
        return "SILVER"
    
    # Equity - Large Cap
    if any(x in category_lower for x in ['large cap', 'blue chip', 'index fund']):
        return "NIFTY 50"
    
    # Equity - Mid Cap
    elif any(x in category_lower for x in ['mid cap', 'midcap']):
        return "NIFTY MIDCAP 150"
    
    # Equity - Small Cap
    elif any(x in category_lower for x in ['small cap', 'smallcap']):
        return "NIFTY SMALLCAP 250"
    
    # Equity - Multi/Flexi Cap
    elif any(x in category_lower for x in ['flexi cap', 'multi cap', 'diversified']):
        return "NIFTY 500"
    
    # Sectoral - Banking
    elif 'bank' in category_lower:
        return "NIFTY BANK"
    
    # Commodity - Gold (Category based)
    elif 'gold' in category_lower:
        return "GOLD"
    
    # Commodity - Silver (Category based)
    elif 'silver' in category_lower:
        return "SILVER"
    
    # Debt funds
    elif any(x in category_lower for x in ['debt', 'gilt', 'bond', 'corporate bond', 'credit risk', 'floater', 'psu']):
        return "NIFTY 10 YR BENCHMARK G-SEC"
    
    # Hybrid - Aggressive/Balanced (Equity oriented)
    elif 'aggressive' in category_lower or 'balanced' in category_lower:
        return "NIFTY 50"
        
    # Default to Nifty 50 for other equity schemes
    elif 'equity' in category_lower:
        return "NIFTY 50"
    
    # Default fallback
    else:
        return None

def calculate_alpha(series, benchmark_series=None, risk_free_rate=0.06, category=None):
    """
    Calculate Alpha (excess return vs benchmark adjusted for risk)
    Alpha = Fund Return - (Risk Free Rate + Beta * (Benchmark Return - Risk Free Rate))
    
    Returns percentage or None if insufficient data or no benchmark
    """
    if series.empty or len(series) < 252:  # Need at least 1 year of data
        return None
    
    # If no benchmark provided but category is given, fetch appropriate benchmark
    if benchmark_series is None or benchmark_series.empty:
        if category:
            benchmark_name = get_benchmark_for_category(category)
            if not benchmark_name:
                return None  # No appropriate benchmark for this category
            
            # Import here to avoid circular dependency
            try:
                from backend.analytics import download_benchmark
                benchmark_series = download_benchmark(benchmark_name, period="max")
                if benchmark_series.empty:
                    return None
            except:
                return None
        else:
            return None
    
    # Align dates between fund and benchmark
    # Ensure DatetimeIndex and TZ-naive
    if not isinstance(series.index, pd.DatetimeIndex):
        series.index = pd.to_datetime(series.index)
    if not isinstance(benchmark_series.index, pd.DatetimeIndex):
        benchmark_series.index = pd.to_datetime(benchmark_series.index)
    
    if series.index.tz is not None: series.index = series.index.tz_localize(None)
    if benchmark_series.index.tz is not None: benchmark_series.index = benchmark_series.index.tz_localize(None)

    common_dates = series.index.intersection(benchmark_series.index)
    if len(common_dates) < 50:  # Relaxed constraint
        print(f"DEBUG ALPHA FAIL: Common dates {len(common_dates)} < 50. Series range: {series.index.min()} to {series.index.max()}. Bench range: {benchmark_series.index.min()} to {benchmark_series.index.max()}")
        return None
    
    fund_aligned = series.loc[common_dates]
    bench_aligned = benchmark_series.loc[common_dates]
    
    # Calculate returns
    fund_returns = fund_aligned.pct_change().dropna()
    bench_returns = bench_aligned.pct_change().dropna()
    
    if len(fund_returns) < 100:  # Need sufficient data points
        return None
    
    # Calculate Beta using covariance
    covariance = np.cov(fund_returns, bench_returns)[0][1]
    bench_variance = np.var(bench_returns)
    
    if bench_variance == 0:
        return None
    
    beta = covariance / bench_variance
    
    # Calculate annualized returns
    days = (fund_aligned.index[-1] - fund_aligned.index[0]).days
    if days < 30:
        return None
    
    fund_cagr = ((fund_aligned.iloc[-1] / fund_aligned.iloc[0]) ** (365.25/days)) - 1
    bench_cagr = ((bench_aligned.iloc[-1] / bench_aligned.iloc[0]) ** (365.25/days)) - 1
    
    # Calculate Alpha
    # Alpha = Fund Return - [Risk Free Rate + Beta * (Benchmark Return - Risk Free Rate)]
    alpha = fund_cagr - (risk_free_rate + beta * (bench_cagr - risk_free_rate))
    
    print(f"DEBUG ALPHA: FundCAGR={fund_cagr:.4f}, BenchCAGR={bench_cagr:.4f}, Beta={beta:.4f}, Alpha={alpha:.4f}")
    
    return alpha * 100  # Return as percentage


def calculate_tracking_error(series, benchmark_series=None, category=None):
    """
    Calculate Tracking Error - annualized standard deviation of difference 
    in returns between fund and benchmark
    
    Returns percentage or None if insufficient data
    """
    if series.empty or len(series) < 252:  # Need at least 1 year
        return None
    
    # If no benchmark provided but category is given, fetch appropriate benchmark
    if benchmark_series is None or benchmark_series.empty:
        if category:
            benchmark_name = get_benchmark_for_category(category)
            if not benchmark_name:
                return None
            
            try:
                from backend.analytics import download_benchmark
                benchmark_series = download_benchmark(benchmark_name, period="max")
                if benchmark_series.empty:
                    return None
            except:
                return None
        else:
            return None
    
    # Align dates
    if not isinstance(series.index, pd.DatetimeIndex):
        series.index = pd.to_datetime(series.index)
    if not isinstance(benchmark_series.index, pd.DatetimeIndex):
        benchmark_series.index = pd.to_datetime(benchmark_series.index)
        
    if series.index.tz is not None: series.index = series.index.tz_localize(None)
    if benchmark_series.index.tz is not None: benchmark_series.index = benchmark_series.index.tz_localize(None)

    common_dates = series.index.intersection(benchmark_series.index)
    if len(common_dates) < 50:
        return None
    
    fund_aligned = series.loc[common_dates]
    bench_aligned = benchmark_series.loc[common_dates]
    
    # Calculate daily returns
    fund_returns = fund_aligned.pct_change().dropna()
    bench_returns = bench_aligned.pct_change().dropna()
    
    if len(fund_returns) < 100:
        return None
    
    # Calculate difference in returns
    return_diff = fund_returns - bench_returns
    
    # Annualized standard deviation of differences
    tracking_error = return_diff.std() * np.sqrt(252) * 100
    
    return tracking_error


def get_sebi_risk_category(volatility):
    """
    Determine SEBI Risk Category based on volatility
    Low: < 10%, Moderately Low: 10-15%, Moderate: 15-20%, 
    Moderately High: 20-25%, High: > 25%
    """
    if volatility is None:
        return "N/A"
    
    if volatility < 10:
        return "Low"
    elif volatility < 15:
        return "Moderately Low"
    elif volatility < 20:
        return "Moderate"
    elif volatility < 25:
        return "Moderately High"
    else:
        return "High"


def calculate_information_ratio(series, benchmark_series=None, category=None):
    """
    Calculate Information Ratio - measures excess return per unit of tracking error
    Information Ratio = (Fund Return - Benchmark Return) / Tracking Error
    
    Returns ratio or None if insufficient data
    """
    if series.empty or len(series) < 252:  # Need at least 1 year
        return None
    
    # If no benchmark provided but category is given, fetch appropriate benchmark
    if benchmark_series is None or benchmark_series.empty:
        if category:
            benchmark_name = get_benchmark_for_category(category)
            if not benchmark_name:
                return None
            
            try:
                from backend.analytics import download_benchmark
                benchmark_series = download_benchmark(benchmark_name, period="max")
                if benchmark_series.empty:
                    return None
            except:
                return None
        else:
            return None
    
    # Align dates
    # Ensure both are DatetimeIndex
    if not isinstance(series.index, pd.DatetimeIndex):
        series.index = pd.to_datetime(series.index)
    if not isinstance(benchmark_series.index, pd.DatetimeIndex):
        benchmark_series.index = pd.to_datetime(benchmark_series.index)
        
    # Remove timezone info if any
    if series.index.tz is not None: series.index = series.index.tz_localize(None)
    if benchmark_series.index.tz is not None: benchmark_series.index = benchmark_series.index.tz_localize(None)

    common_dates = series.index.intersection(benchmark_series.index)
    if len(common_dates) < 50:  # Relaxed from 252 to 50 to allow recent funds/proxies to work
        return None
    
    fund_aligned = series.loc[common_dates]
    bench_aligned = benchmark_series.loc[common_dates]
    
    # Calculate annualized returns
    days = (fund_aligned.index[-1] - fund_aligned.index[0]).days
    if days < 30:
        return None
    
    fund_cagr = ((fund_aligned.iloc[-1] / fund_aligned.iloc[0]) ** (365.25/days)) - 1
    bench_cagr = ((bench_aligned.iloc[-1] / bench_aligned.iloc[0]) ** (365.25/days)) - 1
    
    # Calculate tracking error
    fund_returns = fund_aligned.pct_change().dropna()
    bench_returns = bench_aligned.pct_change().dropna()
    
    if len(fund_returns) < 100:
        return None
    
    return_diff = fund_returns - bench_returns
    tracking_error = return_diff.std() * np.sqrt(252)
    
    if tracking_error == 0:
        return None
    
    # Information Ratio
    excess_return = fund_cagr - bench_cagr
    information_ratio = excess_return / tracking_error
    
    return information_ratio



def calculate_upside_downside_capture(series, benchmark_series):
    """
    Calculate Upside and Downside Capture Ratios
    Returns tuple (upside_capture, downside_capture) or (None, None)
    """
    if series.empty or benchmark_series is None or benchmark_series.empty:
        return None, None
        
    # Align dates
    if not isinstance(series.index, pd.DatetimeIndex):
        series.index = pd.to_datetime(series.index)
    if not isinstance(benchmark_series.index, pd.DatetimeIndex):
        benchmark_series.index = pd.to_datetime(benchmark_series.index)
        
    # Remove timezone info if any
    if series.index.tz is not None: series.index = series.index.tz_localize(None)
    if benchmark_series.index.tz is not None: benchmark_series.index = benchmark_series.index.tz_localize(None)

    common_dates = series.index.intersection(benchmark_series.index)
    if len(common_dates) < 50:
        return None, None
    
    fund_aligned = series.loc[common_dates]
    bench_aligned = benchmark_series.loc[common_dates]
    
    # Calculate returns
    fund_returns = fund_aligned.pct_change().dropna()
    bench_returns = bench_aligned.pct_change().dropna()
    
    if len(fund_returns) < 20:
        return None, None
        
    # Upside Capture
    up_periods = bench_returns > 0
    if up_periods.sum() > 0:
        fund_up = fund_returns[up_periods].mean()
        bench_up = bench_returns[up_periods].mean()
        upside = (fund_up / bench_up * 100) if bench_up != 0 else None
    else:
        upside = None
        
    # Downside Capture
    down_periods = bench_returns < 0
    if down_periods.sum() > 0:
        fund_down = fund_returns[down_periods].mean()
        bench_down = bench_returns[down_periods].mean()
        downside = (fund_down / bench_down * 100) if bench_down != 0 else None
    else:
        downside = None
        
    return upside, downside


# --- METADATA FETCHING ---

# Initialize Mftool once if possible, or per call (it's lightweight, just a wrapper)
mf = Mftool()

def get_fund_meta(code):
    """
    Fetches AUM and Age for a given scheme code using mftool.
    Returns dict with 'Age' (years) and 'AUM' (Crores), or None values if not found.
    """
    try:
        details = mf.get_scheme_details(code)
        
        # details example: 
        # {'fund_house': '...', 'scheme_type': '...', 'scheme_category': '...', 
        #  'scheme_code': '...', 'scheme_name': '...', 'scheme_start_date': '01-Jan-2013', 
        #  'scheme_nav': '...', 'scheme_aum': 'Rs. 15,000 Cr', ...}
        
        age = None
        aum = None
        
        # 1. Calculate Age
        start_date_str = details.get('scheme_start_date', {}).get('date', '') if isinstance(details.get('scheme_start_date'), dict) else details.get('scheme_start_date', '')
        # mftool sometimes returns dict for date, sometimes string. Handle both.
        # Check if details is just a message "False" or valid dict
        
        if not isinstance(details, dict) or 'scheme_start_date' not in details:
             return {'Age': None, 'AUM': None}

        date_str = details['scheme_start_date']
        if date_str:
            try:
                # Format is usually '01-Jan-2013'
                launch_date = datetime.strptime(date_str, '%d-%b-%Y')
                delta = datetime.now() - launch_date
                age = delta.days / 365.25
            except:
                pass
                
        # 2. Parse AUM
        # Format: "Rs. 1,234.56 Cr" or "₹ 1,234.56 Cr"
        aum_str = details.get('scheme_aum', '')
        if aum_str:
            try:
                # Use regex to find the number pattern (digits, commas, optional decimal)
                match = re.search(r"([\d,]+\.?\d*)", aum_str)
                if match:
                    clean_str = match.group(1).replace(',', '')
                    aum = float(clean_str)
            except:
                pass
                
        return {'Age': age, 'AUM': aum}
            
    except Exception as e:
        # print(f"Meta fetch error: {e}")
        return {'Age': None, 'AUM': None}

def calculate_period_metrics(fund_series, benchmark_series, years):
    """
    Calculate all risk metrics for a specific time period
    Returns dict with Alpha, Beta, Sharpe, Std Dev, Upside Capture, Downside Capture
    """
    if fund_series.empty or len(fund_series) < 2:
        return {
            'alpha': None, 'beta': None, 'sharpe': None, 
            'std_dev': None, 'upside_capture': None, 'downside_capture': None
        }
    
    # Get data for the specified period
    days_needed = int(years * 365.25)
    end_date = fund_series.index[-1]
    start_date = end_date - pd.Timedelta(days=days_needed)
    
    # Filter to period
    # Ensure fund data has just normalized timestamps
    if not fund_series.empty:
        if not isinstance(fund_series.index, pd.DatetimeIndex):
             fund_series.index = pd.to_datetime(fund_series.index)
        if fund_series.index.tz is not None:
             fund_series.index = fund_series.index.tz_localize(None)
        fund_series.index = fund_series.index.normalize()
        fund_series = fund_series[~fund_series.index.duplicated(keep='first')]

    period_fund = fund_series[fund_series.index >= start_date]

    if len(period_fund) < 2:
        return {
            'alpha': None, 'beta': None, 'sharpe': None,
            'std_dev': None, 'upside_capture': None, 'downside_capture': None
        }
    
    # Calculate metrics
    result = {}
    
    # 1. Standard Deviation (Volatility)
    daily_ret = period_fund.pct_change().dropna()
    result['std_dev'] = daily_ret.std() * np.sqrt(252) * 100 if len(daily_ret) > 0 else None
    
    # 2. Sharpe Ratio
    if len(period_fund) >= 2:
        days = (period_fund.index[-1] - period_fund.index[0]).days
        if days >= 30:
            cagr = ((period_fund.iloc[-1] / period_fund.iloc[0]) ** (365.25/days)) - 1
            vol = daily_ret.std() * np.sqrt(252)
            result['sharpe'] = (cagr - 0.06) / vol if vol > 0 else None
        else:
            result['sharpe'] = None
    else:
        result['sharpe'] = None
    
    # 3. Alpha and Beta (require benchmark)
    if benchmark_series is not None and not benchmark_series.empty:
        # Normalize benchmark index
        bench_norm = benchmark_series.copy()
        bench_norm = benchmark_series.copy()
        if not isinstance(bench_norm.index, pd.DatetimeIndex):
             bench_norm.index = pd.to_datetime(bench_norm.index)
        if bench_norm.index.tz is not None:
             bench_norm.index = bench_norm.index.tz_localize(None)
        bench_norm.index = bench_norm.index.normalize()
        
        period_bench = bench_norm[bench_norm.index >= start_date]
        
        if len(period_bench) >= 2:
            # Align dates
            common_dates = period_fund.index.intersection(period_bench.index)
            if len(common_dates) > 5:
                aligned_fund = period_fund.loc[common_dates]
                aligned_bench = period_bench.loc[common_dates]
                
                fund_ret = aligned_fund.pct_change().dropna()
                bench_ret = aligned_bench.pct_change().dropna()
                
                common_idx = fund_ret.index.intersection(bench_ret.index)
                if len(common_idx) > 10:
                    fund_ret = fund_ret.loc[common_idx]
                    bench_ret = bench_ret.loc[common_idx]
                    
                    # Beta
                    cov = np.cov(fund_ret, bench_ret)[0][1]
                    bench_var = np.var(bench_ret)
                    result['beta'] = cov / bench_var if bench_var > 0 else None
                    
                    # Alpha & Beta using direct calculation on aligned data
                    fund_days = (aligned_fund.index[-1] - aligned_fund.index[0]).days
                    bench_days = (aligned_bench.index[-1] - aligned_bench.index[0]).days
                    
                    if fund_days > 20 and bench_days > 20: # Minimal requirement
                        # Annualized returns
                        fund_ann_ret = ((aligned_fund.iloc[-1] / aligned_fund.iloc[0]) ** (365.25/fund_days)) - 1
                        bench_ann_ret = ((aligned_bench.iloc[-1] / aligned_bench.iloc[0]) ** (365.25/bench_days)) - 1
                        
                        # Alpha (Jensen's)
                        result['alpha'] = (fund_ann_ret - (0.06 + result['beta'] * (bench_ann_ret - 0.06))) * 100
                    else:
                        result['alpha'] = None
                    
                    # Upside/Downside Capture
                    up_periods = bench_ret > 0
                    down_periods = bench_ret < 0
                    
                    if up_periods.sum() > 0:
                        fund_up = fund_ret[up_periods].mean()
                        bench_up = bench_ret[up_periods].mean()
                        result['upside_capture'] = (fund_up / bench_up * 100) if bench_up != 0 else None
                    else:
                        result['upside_capture'] = None
                    
                    if down_periods.sum() > 0:
                        fund_down = fund_ret[down_periods].mean()
                        bench_down = bench_ret[down_periods].mean()
                        result['downside_capture'] = (fund_down / bench_down * 100) if bench_down != 0 else None
                    else:
                        result['downside_capture'] = None
                else:
                    result['alpha'] = None
                    result['beta'] = None
                    result['upside_capture'] = None
                    result['downside_capture'] = None
            else:
                result['alpha'] = None
                result['beta'] = None
                result['upside_capture'] = None
                result['downside_capture'] = None
        else:
            result['alpha'] = None
            result['beta'] = None
            result['upside_capture'] = None
            result['downside_capture'] = None
    else:
        result['alpha'] = None
        result['beta'] = None
        result['upside_capture'] = None
        result['downside_capture'] = None
    
    return result
