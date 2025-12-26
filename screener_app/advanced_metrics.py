"""
Advanced metrics calculations for mutual fund screening
Includes Rolling Returns, Capture Ratios, and Quartile Rankings
"""

import pandas as pd
import numpy as np


def calculate_simple_rolling_return(series, window_years):
    """
    Calculate simple average rolling return for a given window
    Returns: average annualized rolling return as percentage
    """
    if series.empty or len(series) < window_years * 200: # Slightly more lenient
        return None
    
    # Ensure series is sorted and has normalized timestamp index
    s = series.sort_index()
    s.index = pd.to_datetime(s.index).normalize()
    
    rolling_returns = []
    window_days = int(window_years * 252)
    
    # Calculate rolling returns with step of 5 to speed up
    for i in range(0, len(s) - window_days, 5):
        start_val = s.iloc[i]
        end_val = s.iloc[i + window_days]
        
        if start_val > 0 and end_val > 0:
            # Correct CAGR formula: (end/start)^(1/years) - 1
            annual_return = ((end_val / start_val) ** (1.0 / window_years) - 1) * 100
            rolling_returns.append(annual_return)
    
    if not rolling_returns:
        return None
    
    return np.mean(rolling_returns)


def calculate_upside_downside_capture(series, benchmark_series):
    """
    Calculate upside and downside capture ratios
    """
    if series.empty or benchmark_series is None or benchmark_series.empty:
        return None, None
    
    # Normalize indices to Timestamps (normalized) for robust alignment
    s1 = series.copy()
    s1.index = pd.to_datetime(s1.index).normalize()
    s1 = s1[~s1.index.duplicated(keep='first')]
    
    b1 = benchmark_series.copy()
    b1.index = pd.to_datetime(b1.index).normalize()
    
    # Align dates
    common_dates = s1.index.intersection(b1.index)
    if len(common_dates) < 60:
        return None, None
    
    fund_aligned = s1.loc[common_dates]
    bench_aligned = b1.loc[common_dates]
    
    # Calculate daily returns
    fund_returns = fund_aligned.pct_change().dropna()
    bench_returns = bench_aligned.pct_change().dropna()
    
    if len(fund_returns) < 20:
        return None, None
    
    # Separate up and down periods
    up_periods = bench_returns > 0
    down_periods = bench_returns < 0
    
    # Calculate average returns in up/down periods
    fund_up_avg = fund_returns[up_periods].mean() if up_periods.sum() > 0 else 0
    bench_up_avg = bench_returns[up_periods].mean() if up_periods.sum() > 0 else 0
    
    fund_down_avg = fund_returns[down_periods].mean() if down_periods.sum() > 0 else 0
    bench_down_avg = bench_returns[down_periods].mean() if down_periods.sum() > 0 else 0
    
    # Calculate capture ratios
    upside_capture = (fund_up_avg / bench_up_avg * 100) if bench_up_avg != 0 else None
    downside_capture = (fund_down_avg / bench_down_avg * 100) if bench_down_avg != 0 else None
    
    return upside_capture, downside_capture


def calculate_quartile_rank(fund_return, category_returns):
    """
    Calculate quartile rank for a fund within its category
    
    Returns: 1 (top quartile), 2, 3, or 4 (bottom quartile)
    """
    if fund_return is None or pd.isna(fund_return):
        return None
    
    if not category_returns or len(category_returns) < 4:
        return None
    
    # Remove None/NaN values
    valid_returns = [r for r in category_returns if r is not None and not pd.isna(r)]
    
    if len(valid_returns) < 4:
        return None
    
    # Calculate quartiles
    q1 = np.percentile(valid_returns, 75)  # Top 25%
    q2 = np.percentile(valid_returns, 50)  # Median
    q3 = np.percentile(valid_returns, 25)  # Bottom 25%
    
    # Determine quartile (higher returns = better = lower quartile number)
    if fund_return >= q1:
        return 1
    elif fund_return >= q2:
        return 2
    elif fund_return >= q3:
        return 3
    else:
        return 4
