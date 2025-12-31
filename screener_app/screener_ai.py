import pandas as pd
import numpy as np
import sys
import os
import streamlit as st
import concurrent.futures

# Path setup
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Import from backend via root-relative paths
from backend.data_loader import fetch_latest_nav, fetch_fund_history
from backend.analytics import get_fund_metrics, filter_by_period, download_benchmark
# Import from local package via absolute path
from screener_app.utils import get_benchmark_for_category, calculate_sortino, calculate_alpha

@st.cache_data(show_spinner=False)
def perform_ai_ranking(nav_df, scheme_codes):
    """
    Processes a specific list of funds and returns a uniquely ranked dataframe.
    """
    if nav_df.empty or not scheme_codes:
        return pd.DataFrame()
        
    # Filter nav_df for the specific scheme codes provided
    target_funds = nav_df[nav_df['Scheme Code'].astype(str).isin([str(c) for c in scheme_codes])]
    
    if target_funds.empty:
        return pd.DataFrame()
        
    results = []
    
    # We need a progress bar for UI feedback
    progress_bar = st.progress(0, text=f"AI is analyzing {len(target_funds)} selected funds...")
    

    codes_to_fetch = [str(c) for c in target_funds['Scheme Code']]
    hist_cache = {}
    
    status_msg = st.empty()
    status_msg.info(f"⚡ AI Analyzing {len(codes_to_fetch)} funds...")
    
    # --- PRE-FETCH FUND HISTORIES ---
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_code = {executor.submit(fetch_fund_history, code): code for code in codes_to_fetch}
        for future in concurrent.futures.as_completed(future_to_code):
            code = future_to_code[future]
            try:
                hist_cache[code] = future.result()
            except:
                hist_cache[code] = pd.DataFrame()
    
    status_msg.empty()

    # --- PRE-FETCH BENCHMARKS IN PARALLEL ---
    # Identify unique benchmarks needed
    needed_benchmarks = set()
    for _, row in target_funds.iterrows():
        b_name = get_benchmark_for_category(row.get('Category', 'Unknown'), scheme_name=row['Scheme Name'])
        if b_name:
            needed_benchmarks.add(b_name)
    
    benchmark_cache = {}
    if needed_benchmarks:
        status_msg.info(f"⚡ AI fetching {len(needed_benchmarks)} benchmarks...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_bench = {executor.submit(download_benchmark, b, period="max"): b for b in needed_benchmarks}
            for future in concurrent.futures.as_completed(future_to_bench):
                b = future_to_bench[future]
                try:
                    # Use a timeout for the future itself as a safety net
                    benchmark_cache[b] = future.result(timeout=15)
                except:
                    benchmark_cache[b] = pd.DataFrame()
        status_msg.empty()

    for i, (idx, row) in enumerate(target_funds.iterrows()):
        code = str(row['Scheme Code'])
        name = row['Scheme Name']
        category = row.get('Category', 'Unknown')
        
        # Simple progress update
        progress_bar.progress((i + 1) / len(target_funds), text=f"Analyzing {name}...")
        
        nav_history = hist_cache.get(code, pd.DataFrame())
        if nav_history.empty:
            continue
            
        # 5Y window for quality stats
        nav_5y = filter_by_period(nav_history['nav'], "5Y")
        if len(nav_5y) < 100: # Need sufficient data
            continue
            
        m_stats = get_markov_stats(nav_history)
        
        # Benchmarking - ALREADY CACHED
        benchmark_name = get_benchmark_for_category(category, scheme_name=name)
        benchmark_series = benchmark_cache.get(benchmark_name, None) if benchmark_name else None
        
        # Get Raw Analytics
        kpis = get_fund_metrics(nav_5y, benchmark_series=benchmark_series)
        
        # We RE-CALCULATE Alpha here to be 100% sure it's using the right benchmark
        alpha_val = calculate_alpha(nav_5y, benchmark_series=benchmark_series, category=category) or 0
        
        # Add significant jitter to ensure unique ranks even if data is missing or identical
        # Using 0.01 range ensures we get distinct percentiles (1, 2, 3...) in the UI
        alpha_val += np.random.uniform(-0.01, 0.01)
        
        sortino = calculate_sortino(nav_5y) or 0
        down_cap = kpis.get('Downside Capture', 100)
        max_dd = kpis.get('Max Drawdown', 0)
        
        # Store RAW values for percentile ranking later
        results.append({
            "Scheme Code": code,
            "Scheme Name": name,
            "Category": category,
            "raw_alpha": alpha_val,
            "raw_persistence": m_stats['bull_persistence'],
            "raw_entropy": m_stats['entropy'],
            "raw_sortino": sortino,
            "raw_defense": 150 - down_cap,
            "raw_recovery": m_stats['recovery_prob'],
            "raw_mdd": -max_dd if max_dd else 0 # Higher is better
        })
        
    progress_bar.empty()
    
    if not results:
        return pd.DataFrame()
        
    df = pd.DataFrame(results)
    
    # --- RELATIVE PERCENTILE SCORING ---
    # This ensures that even if absolute values are similar, the relative order is preserved 
    # and the scores 'pop' visually for the user.
    
    # 1. Growth Score (Based on Alpha)
    df['Growth Score'] = df['raw_alpha'].rank(pct=True) * 100
    
    # 2. Consistency Score (Persistence + Predictability)
    predictability = (1 - df['raw_entropy']).clip(0, 1)
    df['Consistency'] = ((df['raw_persistence'].rank(pct=True) * 0.7) + (predictability.rank(pct=True) * 0.3)) * 100
    
    # 3. Safety Score (Sortino + Defense + MDD)
    df['Safety Score'] = ((df['raw_sortino'].rank(pct=True) * 0.4) + 
                          (df['raw_defense'].rank(pct=True) * 0.4) + 
                          (df['raw_mdd'].rank(pct=True) * 0.2)) * 100
    
    # Overall PAR Score (Weighted average)
    df['PAR Score'] = (df['Growth Score'] * 0.4) + (df['Consistency'] * 0.35) + (df['Safety Score'] * 0.25)
    
    # Add tiny noise for unique ranking
    df['PAR Score'] += np.random.uniform(0, 0.0001, size=len(df))
    df = df.sort_values(by="PAR Score", ascending=False).reset_index(drop=True)
    df.insert(0, "AI Rank", df.index + 1)
    
    # --- BADGES & VERDICTS ---
    def assign_badges(row):
        badges = []
        if row['PAR Score'] > 80: badges.append("⭐ ELITE")
        if row['Growth Score'] > 85: badges.append("🔥 GROWTH LEADER")
        if row['Consistency'] > 85: badges.append("🔄 HIGH CONSISTENCY")
        if row['Safety Score'] > 85: badges.append("🛡️ ULTRA SAFE")
        if row['raw_recovery'] > df['raw_recovery'].quantile(0.75): badges.append("📈 FAST RECOVERY")
        return " | ".join(badges) if badges else "⚖️ BALANCED"

    df['AI Badge'] = df.apply(assign_badges, axis=1)
    
    def get_verdict(row):
        rank = row['AI Rank']
        badges = row['AI Badge']
        growth = row['Growth Score']
        safety = row['Safety Score']
        
        if rank == 1: return "🏆 BEST OVERALL"
        if rank <= 3: return "💎 TOP PROSPECT"
        if growth > 80 and safety > 80: return "🤝 PERFECT BALANCE"
        if safety > 80: return "🔰 DEFENSIVE GEM"
        if growth > 80: return "🚀 ALPHA CHASER"
        return "⚖️ NEUTRAL"
        
    df['AI Verdict'] = df.apply(get_verdict, axis=1)
    
    cols_to_keep = [
        'AI Rank', 'Scheme Code', 'Scheme Name', 'Category', 
        'AI Badge', 'Growth Score', 'Consistency', 'Safety Score', 'AI Verdict', 'PAR Score'
    ]
    return df[cols_to_keep]

class CategoryAIRanker:
    """
    Ranks funds using Markov transitions and weighted multi-factor scoring.
    """
    
    def __init__(self):
        self.nav_df = fetch_latest_nav()

def get_markov_stats(nav_history):
    if nav_history.empty:
        return None
    returns = nav_history['nav'].pct_change().dropna()
    def classify(ret):
        if ret < -0.005: return 0 # Bear
        if ret > 0.005: return 2 # Bull
        return 1 # Neutral
    states = returns.apply(classify).values
    n_states = 3
    matrix = np.zeros((n_states, n_states))
    for i in range(len(states)-1):
        matrix[states[i]][states[i+1]] += 1
    row_sums = matrix.sum(axis=1)
    matrix = np.divide(matrix, row_sums[:, np.newaxis], where=row_sums[:, np.newaxis] != 0)
    
    # Calculate Entropy (Predictability)
    def calc_entropy(row):
        row = row[row > 0]
        return -np.sum(row * np.log2(row))
    
    # Weighted average entropy of states
    state_counts = pd.Series(states).value_counts(normalize=True)
    entropy = 0
    for s in range(n_states):
        s_prob = state_counts.get(s, 0)
        if s_prob > 0:
            entropy += s_prob * calc_entropy(matrix[s])
    
    # Normalize entropy (max for 3 states is log2(3) = 1.58)
    norm_entropy = entropy / 1.58
    
    return {
        "bull_persistence": matrix[2][2],
        "bear_persistence": matrix[0][0],
        "recovery_prob": matrix[0][2],
        "entropy": norm_entropy,
        "matrix": matrix
    }

# Note: CategoryAIRanker class removed to prevent Streamlit reload issues.
# Use perform_ai_ranking(nav_df, scheme_codes) directly.

