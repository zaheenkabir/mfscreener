import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import concurrent.futures
from datetime import datetime

# Path setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.data_loader import fetch_latest_nav, fetch_fund_history
from backend.analytics import download_benchmark
from screener_app.screener_ai import perform_ai_ranking
from screener_app.formula_engine import (
    get_premade_formulas, get_formula_help_text, 
    evaluate_formula, prepare_fund_data, validate_formula
)
from screener_app.utils import (
    get_benchmark_for_category, calculate_cagr, calculate_volatility,
    calculate_max_drawdown, calculate_sharpe, calculate_sortino,
    calculate_rolling_return, calculate_alpha, calculate_tracking_error,
    calculate_information_ratio, get_sebi_risk_category, calculate_period_metrics
)
from dashboard.ui_components import (
    apply_custom_css, metric_card, get_neon_color, style_financial_dataframe,
    render_welcome_card, render_compliance_footer
)
from screener_app.advanced_metrics import (
    calculate_simple_rolling_return, calculate_upside_downside_capture, calculate_quartile_rank
)

st.set_page_config(
    page_title="MF Screener",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
apply_custom_css()


def main():
    st.title("🔍 Mutual Fund Screener")
    st.markdown("### Advanced Mutual Fund Screener & Performance Analysis")
    st.caption("Filter funds and analyze returns. Built with robust AMFI data.")

    # --- LOAD DATA ---
    with st.spinner("Loading Master Fund List..."):
        df_master = fetch_latest_nav()
    
    if df_master.empty:
        st.error("Failed to load fund data. Please check connection.")
        return

    # --- SIDEBAR FILTERS ---
    st.sidebar.header("Filters")
    
    # 1. Provide a Search Box
    search_query = st.sidebar.text_input("Search Fund Name", placeholder="e.g. Parag Parikh")
    
    # 2. Fund House Filter
    all_amcs = sorted(df_master['Fund House'].unique().tolist())
    selected_amcs = st.sidebar.multiselect("Fund House (AMC)", all_amcs)
    
    # 3. Category Filter (Hierarchical)
    st.sidebar.subheader("Category")
    
    CATEGORY_HIERARCHY = {
    "Equity": [
        "Equity Scheme - Large Cap Fund", "Equity Scheme - Large & Mid Cap Fund", 
        "Equity Scheme - Mid Cap Fund", "Equity Scheme - Small Cap Fund",
        "Equity Scheme - ELSS", "Equity Scheme - Contra Fund", 
        "Equity Scheme - Dividend Yield Fund", "Equity Scheme - Value Fund", 
        "Equity Scheme - Focused Fund", "Equity Scheme - Multi Cap Fund", 
        "Equity Scheme - Flexi Cap Fund", "Equity Scheme - Sectoral/ Thematic"
    ],

    "Hybrid": [
        "Hybrid Scheme - Conservative Hybrid Fund", "Hybrid Scheme - Balanced Hybrid Fund", 
        "Hybrid Scheme - Aggressive Hybrid Fund",
        "Hybrid Scheme - Dynamic Asset Allocation or Balanced Advantage", 
        "Hybrid Scheme - Equity Savings", "Hybrid Scheme - Multi Asset Allocation",
        "Hybrid Scheme - Arbitrage Fund"
    ],
    "Commodity": [
        "Other Scheme - Other  ETFs", 
        "Silver ETF", "Gold ETF"
    ],
    "Global Funds": [
        "Other Scheme - FoF Overseas"
    ],
    "Other": [
        "Other Scheme - Index Funds", 
        "Other Scheme - Other  ETFs",
        "Solution Oriented Scheme - Children's Fund",
        "Solution Oriented Scheme - Retirement Fund"
    ]
}
    
    selected_subcats = []
    
    def clean_category_name(name):
        """Removes AMFI prefixes for cleaner UI display."""
        prefixes = [
            "Equity Scheme - ", "Debt Scheme - ", "Hybrid Scheme - ", 
            "Other Scheme - ", "Solution Oriented Scheme - "
        ]
        for p in prefixes:
            if name.startswith(p):
                return name.replace(p, "")
        return name

    for broad_cat, sub_cats in CATEGORY_HIERARCHY.items():
        with st.sidebar.expander(broad_cat, expanded=False):
            # Allow selecting sub-categories
            selected = st.multiselect(
                f"Select {broad_cat} Schemes", 
                sub_cats,
                format_func=clean_category_name
            )
            if selected:
                selected_subcats.extend(selected)
                
    # 4. Plan Filter
    st.sidebar.subheader("Plan Option")
    selected_plans = st.sidebar.multiselect("Select Plan", ["Growth", "IDCW", "Bonus"])
    
    # 5. Plan Type Filter (Regular vs Direct)
    # Defaulting to both/empty implies showing all. 
    selected_plan_types = st.sidebar.multiselect("Plan Type", ["Direct", "Regular"])
    
    # 6. Advanced Filters (Post-Processing)
    st.sidebar.subheader("Advanced Criteria")
    min_age = st.sidebar.number_input("Min Age (Years)", min_value=0, value=0, step=1, help="Filters applied during 'Compute' step")
    
    # 7. Custom Formula Filter
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔬 Custom Formula")
    
    # Formula logic remains handled by global imports
    
    # Premade formulas dropdown
    premade_formulas = get_premade_formulas()
    formula_names = ["Custom..."] + list(premade_formulas.keys())
    selected_formula_name = st.sidebar.selectbox(
        "Select Formula Template",
        formula_names,
        help="Choose a premade formula or create your own"
    )
    
    # Get formula text
    if selected_formula_name == "Custom...":
        default_formula = ""
    else:
        default_formula = premade_formulas[selected_formula_name]
    
    # Quick Reference - Collapsible metric categories
    with st.sidebar.expander("📋 Quick Metric Reference", expanded=False):
        st.markdown("**Returns:**")
        st.code("CAGR_1Y, CAGR_3Y, CAGR_5Y, CAGR_10Y\nRolling_Return_3Y, Alpha", language="python")
        
        st.markdown("**Risk Metrics:**")
        st.code("Volatility, Max_Drawdown, Tracking_Error\nCategory_StDev", language="python")
        
        st.markdown("**Risk-Adjusted:**")
        st.code("Sharpe_Ratio, Sortino_Ratio\nInformation_Ratio", language="python")
        
        st.markdown("**Other:**")
        st.code("Age, SEBI_Risk_Category", language="python")
        
        st.markdown("**Operators:**")
        st.code("> < >= <= == != and or in", language="python")
    
    # Formula input
    custom_formula = st.sidebar.text_area(
        "Formula Expression",
        value=default_formula,
        height=120,
        help="Type metric names and operators. Use the Quick Reference above for available metrics.",
        placeholder="Example: CAGR_5Y > 15 and Sharpe_Ratio > 1.5"
    )
    
    # Enable formula checkbox
    use_formula = st.sidebar.checkbox(
        "Apply Formula Filter",
        value=False,
        help="Enable to filter results using the formula above"
    )
    
    # Validate formula if enabled
    formula_error = None
    if use_formula and custom_formula:
        is_valid, error_msg = validate_formula(custom_formula)
        if not is_valid:
            formula_error = error_msg
            st.sidebar.error(f"❌ {error_msg}")
        else:
            st.sidebar.success("✅ Formula is valid")
    
    # Help expander
    with st.sidebar.expander("📖 Formula Help"):
        st.markdown(get_formula_help_text())
                
    # --- FILTERING LOGIC ---
    filtered_df = df_master.copy()
    
    # 1. Search Query (Literal match, not regex)
    if search_query:
        filtered_df = filtered_df[filtered_df['Scheme Name'].str.contains(search_query, case=False, na=False, regex=False)]
        
    # 2. Fund House Filter
    if selected_amcs:
        filtered_df = filtered_df[filtered_df['Fund House'].isin(selected_amcs)]
    
    # 3. Plan Type Filter (Regular vs Direct)
    if selected_plan_types:
        if "Direct" in selected_plan_types and "Regular" not in selected_plan_types:
            filtered_df = filtered_df[filtered_df['Scheme Name'].str.contains("Direct", case=False, na=False, regex=False)]
        elif "Regular" in selected_plan_types and "Direct" not in selected_plan_types:
            filtered_df = filtered_df[~filtered_df['Scheme Name'].str.contains("Direct", case=False, na=False, regex=False)]
            
    # 4. Plan Option Filter (Growth/IDCW/Bonus) - Using Mutual Exclusion for accuracy
    if selected_plans:
        # 1. Define negative patterns for things NOT selected
        excluded_patterns = []
        if "Growth" not in selected_plans: excluded_patterns.append("Growth")
        if "IDCW" not in selected_plans: 
            excluded_patterns.extend(["IDCW", "Dividend", "Income Distribution"])
        if "Bonus" not in selected_plans: excluded_patterns.append("Bonus")
        
        # 2. Hard block any fund containing an unselected pattern
        for pattern in excluded_patterns:
            filtered_df = filtered_df[~filtered_df['Scheme Name'].str.contains(pattern, case=False, na=False, regex=False)]
            
        # 3. Ensure the fund contains AT LEAST ONE of the selected patterns
        # (Special case: if Growth is selected, IDCW/Bonus are now blocked above, we just need to ensure the remaining funds are actually Growth)
        include_mask = pd.Series(False, index=filtered_df.index)
        include_patterns = []
        if "Growth" in selected_plans: include_patterns.append("Growth")
        if "IDCW" in selected_plans: include_patterns.extend(["IDCW", "Dividend"])
        if "Bonus" in selected_plans: include_patterns.append("Bonus")
        
        for p in include_patterns:
            include_mask |= filtered_df['Scheme Name'].str.contains(p, case=False, na=False, regex=False)
            
        filtered_df = filtered_df[include_mask]
            
    # 5. Global Exclusions (Debt) - Only applied if NO specific category is selected
    # This keeps the app clean by default but allows users to find Debt if they really want (via search/selection)
    if not selected_subcats and not search_query:
        # Pre-filter Debt and diversified FoF by default to keep screener "clean"
        filtered_df = filtered_df[~filtered_df['Category'].str.contains("Debt Scheme", case=False, na=False)]
        
        # Default FoF removal (keep Gold/Silver as those are popular commodities)
        fof_mask = filtered_df['Category'].str.contains("FoF Domestic", case=False, na=False)
        gs_rescue = filtered_df['Scheme Name'].str.contains("Gold|Silver", case=False, na=False)
        filtered_df = filtered_df[~fof_mask | gs_rescue]
        
    if selected_subcats:
        # Separate standard categories from special keyword-based ones
        special_cats = {"Silver ETF", "Gold ETF"}
        standard_selected = [s for s in selected_subcats if s not in special_cats]
        
        filtered_parts = []
        
        # 1. Standard Category Match
        if standard_selected:
            part_std = filtered_df[filtered_df['Category'].isin(standard_selected)]
            filtered_parts.append(part_std)
            
        # 2. Silver ETF Custom Match
        if "Silver ETF" in selected_subcats:
            silver_part = filtered_df[
                (filtered_df['Scheme Name'].str.contains('Silver', case=False, na=False, regex=False)) &
                (filtered_df['Category'].str.contains('ETF|FoF', case=False, na=False))
            ]
            filtered_parts.append(silver_part)
            
        # 3. Gold ETF Custom Match
        if "Gold ETF" in selected_subcats:
             gold_part = filtered_df[
                (filtered_df['Scheme Name'].str.contains('Gold', case=False, na=False, regex=False)) &
                (filtered_df['Category'].str.contains('ETF|FoF', case=False, na=False))
            ]
             filtered_parts.append(gold_part)
        
        if filtered_parts:
            # Combine all selected filters
            filtered_df = pd.concat(filtered_parts).drop_duplicates(subset=['Scheme Code'])
        else:
            # Should not happen if selected_subcats is not empty but good safeguard
            filtered_df = pd.DataFrame(columns=df_master.columns)
        
    # --- TABS ---
    tab_overview, tab_performance, tab_ai = st.tabs([
        "📊 Basic Overview", "⚡ Performance & Risk", "🤖 AI Smart Category Rank"
    ])
    
    with tab_overview:
        col1, col2 = st.columns(2)
        with col1:
            metric_card("Total Funds Scanned", f"{len(df_master):,}")
        with col2:
            metric_card("Funds Matching Criteria", f"{len(filtered_df):,}")
        
        st.markdown("---")
        
        # --- DATA GRID ---
        display_cols = ['Scheme Code', 'Scheme Name', 'Category', 'Net Asset Value', 'Date', 'Fund House']
        st.subheader("Filtered Results")
        st.dataframe(filtered_df[display_cols], use_container_width=True, hide_index=True)

    with tab_performance:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Analysis Tools")
        
        if not filtered_df.empty:
            count = len(filtered_df)
            if count > 50:
                st.warning(f"⚠️ You are about to analyze {count} funds. This involves fetching data for EACH fund and may take some time. Please be patient.")
            else:
                st.info(f"Ready to analyze {count} funds.")
            
            if st.sidebar.button("⚡ Compute Returns & Risk"):
                st.markdown("### 📈 Performance & Risk Analysis")
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Analyze ALL filtered candidates
                candidates = filtered_df.copy()
                results = []
                codes_to_fetch = [str(c) for c in candidates['Scheme Code']]
                hist_cache = {}
                
                status_msg = st.empty()
                status_msg.info(f"⚡ Pre-fetching history for {len(codes_to_fetch)} funds in parallel...")
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                    future_to_code = {executor.submit(fetch_fund_history, code): code for code in codes_to_fetch}
                    for future in concurrent.futures.as_completed(future_to_code):
                        code = future_to_code[future]
                        try:
                            hist_cache[code] = future.result()
                        except Exception as exc:
                            print(f'{code} generated an exception: {exc}')
                            hist_cache[code] = pd.DataFrame()
                
                status_msg.empty()
                
                # Caching for benchmarks and metrics
                benchmark_cache = {}
                benchmark_metrics_cache = {}
                
                # Iterate through Candidates
                for i, (idx, row) in enumerate(candidates.iterrows()):
                    code = str(row['Scheme Code'])
                    name = row['Scheme Name']
                    category = row['Category']
                    
                    # Update progress generic
                    progress_bar.progress((i + 1) / len(candidates))
                    status_text.text(f"Processing {i+1}/{len(candidates)}: {name[:30]}...")
                    
                    # 1. Use Pre-fetched History
                    hist_df = hist_cache.get(code, pd.DataFrame())
                    
                    if hist_df.empty:
                        continue # Can't analyze without data
    
                    # Normalize index to Timestamp objects (normalized) for robust alignment
                    hist_series = pd.Series(hist_df['nav'], index=pd.to_datetime(hist_df.index).normalize())
                    hist_series = hist_series[~hist_series.index.duplicated(keep='first')]
                    hist_series = hist_series.sort_index()
                    
                    # 2. Calculate Precise Age from History
                    # (Last Date - First Date)
                    start_date = hist_series.index.min()
                    end_date = hist_series.index.max()
                    obs_age = (end_date - start_date).days / 365.25 if start_date and end_date else 0
                    
                    # 3. Apply Age Filter
                    if min_age > 0:
                        if obs_age < min_age:
                            continue
                    
                    # 4. Compute Metrics
                    res_row = row.to_dict()
                    res_row['Age (Yrs)'] = obs_age
                    
                    # CAGR Returns
                    res_row['CAGR 1Y'] = calculate_cagr(hist_series, 1)
                    res_row['CAGR 3Y'] = calculate_cagr(hist_series, 3)
                    res_row['CAGR 5Y'] = calculate_cagr(hist_series, 5)
                    res_row['CAGR 10Y'] = calculate_cagr(hist_series, 10)
                    
                    # Rolling Return
                    res_row['3Y Avg Rolling Return'] = calculate_rolling_return(hist_series, 3)
                    
                    # 5. Benchmark Data - CACHED to avoid repeated calls
                    benchmark_name = get_benchmark_for_category(category, scheme_name=name)
                    benchmark_series = None
                    
                    if benchmark_name:
                        if benchmark_name in benchmark_cache:
                            benchmark_series = benchmark_cache[benchmark_name]
                        else:
                            try:
                                benchmark_series = download_benchmark(benchmark_name, period="max")
                                benchmark_cache[benchmark_name] = benchmark_series
                            except Exception as e:
                                print(f"Error fetching benchmark {benchmark_name}: {e}")
    
                    # Alpha (with explicit benchmark)
                    res_row['Alpha'] = calculate_alpha(hist_series, benchmark_series=benchmark_series, category=category)
                    
                    # Risk Metrics
                    volatility = calculate_volatility(hist_series)
                    res_row['Volatility'] = volatility
                    res_row['Category St Dev'] = volatility  # Same as volatility
                    res_row['Max Drawdown'] = calculate_max_drawdown(hist_series)
                    res_row['Sharpe Ratio'] = calculate_sharpe(hist_series)
                    res_row['Sortino Ratio'] = calculate_sortino(hist_series)
                    res_row['Information Ratio'] = calculate_information_ratio(hist_series, benchmark_series=benchmark_series, category=category)
                    res_row['Tracking Error'] = calculate_tracking_error(hist_series, benchmark_series=benchmark_series, category=category)
                    res_row['SEBI Risk Category'] = get_sebi_risk_category(volatility)
                    
                    # Rolling Returns (1Y, 3Y, 5Y, 10Y)
                    res_row['1Y Rolling Return'] = calculate_simple_rolling_return(hist_series, 1)
                    res_row['3Y Rolling Return'] = calculate_simple_rolling_return(hist_series, 3)
                    res_row['5Y Rolling Return'] = calculate_simple_rolling_return(hist_series, 5)
                    res_row['10Y Rolling Return'] = calculate_simple_rolling_return(hist_series, 10)
                    
                    # Upside/Downside Capture + Benchmark Metrics + Period-Specific Metrics
                    if benchmark_series is not None and not benchmark_series.empty:
                        # 6. Benchmark Metrics - CACHED to avoid repeated calculations
                        if benchmark_name in benchmark_metrics_cache:
                            bench_metrics = benchmark_metrics_cache[benchmark_name]
                        else:
                            bench_metrics = {
                                'cagr': {p: calculate_cagr(benchmark_series, p) for p in [1, 3, 5, 10]},
                                'rolling': {p: calculate_simple_rolling_return(benchmark_series, p) for p in [1, 3, 5, 10]}
                            }
                            benchmark_metrics_cache[benchmark_name] = bench_metrics
                        
                        # Apply cached benchmark metrics
                        for p in [1, 3, 5, 10]:
                            res_row[f'Benchmark CAGR {p}Y'] = bench_metrics['cagr'][p]
                            res_row[f'Benchmark {p}Y Rolling'] = bench_metrics['rolling'][p]

                        # Upside/Downside Capture (overall)
                        upside, downside = calculate_upside_downside_capture(
                            hist_series, benchmark_series
                        )
                        res_row['Upside Capture'] = upside
                        res_row['Downside Capture'] = downside
                        
                        # Period-Specific Metrics (1Y, 3Y, 5Y, 10Y)
                        for period in [1, 3, 5, 10]:
                            try:
                                # Use the pre-fetched benchmark series
                                metrics = calculate_period_metrics(hist_series, benchmark_series, period)
                                res_row[f'Alpha {period}Y'] = metrics['alpha']
                                res_row[f'Beta {period}Y'] = metrics['beta']
                                res_row[f'Sharpe {period}Y'] = metrics['sharpe']
                                res_row[f'Std Dev {period}Y'] = metrics['std_dev']
                                res_row[f'Upside Capture {period}Y'] = metrics['upside_capture']
                                res_row[f'Downside Capture {period}Y'] = metrics['downside_capture']
                            except Exception as inner_e:
                                print(f"Error calculating {period}Y metrics for {name}: {inner_e}")
                    results.append(res_row)
                
                progress_bar.empty()
                status_text.empty()
                
                if not results:
                     st.warning("No funds matched your Age criteria within the filtered list.")
                else:
                    # Create Result DataFrame
                    res_df = pd.DataFrame(results)
                    
                    # Summary Section - Show fund counts
                    st.markdown("---")
                    st.subheader("📊 Results Summary")
                    
                    initial_count = len(res_df)
                    
                    # Apply Custom Formula Filter (if enabled)
                    if use_formula and custom_formula and not formula_error:
                        # Show initial count
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Funds Analyzed", initial_count, help="Funds that passed Age and other filters")
                        
                        # Filter using formula
                        formula_passed = []
                        failed_reasons = []
                        
                        for idx, row in res_df.iterrows():
                            fund_data = prepare_fund_data(row)
                            passed, error = evaluate_formula(custom_formula, fund_data)
                            
                            if error:
                                failed_reasons.append(f"{row['Scheme Name']}: {error}")
                                continue
                            
                            if passed:
                                formula_passed.append(idx)
                            else:
                                # Track why it failed (for debugging)
                                if len(failed_reasons) < 3:  # Only show first 3 failures
                                    failed_reasons.append(f"{row['Scheme Name']}: Did not meet criteria")
                        
                        # Filter dataframe
                        res_df = res_df.loc[formula_passed]
                        filtered_count = len(res_df)
                        
                        # Show filtered count
                        with col2:
                            st.metric("Passed Formula", filtered_count, 
                                     delta=f"{filtered_count - initial_count}",
                                     delta_color="off",
                                     help="Funds that passed your custom formula")
                        
                        with col3:
                            pass_rate = (filtered_count / initial_count * 100) if initial_count > 0 else 0
                            st.metric("Pass Rate", f"{pass_rate:.1f}%", 
                                     help="Percentage of funds that passed the formula")
                        
                        if res_df.empty:
                            st.warning("⚠️ No funds matched your custom formula. Try adjusting the criteria.")
                            
                            # Show debug info
                            if failed_reasons:
                                with st.expander("🔍 Debug: Why funds failed", expanded=False):
                                    st.write("Sample of funds that didn't pass:")
                                    for reason in failed_reasons[:5]:
                                        st.text(f"• {reason}")
                                    
                                    # Show a sample fund's metrics
                                    if not res_df.empty or initial_count > 0:
                                        st.write("\n**Sample fund metrics (first fund analyzed):**")
                                        sample_idx = res_df.index[0] if not res_df.empty else 0
                                        sample_row = pd.DataFrame(results).iloc[0] if results else None
                                        if sample_row is not None:
                                            metrics_to_show = ['CAGR 5Y', 'CAGR 3Y', 'Sharpe Ratio', 'Max Drawdown', 
                                                              'Volatility', 'Alpha', '1Y Rolling Return', '3Y Rolling Return']
                                            for metric in metrics_to_show:
                                                if metric in sample_row:
                                                    value = sample_row[metric]
                                                    st.text(f"  {metric}: {value}")
                    else:
                        # No formula applied - just show total count
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("✅ Funds Found", initial_count, 
                                     help="Total funds that passed all filters and have sufficient data")
                        with col2:
                            if initial_count > 0:
                                st.success(f"🎯 Ready to analyze {initial_count} fund{'s' if initial_count != 1 else ''}")
                    
                    st.markdown("---")
                    
                    # Formatting for display
                    final_cols = [
                        'Scheme Name', 'Age (Yrs)',
                        'CAGR 1Y', 'CAGR 3Y', 'CAGR 5Y', 'CAGR 10Y',
                        'Benchmark CAGR 1Y', 'Benchmark CAGR 3Y', 'Benchmark CAGR 5Y', 'Benchmark CAGR 10Y',
                        '3Y Avg Rolling Return', 'Alpha',
                        '1Y Rolling Return', '3Y Rolling Return', '5Y Rolling Return', '10Y Rolling Return',
                        'Benchmark 1Y Rolling', 'Benchmark 3Y Rolling', 'Benchmark 5Y Rolling', 'Benchmark 10Y Rolling',
                        # Period-specific Alpha
                        'Alpha 1Y', 'Alpha 3Y', 'Alpha 5Y', 'Alpha 10Y',
                        # Period-specific Beta
                        'Beta 1Y', 'Beta 3Y', 'Beta 5Y', 'Beta 10Y',
                        # Period-specific Sharpe
                        'Sharpe 1Y', 'Sharpe 3Y', 'Sharpe 5Y', 'Sharpe 10Y',
                        # Period-specific Std Dev
                        'Std Dev 1Y', 'Std Dev 3Y', 'Std Dev 10Y',
                        # Period-specific Upside Capture
                        'Upside Capture 1Y', 'Upside Capture 3Y', 'Upside Capture 5Y', 'Upside Capture 10Y',
                        # Period-specific Downside Capture
                        'Downside Capture 1Y', 'Downside Capture 3Y', 'Downside Capture 5Y', 'Downside Capture 10Y',
                        # Overall metrics
                        'Upside Capture', 'Downside Capture',
                        'Volatility', 'Category St Dev', 'Max Drawdown', 'Tracking Error',
                        'Sharpe Ratio', 'Sortino Ratio', 'Information Ratio', 'SEBI Risk Category'
                    ]
                    
                    # Export functionality
                    if not res_df.empty:
                        from datetime import datetime
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        # Only include columns that actually exist in the result
                        valid_cols = [c for c in final_cols if c in res_df.columns]
                        csv_data = res_df[valid_cols].to_csv(index=False)
                        
                        col1, col2 = st.columns([3, 1])
                        with col2:
                            st.download_button(
                                label="📥 Export to Excel",
                                data=csv_data,
                                file_name=f"screener_results_{timestamp}.csv",
                                mime="text/csv",
                                help="Download results as CSV file (opens in Excel)"
                            )
                    
                    # Only display if we have results
                    if not res_df.empty:
                        # Helper to style
                        def style_returns(val):
                            if pd.isna(val): return ""
                            color = "#4ade80" if val > 0 else "#f87171"
                            return f"color: {color}"
                        
                        # Safe formatter to avoid NoneType errors
                        def safe_fmt(val, pattern):
                            if pd.isna(val) or val is None:
                                return "-"
                            try:
                                return pattern.format(val)
                            except:
                                return str(val)
    
                        # Filter final_cols to only those present in res_df to avoid KeyError
                        valid_final_cols = [c for c in final_cols if c in res_df.columns]
                        
                        # Filter formatting keys
                        format_dict = {
                                'Age (Yrs)': lambda x: safe_fmt(x, "{:.1f}"),
                                'CAGR 1Y': lambda x: safe_fmt(x, "{:.2f}%"),
                                'CAGR 3Y': lambda x: safe_fmt(x, "{:.2f}%"),
                                'CAGR 5Y': lambda x: safe_fmt(x, "{:.2f}%"),
                                'CAGR 10Y': lambda x: safe_fmt(x, "{:.2f}%"),
                                'Benchmark CAGR 1Y': lambda x: safe_fmt(x, "{:.2f}%"),
                                'Benchmark CAGR 3Y': lambda x: safe_fmt(x, "{:.2f}%"),
                                'Benchmark CAGR 5Y': lambda x: safe_fmt(x, "{:.2f}%"),
                                'Benchmark CAGR 10Y': lambda x: safe_fmt(x, "{:.2f}%"),
                                '3Y Avg Rolling Return': lambda x: safe_fmt(x, "{:.2f}%"),
                                'Alpha': lambda x: safe_fmt(x, "{:.2f}%"),
                                '1Y Rolling Return': lambda x: safe_fmt(x, "{:.2f}%"),
                                '3Y Rolling Return': lambda x: safe_fmt(x, "{:.2f}%"),
                                '5Y Rolling Return': lambda x: safe_fmt(x, "{:.2f}%"),
                                '10Y Rolling Return': lambda x: safe_fmt(x, "{:.2f}%"),
                                'Benchmark 1Y Rolling': lambda x: safe_fmt(x, "{:.2f}%"),
                                'Benchmark 3Y Rolling': lambda x: safe_fmt(x, "{:.2f}%"),
                                'Benchmark 5Y Rolling': lambda x: safe_fmt(x, "{:.2f}%"),
                                'Benchmark 10Y Rolling': lambda x: safe_fmt(x, "{:.2f}%"),
                                # Period-specific Alpha
                                'Alpha 1Y': lambda x: safe_fmt(x, "{:.2f}%"),
                                'Alpha 3Y': lambda x: safe_fmt(x, "{:.2f}%"),
                                'Alpha 5Y': lambda x: safe_fmt(x, "{:.2f}%"),
                                'Alpha 10Y': lambda x: safe_fmt(x, "{:.2f}%"),
                                # Period-specific Beta
                                'Beta 1Y': lambda x: safe_fmt(x, "{:.2f}"),
                                'Beta 3Y': lambda x: safe_fmt(x, "{:.2f}"),
                                'Beta 5Y': lambda x: safe_fmt(x, "{:.2f}"),
                                'Beta 10Y': lambda x: safe_fmt(x, "{:.2f}"),
                                # Period-specific Sharpe
                                'Sharpe 1Y': lambda x: safe_fmt(x, "{:.2f}"),
                                'Sharpe 3Y': lambda x: safe_fmt(x, "{:.2f}"),
                                'Sharpe 5Y': lambda x: safe_fmt(x, "{:.2f}"),
                                'Sharpe 10Y': lambda x: safe_fmt(x, "{:.2f}"),
                                # Period-specific Std Dev
                                'Std Dev 1Y': lambda x: safe_fmt(x, "{:.2f}%"),
                                'Std Dev 3Y': lambda x: safe_fmt(x, "{:.2f}%"),
                                'Std Dev 5Y': lambda x: safe_fmt(x, "{:.2f}%"),
                                'Std Dev 10Y': lambda x: safe_fmt(x, "{:.2f}%"),
                                # Period-specific Upside Capture
                                'Upside Capture 1Y': lambda x: safe_fmt(x, "{:.1f}%"),
                                'Upside Capture 3Y': lambda x: safe_fmt(x, "{:.1f}%"),
                                'Upside Capture 5Y': lambda x: safe_fmt(x, "{:.1f}%"),
                                'Upside Capture 10Y': lambda x: safe_fmt(x, "{:.1f}%"),
                                # Period-specific Downside Capture
                                'Downside Capture 1Y': lambda x: safe_fmt(x, "{:.1f}%"),
                                'Downside Capture 3Y': lambda x: safe_fmt(x, "{:.1f}%"),
                                'Downside Capture 5Y': lambda x: safe_fmt(x, "{:.1f}%"),
                                'Downside Capture 10Y': lambda x: safe_fmt(x, "{:.1f}%"),
                                # Overall metrics
                                'Upside Capture': lambda x: safe_fmt(x, "{:.1f}%"),
                                'Downside Capture': lambda x: safe_fmt(x, "{:.1f}%"),
                                'Volatility': lambda x: safe_fmt(x, "{:.2f}%"),
                                'Category St Dev': lambda x: safe_fmt(x, "{:.2f}%"),
                                'Max Drawdown': lambda x: safe_fmt(x, "{:.2f}%"),
                                'Tracking Error': lambda x: safe_fmt(x, "{:.2f}%"),
                                'Sharpe Ratio': lambda x: safe_fmt(x, "{:.2f}"),
                                'Sortino Ratio': lambda x: safe_fmt(x, "{:.2f}"),
                                'Information Ratio': lambda x: safe_fmt(x, "{:.2f}"),
                                'SEBI Risk Category': lambda x: x if x else "-"
                        }
                        
                        # Filter keys to only those present in res_df
                        valid_format_dict = {k: v for k, v in format_dict.items() if k in valid_final_cols}
                        
                        # Columns to apply conditional styling to
                        style_subset_cols = [
                                'CAGR 1Y', 'CAGR 3Y', 'CAGR 5Y', 'CAGR 10Y',
                                'Benchmark CAGR 1Y', 'Benchmark CAGR 3Y', 'Benchmark CAGR 5Y', 'Benchmark CAGR 10Y',
                                '3Y Avg Rolling Return', 'Alpha',
                                '1Y Rolling Return', '3Y Rolling Return', '5Y Rolling Return', '10Y Rolling Return',
                                'Benchmark 1Y Rolling', 'Benchmark 3Y Rolling', 'Benchmark 5Y Rolling', 'Benchmark 10Y Rolling',
                                'Alpha 1Y', 'Alpha 3Y', 'Alpha 5Y', 'Alpha 10Y'
                        ]
                        
                        # Filter for validity
                        valid_style_subset = [c for c in style_subset_cols if c in valid_final_cols]
    
                        st.dataframe(
                            res_df[valid_final_cols].style.format(valid_format_dict).applymap(
                                style_returns, 
                                subset=valid_style_subset
                            ),
                            use_container_width=True,
                            hide_index=True
                        )
        else:
            st.warning("No funds match your current filters.")

    with tab_ai:
        st.subheader("🤖 AI Smart Category Ranker")
        st.markdown("""
        Analyze the **currently filtered funds** to generate a uniquely sorted performance prediction 
        using Markov state transitions and peer ranking.
        """)
        
        if filtered_df.empty:
            st.info("👈 Please filter for some funds in the sidebar to run the AI Ranker.")
        else:
            count = len(filtered_df)
            st.info(f"Ready to rank {count} selected funds.")
            
            if st.button("🚀 Run AI Ranker", help="Analyzes the filtered selection and ranks them by AI Score"):
                # Freshly renamed function to avoid Streamlit signature caching issues
                ai_df = perform_ai_ranking(filtered_df, filtered_df['Scheme Code'].tolist())
                
                if not ai_df.empty:
                    st.success(f"Successfully ranked {len(ai_df)} funds")
                    
                    # --- SECTION 1: TOP 3 ARCHETYPE SUMMARY ---
                    st.markdown("### 🎯 AI Smart Summary (Top 3 per Focus)")
                    
                    # Compute categories
                    best_3 = ai_df.sort_values("PAR Score", ascending=False).head(3).copy(); best_3["Focus"] = "🏆 BEST OVERALL"
                    growth_3 = ai_df.sort_values("Growth Score", ascending=False).head(3).copy(); growth_3["Focus"] = "🚀 ALPHA CHASER"
                    safe_3 = ai_df.sort_values("Safety Score", ascending=False).head(3).copy(); safe_3["Focus"] = "🛡️ DEFENSIVE GEM"
                    
                    summary_df = pd.concat([best_3, growth_3, safe_3])[["Focus", "Scheme Name", "Growth Score", "Consistency", "Safety Score", "AI Verdict"]].reset_index(drop=True)
                    
                    def style_summary_table(val):
                        if "BEST" in str(val): return "background-color: rgba(16,185,129,0.1); color: #10b981; font-weight: 700;"
                        if "ALPHA" in str(val): return "background-color: rgba(245,158,11,0.1); color: #f59e0b;"
                        if "DEFENSIVE" in str(val): return "background-color: rgba(59,130,246,0.1); color: #3b82f6;"
                        return ""

                    st.dataframe(
                        summary_df.style.applymap(style_summary_table, subset=['Focus']),
                        column_config={
                            "Growth Score": st.column_config.ProgressColumn("Growth Skill", format="%.0f", min_value=0, max_value=100),
                            "Consistency": st.column_config.ProgressColumn("Consistency", format="%.0f", min_value=0, max_value=100),
                            "Safety Score": st.column_config.ProgressColumn("Safety & Defense", format="%.0f", min_value=0, max_value=100),
                            "Focus": st.column_config.TextColumn("🎯 Investment Focus")
                        },
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    st.write("") # Spacer
                    
                    # --- Section 2: Full Ranking Table ---
                    st.markdown("### 📋 Full Performance Ranking")
                    # Highlight colors
                    def color_verdict(val):
                        if "BEST" in val or "TOP" in val: return "background-color: rgba(16,185,129,0.2); color: #10b981; font-weight: 700;"
                        if "PERFECT" in val: return "background-color: rgba(59,130,246,0.1); color: #3b82f6; font-weight: 600;"
                        if "🚀" in val: return "color: #10b981;"
                        if "🔰" in val: return "color: #3b82f6;"
                        return "color: #8b949e;"
                        
                    st.dataframe(
                        ai_df.style.applymap(color_verdict, subset=['AI Verdict']),
                        column_config={
                            "Growth Score": st.column_config.ProgressColumn(
                                "Growth skill", help="Alpha-based score", format="%.0f", min_value=0, max_value=100
                            ),
                            "Consistency": st.column_config.ProgressColumn(
                                "Consistency", help="Markov-based stability", format="%.0f", min_value=0, max_value=100
                            ),
                            "Safety Score": st.column_config.ProgressColumn(
                                "Safety", help="Downside protection & risk-adjusted performance", format="%.0f", min_value=0, max_value=100
                            ),
                            "AI Badge": st.column_config.TextColumn("🏷️ Smart Badges")
                        },
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    st.caption("PAR Model (V2): Growth Skill (40%), Consistency (35%), Safety & Defense (25%). Scores are peer-relative.")
                    
                    # --- AI Export ---
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    csv_ai = ai_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Export AI Rankings",
                        data=csv_ai,
                        file_name=f"ai_rankings_{timestamp}.csv",
                        mime="text/csv",
                        help="Download AI ranking results as CSV"
                    )
                else:
                    st.error("No funds with sufficient 5Y history found in the selection for AI analysis.")

    # --- GLOBAL COMPLIANCE FOOTER ---
    render_compliance_footer()

if __name__ == "__main__":
    main()
