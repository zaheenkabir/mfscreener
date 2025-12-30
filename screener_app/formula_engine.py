"""
Formula Engine for Mutual Fund Screener
Provides safe evaluation of user-defined formulas for advanced filtering
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List

# Metric name mapping: Display Name -> Formula-safe Name
METRIC_MAPPING = {
    'Age (Yrs)': 'Age',
    'CAGR 1Y': 'CAGR_1Y',
    'CAGR 3Y': 'CAGR_3Y',
    'CAGR 5Y': 'CAGR_5Y',
    'CAGR 10Y': 'CAGR_10Y',
    'Benchmark CAGR 1Y': 'Benchmark_CAGR_1Y',
    'Benchmark CAGR 3Y': 'Benchmark_CAGR_3Y',
    'Benchmark CAGR 5Y': 'Benchmark_CAGR_5Y',
    'Benchmark CAGR 10Y': 'Benchmark_CAGR_10Y',
    '3Y Avg Rolling Return': 'Rolling_Return_3Y_Avg',
    'Alpha': 'Alpha',
    '1Y Rolling Return': 'Rolling_Return_1Y',
    '3Y Rolling Return': 'Rolling_Return_3Y',
    '5Y Rolling Return': 'Rolling_Return_5Y',
    '10Y Rolling Return': 'Rolling_Return_10Y',
    'Upside Capture': 'Upside_Capture',
    'Downside Capture': 'Downside_Capture',
    'Volatility': 'Volatility',
    'Category St Dev': 'Category_StDev',
    'Max Drawdown': 'Max_Drawdown',
    'Tracking Error': 'Tracking_Error',
    'Sharpe Ratio': 'Sharpe_Ratio',
    'Sortino Ratio': 'Sortino_Ratio',
    'Information Ratio': 'Information_Ratio',
    'SEBI Risk Category': 'SEBI_Risk_Category'
}

# Reverse mapping for display
REVERSE_METRIC_MAPPING = {v: k for k, v in METRIC_MAPPING.items()}

def get_premade_formulas() -> Dict[str, str]:
    """
    Returns dictionary of premade formula templates
    Based on market research of most popular screening criteria
    Optimized with achievable thresholds for Indian equity funds
    """
    return {
        # Most Popular: Consistent performers across time periods
        "Consistent Long-term Performers": "CAGR_3Y > 8 and CAGR_5Y > 9",
        
        # Quality funds with controlled risk
        "Quality with Low Volatility": "CAGR_5Y > 9 and Volatility < 25 and Max_Drawdown < 40",
        
        # Alpha generators - beating benchmark
        "Alpha Generators": "Alpha > 0.5 and CAGR_5Y > 10",
        
        # Defensive equity - lower risk, steady returns
        "Defensive Equity": "Max_Drawdown < 35 and Volatility < 22 and CAGR_3Y > 6",
        
        # High risk-adjusted returns
        "High Sharpe Ratio Funds": "Sharpe_Ratio > 1 and CAGR_5Y > 8",
        
        # Consistent benchmark beaters
        "Benchmark Beaters": "Alpha > 0.5 and CAGR_3Y > 8 and CAGR_5Y > 9",
        
        # Low drawdown with good returns
        "Low Drawdown High Returns": "Max_Drawdown < 35 and CAGR_5Y > 9",
        
        # Value + Momentum combination
        "Value with Momentum": "CAGR_1Y > 12 and CAGR_3Y > 10",
        
        # Top quartile across metrics
        "Top Quartile All-Round": "CAGR_5Y > 10 and Sharpe_Ratio > 0.8 and Max_Drawdown < 38",
        
        # Conservative growth
        "Conservative Growth": "CAGR_5Y > 8 and Max_Drawdown < 32 and Volatility < 22",
        
        # Recent strong performance
        "Recent Outperformers": "CAGR_1Y > 10 and CAGR_3Y > 12",
        
        # Low tracking error index-like
        "Index-like Low Cost": "Tracking_Error < 8 and CAGR_5Y > 8",
        
        # Rolling returns focused
        "Consistent Rolling Returns": "Rolling_Return_3Y > 9 and Rolling_Return_5Y > 10",
        
        # Capture ratio focused (only if data available)
        "Bull Market Winners": "Upside_Capture > 85 and CAGR_5Y > 9",
        
        # Downside protection (only if data available)
        "Bear Market Defenders": "Downside_Capture < 105 and Max_Drawdown < 32 and CAGR_5Y > 8"
    }

def prepare_fund_data(fund_row: pd.Series) -> Dict:
    """
    Convert fund row to formula-safe dictionary with mapped names
    Handles None/NaN values
    """
    safe_data = {}
    
    for display_name, formula_name in METRIC_MAPPING.items():
        if display_name in fund_row.index:
            value = fund_row[display_name]
            
            # Handle None/NaN
            if pd.isna(value) or value is None:
                safe_data[formula_name] = None
            else:
                safe_data[formula_name] = value
    
    return safe_data

def validate_formula(formula: str) -> Tuple[bool, str]:
    """
    Validate formula syntax and check for dangerous operations
    Returns (is_valid, error_message)
    """
    if not formula or not formula.strip():
        return False, "Formula cannot be empty"
    
    # Check for dangerous keywords
    dangerous_keywords = [
        'import', 'exec', 'eval', 'compile', 'open', 'file', 
        '__', 'lambda', 'def', 'class', 'yield', 'global'
    ]
    
    formula_lower = formula.lower()
    for keyword in dangerous_keywords:
        if keyword in formula_lower:
            return False, f"Forbidden keyword '{keyword}' in formula"
    
    # Try to compile the expression
    try:
        compile(formula, '<string>', 'eval')
    except SyntaxError as e:
        return False, f"Syntax error: {str(e)}"
    except Exception as e:
        return False, f"Invalid formula: {str(e)}"
    
    return True, ""

def evaluate_formula(formula: str, fund_data: Dict) -> Tuple[bool, str]:
    """
    Safely evaluate formula against fund data
    Returns (result, error_message)
    """
    # Validate first
    is_valid, error_msg = validate_formula(formula)
    if not is_valid:
        return False, error_msg
    
    # Prepare safe globals (only allow basic operations and comparisons)
    safe_globals = {
        '__builtins__': {
            'True': True,
            'False': False,
            'None': None,
            'abs': abs,
            'min': min,
            'max': max,
            'round': round,
            'len': len,
            'str': str,
            'int': int,
            'float': float,
        }
    }
    
    # Check if any metrics used in the formula have None values
    # Extract metric names from formula (simple approach)
    import re
    metric_pattern = r'\b(' + '|'.join(re.escape(m) for m in METRIC_MAPPING.values()) + r')\b'
    used_metrics = set(re.findall(metric_pattern, formula))
    
    # If any used metric is None, the fund fails the formula
    for metric in used_metrics:
        if metric in fund_data and (fund_data[metric] is None or pd.isna(fund_data[metric])):
            return False, ""  # Silently fail - fund doesn't have required data
    
    # Prepare locals with actual values (no None values at this point)
    safe_locals = {}
    for key, value in fund_data.items():
        if value is None or pd.isna(value):
            safe_locals[key] = None
        else:
            safe_locals[key] = value
    
    try:
        result = eval(formula, safe_globals, safe_locals)
        
        # Result must be boolean
        if not isinstance(result, bool):
            return False, f"Formula must return True/False, got {type(result).__name__}"
        
        return result, ""
        
    except NameError as e:
        return False, f"Unknown metric in formula: {str(e)}"
    except TypeError as e:
        # This happens when comparing None with numbers
        return False, ""  # Silently fail - missing data
    except Exception as e:
        return False, f"Evaluation error: {str(e)}"

def get_available_metrics() -> List[str]:
    """
    Returns list of available metric names for formulas
    """
    return sorted(list(METRIC_MAPPING.values()))

def get_formula_help_text() -> str:
    """
    Returns help text for formula syntax
    """
    return """
### 📊 Available Metrics

**Returns (%):**
- `CAGR_1Y`, `CAGR_3Y`, `CAGR_5Y`, `CAGR_10Y` - Compound annual growth rates
- `Benchmark_CAGR_1Y`, `Benchmark_CAGR_3Y`, `Benchmark_CAGR_5Y`, `Benchmark_CAGR_10Y` - Benchmark returns
- `Rolling_Return_3Y_Avg` - 3-year average rolling return (legacy)
- `Alpha` - Excess return vs benchmark

**Rolling Returns:**
- `Rolling_Return_1Y` - Average 1-year rolling return
- `Rolling_Return_3Y` - Average 3-year rolling return
- `Rolling_Return_5Y` - Average 5-year rolling return
- `Rolling_Return_10Y` - Average 10-year rolling return

**Capture Ratios:**
- `Upside_Capture` - Performance in bull markets (% of benchmark gains captured)
- `Downside_Capture` - Performance in bear markets (% of benchmark losses captured)

**Risk Metrics (%):**
- `Volatility` - Annualized standard deviation
- `Max_Drawdown` - Maximum peak-to-trough decline
- `Tracking_Error` - Deviation from benchmark
- `Category_StDev` - Category standard deviation

**Risk-Adjusted Ratios:**
- `Sharpe_Ratio` - Return per unit of total risk
- `Sortino_Ratio` - Return per unit of downside risk
- `Information_Ratio` - Excess return per unit of tracking error

**Other:**
- `Age` - Fund age in years
- `SEBI_Risk_Category` - Risk classification (text)

---

### 🔧 Operators

**Comparison:** `>`, `<`, `>=`, `<=`, `==`, `!=`  
**Logical:** `and`, `or`, `not`  
**Membership:** `in`, `not in`

---

### 💡 Example Formulas

**Simple:**
```
CAGR_5Y > 15
Sharpe_Ratio > 1.5
Max_Drawdown < 25
```

**Rolling Returns:**
```
Rolling_Return_3Y > 14
Rolling_Return_5Y > 15 and Rolling_Return_10Y > 12
```

**Capture Ratios:**
```
Upside_Capture > 100 and Downside_Capture < 80
```

**Combined:**
```
CAGR_5Y > 15 and Sharpe_Ratio > 1.5
Alpha > 2 or Information_Ratio > 1
Rolling_Return_5Y > 15 and Max_Drawdown < 25
```

**With Risk Category:**
```
SEBI_Risk_Category in ['Low', 'Moderately Low']
CAGR_5Y > 12 and SEBI_Risk_Category == 'Moderate'
```

**Complex:**
```
CAGR_5Y > 15 and Sharpe_Ratio > 1.5 and Max_Drawdown < 30 and Alpha > 2
(CAGR_3Y > 12 or CAGR_5Y > 14) and Volatility < 20
Upside_Capture > 95 and Downside_Capture < 85 and Rolling_Return_5Y > 14
```

---

### ✅ Realistic Thresholds

- **CAGR:** 12-18% (good for equity)
- **Sharpe:** > 1 (good), > 1.5 (excellent)
- **Alpha:** > 2% (good), > 3% (excellent)
- **Max Drawdown:** < 30% (acceptable), < 25% (good)
- **Volatility:** < 20% (moderate), < 15% (low)
- **Rolling Returns:** > 12% (good), > 15% (excellent)
- **Upside Capture:** > 100% (outperforming in bull markets)
- **Downside Capture:** < 100% (protecting in bear markets)
"""
