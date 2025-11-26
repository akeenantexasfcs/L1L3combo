"""
PRF Analysis Tool - Consolidated Application
Combines Layer 1 (Backtesting and Optimization) and Layer 3 (Systematic Positioning)
into a unified three-tab application.

Tabs:
1. Decision Support (Audit) - Single-grid what-if calculations
2. Portfolio Backtest (Audit) - Multi-grid historical backtests
3. Champion vs Challenger(s) - Strategy comparison with weather view enhancement
"""

import pandas as pd
import numpy as np
from itertools import combinations
from snowflake.snowpark.context import get_active_session
import time
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from scipy.optimize import minimize
from decimal import Decimal, ROUND_HALF_UP
import io
import re
import random

# =============================================================================
# === GLOBAL CONSTANTS ===
# =============================================================================

# The 11 valid PRF intervals
INTERVAL_ORDER_11 = ['Jan-Feb', 'Feb-Mar', 'Mar-Apr', 'Apr-May', 'May-Jun',
                     'Jun-Jul', 'Jul-Aug', 'Aug-Sep', 'Sep-Oct', 'Oct-Nov', 'Nov-Dec']

MONTH_TO_INTERVAL = {
    1: 'Jan-Feb', 2: 'Feb-Mar', 3: 'Mar-Apr', 4: 'Apr-May',
    5: 'May-Jun', 6: 'Jun-Jul', 7: 'Jul-Aug', 8: 'Aug-Sep',
    9: 'Sep-Oct', 10: 'Oct-Nov', 11: 'Nov-Dec'
}

INTERVAL_TO_MONTH_NUM = {name: month for month, name in MONTH_TO_INTERVAL.items()}
INTERVAL_INDICES = {name: i for i, name in enumerate(INTERVAL_ORDER_11)}

# Z-Score to Plain Language Mapping (from Layer 3)
HISTORICAL_CONTEXT_MAP = {
    'Dry': (-999, -0.25),
    'Normal': (-0.25, 0.25),
    'Wet': (0.25, 999)
}

TREND_MAP = {
    'Get Drier': (-999, -0.2),
    'Stay Stable': (-0.2, 0.2),
    'Get Wetter': (0.2, 999)
}

# Allocation constraints
MIN_ALLOCATION = 0.10  # 10% minimum per active interval
MAX_ALLOCATION = 0.50  # 50% maximum per active interval
ALLOCATION_INCREMENT = 0.01  # 1% increments

# Full coverage staggered patterns (from Layer 3)
PATTERN_A_INTERVALS = [0, 2, 4, 6, 8, 10]  # Jan-Feb, Mar-Apr, May-Jun, Jul-Aug, Sep-Oct, Nov-Dec
PATTERN_B_INTERVALS = [1, 3, 5, 7, 9]  # Feb-Mar, Apr-May, Jun-Jul, Aug-Sep, Oct-Nov

# Sorting metric maps
SORT_METRIC_DB_MAP = {
    'Portfolio Return': 'Cumulative_ROI',
    'Risk-Adjusted Return': 'Risk_Adjusted_Return',
    'Median ROI': 'Median_ROI',
    'Win Rate': 'Win_Rate'
}
SORT_METRIC_DISPLAY_MAP = {v: k for k, v in SORT_METRIC_DB_MAP.items()}

# King Ranch preset configuration
KING_RANCH_PRESET = {
    'grids': [9128, 9129, 8829, 9130, 7929, 8230, 8228, 8229],
    'counties': {
        'Kleberg': [9128, 9129, 8829, 9130],
        'Kenedy': [7929, 8230],
        'Brooks': [8228, 8229]
    },
    'acres': {
        9128: 56662,
        9129: 56662,
        8829: 56662,
        9130: 56662,
        7929: 86774,
        8230: 86774,
        8228: 26386,
        8229: 26386
    },
    'allocations': {
        9128: {'Feb-Mar': 20, 'Apr-May': 20, 'Jun-Jul': 20, 'Aug-Sep': 20, 'Oct-Nov': 20},
        9129: {'Jan-Feb': 17, 'Mar-Apr': 17, 'May-Jun': 17, 'Jul-Aug': 17, 'Sep-Oct': 16, 'Nov-Dec': 16},
        8829: {'Jan-Feb': 17, 'Mar-Apr': 17, 'May-Jun': 17, 'Jul-Aug': 16, 'Sep-Oct': 16, 'Nov-Dec': 17},
        9130: {'Feb-Mar': 20, 'Apr-May': 20, 'Jun-Jul': 20, 'Aug-Sep': 20, 'Oct-Nov': 20},
        7929: {'Jan-Feb': 17, 'Mar-Apr': 16, 'May-Jun': 16, 'Jul-Aug': 17, 'Sep-Oct': 17, 'Nov-Dec': 17},
        8230: {'Feb-Mar': 20, 'Apr-May': 20, 'Jun-Jul': 20, 'Aug-Sep': 20, 'Oct-Nov': 20},
        8228: {'Feb-Mar': 20, 'Apr-May': 20, 'Jun-Jul': 20, 'Aug-Sep': 20, 'Oct-Nov': 20},
        8229: {'Jan-Feb': 17, 'Mar-Apr': 17, 'May-Jun': 17, 'Jul-Aug': 16, 'Sep-Oct': 16, 'Nov-Dec': 17}
    }
}

st.set_page_config(layout="wide", page_title="PRF Analysis Tool")

# =============================================================================
# === ROUNDING AND PRECISION HELPERS ===
# =============================================================================

def round_half_up(value, decimals=2):
    """
    Round using 'round half up' to match PRF official tool.
    Handles floating-point precision issues by converting to Decimal early.
    Python's built-in round() uses banker's rounding (12.675 -> 12.67).
    PRF Tool uses round half up (12.675 -> 12.68).
    """
    if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
        return 0.0

    d = Decimal(str(value))

    if decimals == 0:
        quantize_to = Decimal('1')
    else:
        quantize_to = Decimal('0.' + '0' * (decimals - 1) + '1')

    return float(d.quantize(quantize_to, rounding=ROUND_HALF_UP))


def calculate_protection(county_base_value, coverage_level, productivity_factor, decimals=2):
    """
    Calculate dollar protection with proper precision and rounding.
    Uses Decimal arithmetic to avoid floating-point errors.
    """
    cbv = Decimal(str(county_base_value))
    cov = Decimal(str(coverage_level))
    prod = Decimal(str(productivity_factor))

    result = cbv * cov * prod

    if decimals == 0:
        quantize_to = Decimal('1')
    else:
        quantize_to = Decimal('0.' + '0' * (decimals - 1) + '1')

    return float(result.quantize(quantize_to, rounding=ROUND_HALF_UP))


# =============================================================================
# === GRID ID HELPER FUNCTIONS ===
# =============================================================================

def extract_numeric_grid_id(grid_id):
    """
    Extract numeric grid ID from formatted string.
    Examples:
        "9128 (Kleberg - TX)" -> 9128
        "8230 (Brooks - TX)" -> 8230
        9128 -> 9128 (handles plain integers)
    """
    if isinstance(grid_id, (int, float)):
        return int(grid_id)

    if isinstance(grid_id, str):
        match = re.match(r'(\d+)', grid_id)
        if match:
            return int(match.group(1))
    return None


def extract_county_from_grid_id(grid_id):
    """
    Extract county name from formatted grid ID.
    Examples:
        "9128 (Kleberg - TX)" -> "Kleberg"
        "8230 (Brooks - TX)" -> "Brooks"
    """
    if isinstance(grid_id, str) and '(' in grid_id:
        county_part = grid_id.split('(')[1].split('-')[0].strip()
        return county_part
    return None


def format_grid_display(grid_id_numeric, county_state=None):
    """Format grid ID for display"""
    if county_state:
        return f"{grid_id_numeric} ({county_state})"
    return str(grid_id_numeric)


# =============================================================================
# === INTERVAL ADJACENCY HELPERS ===
# =============================================================================

def is_adjacent(interval1, interval2):
    """Check if two intervals are adjacent (excluding Nov-Dec/Jan-Feb wrap)"""
    try:
        idx1 = INTERVAL_ORDER_11.index(interval1)
        idx2 = INTERVAL_ORDER_11.index(interval2)
    except ValueError:
        return False

    diff = abs(idx1 - idx2)
    return diff == 1


def has_adjacent_intervals(intervals_list):
    """
    Check if any two intervals in a list are adjacent.
    Allows Nov-Dec and Jan-Feb together (wrap-around exception).
    """
    if len(intervals_list) < 2:
        return False

    for i in range(len(intervals_list)):
        for j in range(i + 1, len(intervals_list)):
            interval1 = intervals_list[i]
            interval2 = intervals_list[j]

            # Allow Nov-Dec and Jan-Feb together (wrap-around exception)
            if (interval1 == 'Nov-Dec' and interval2 == 'Jan-Feb') or \
               (interval1 == 'Jan-Feb' and interval2 == 'Nov-Dec'):
                continue

            if is_adjacent(interval1, interval2):
                return True

    return False


# =============================================================================
# === ALLOCATION GENERATION ===
# =============================================================================

def generate_allocations(intervals_to_use, num_intervals):
    """
    Generate allocation percentages for N intervals, respecting all rules:
    - Only whole number percentages (1% increments)
    - Each interval: 0% OR 10-50%
    - Total must equal exactly 100%
    - Max 50% per interval

    Returns allocations as decimals (0.20 = 20%)
    """
    allocations = []

    if num_intervals == 1:
        # 1-interval: Can't reach 100% with one interval at 50% max
        return []

    elif num_intervals == 2:
        # 2-interval: Only 50/50 split is valid
        allocations.append({intervals_to_use[0]: 0.50, intervals_to_use[1]: 0.50})

    elif num_intervals == 3:
        splits = [
            (0.34, 0.33, 0.33),
            (0.50, 0.25, 0.25),
            (0.40, 0.30, 0.30),
            (0.45, 0.30, 0.25),
            (0.50, 0.30, 0.20),
            (0.40, 0.35, 0.25),
        ]
        for s in splits:
            allocations.append({intervals_to_use[i]: s[i] for i in range(3)})

    elif num_intervals == 4:
        splits = [
            (0.25, 0.25, 0.25, 0.25),
            (0.50, 0.20, 0.15, 0.15),
            (0.40, 0.20, 0.20, 0.20),
            (0.35, 0.25, 0.25, 0.15),
            (0.30, 0.30, 0.20, 0.20),
            (0.40, 0.25, 0.20, 0.15),
            (0.35, 0.30, 0.20, 0.15),
        ]
        for s in splits:
            allocations.append({intervals_to_use[i]: s[i] for i in range(4)})

    elif num_intervals == 5:
        splits = [
            (0.20, 0.20, 0.20, 0.20, 0.20),
            (0.50, 0.15, 0.15, 0.10, 0.10),
            (0.30, 0.20, 0.20, 0.15, 0.15),
            (0.40, 0.15, 0.15, 0.15, 0.15),
            (0.25, 0.25, 0.20, 0.15, 0.15),
            (0.35, 0.20, 0.15, 0.15, 0.15),
            (0.30, 0.25, 0.20, 0.15, 0.10),
        ]
        for s in splits:
            allocations.append({intervals_to_use[i]: s[i] for i in range(5)})

    elif num_intervals == 6:
        splits = [
            (0.17, 0.17, 0.17, 0.17, 0.16, 0.16),
            (0.50, 0.10, 0.10, 0.10, 0.10, 0.10),
            (0.30, 0.15, 0.15, 0.15, 0.15, 0.10),
            (0.40, 0.12, 0.12, 0.12, 0.12, 0.12),
            (0.25, 0.20, 0.15, 0.15, 0.15, 0.10),
            (0.35, 0.15, 0.15, 0.15, 0.10, 0.10),
            (0.20, 0.20, 0.15, 0.15, 0.15, 0.15),
        ]
        for s in splits:
            allocations.append({intervals_to_use[i]: s[i] for i in range(6)})

    return allocations


def is_valid_allocation(alloc_dict):
    """
    Check if allocation meets all rules:
    - Whole number percentages (1% increments)
    - Each interval: 0% OR 10-50%
    - Total equals 100%
    - Max 50% per interval
    """
    total = sum(alloc_dict.values())
    if abs(total - 1.0) > 0.001:
        return False

    for interval, pct in alloc_dict.items():
        if pct > 0.50:
            return False
        if pct > 0 and pct < 0.10:
            return False
        pct_as_percent = pct * 100
        if abs(pct_as_percent - round(pct_as_percent)) > 0.001:
            return False

    return True


# =============================================================================
# === CACHED DATA LOADING FUNCTIONS ===
# =============================================================================

@st.cache_data(ttl=3600)
def load_distinct_grids(_session):
    """Fetches the list of all available Grid IDs with county names from PRF_COUNTY_BASE_VALUES."""
    query = """
        SELECT DISTINCT GRID_ID
        FROM CAPITAL_MARKETS_SANDBOX.PUBLIC.PRF_COUNTY_BASE_VALUES
        ORDER BY GRID_ID
    """
    df = _session.sql(query).to_pandas()
    return df['GRID_ID'].tolist()


@st.cache_data(ttl=3600)
def load_county_base_value(_session, grid_id):
    """Fetches the average county base value for the grid using GRID_ID."""
    query = f"""
        SELECT AVG(COUNTY_BASE_VALUE)
        FROM CAPITAL_MARKETS_SANDBOX.PUBLIC.PRF_COUNTY_BASE_VALUES
        WHERE GRID_ID = '{grid_id}'
    """
    result = _session.sql(query).to_pandas()
    if result.empty or result.iloc[0, 0] is None:
        return 0.0
    return float(result.iloc[0, 0])


@st.cache_data(ttl=3600)
def get_current_rate_year(_session):
    """Finds the most recent year in the premium rates table."""
    return int(_session.sql("SELECT MAX(YEAR) FROM PRF_PREMIUM_RATES").to_pandas().iloc[0, 0])


@st.cache_data(ttl=3600)
def load_premium_rates(_session, grid_id, use, coverage_level, year):
    """Fetches premium rates for a coverage level (single coverage level version)."""
    numeric_grid_id = extract_numeric_grid_id(grid_id)
    cov_string = f"{coverage_level:.0%}"
    query = f"""
        SELECT INDEX_INTERVAL_NAME, PREMIUMRATE
        FROM PRF_PREMIUM_RATES
        WHERE GRID_ID = {numeric_grid_id}
          AND INTENDED_USE = '{use}'
          AND COVERAGE_LEVEL = '{cov_string}'
          AND YEAR = {year}
    """
    df = _session.sql(query).to_pandas()
    df['PREMIUMRATE'] = pd.to_numeric(df['PREMIUMRATE'], errors='coerce')
    return df.set_index('INDEX_INTERVAL_NAME')['PREMIUMRATE'].to_dict()


@st.cache_data(ttl=3600)
def load_premium_rates_multi(_session, grid_id, use, coverage_levels_list, year):
    """Fetches premium rates for multiple coverage levels."""
    numeric_grid_id = extract_numeric_grid_id(grid_id)

    all_premiums = {}
    for cov_level in coverage_levels_list:
        cov_string = f"{cov_level:.0%}"
        query = f"""
            SELECT INDEX_INTERVAL_NAME, PREMIUMRATE
            FROM PRF_PREMIUM_RATES
            WHERE GRID_ID = {numeric_grid_id}
              AND INTENDED_USE = '{use}'
              AND COVERAGE_LEVEL = '{cov_string}'
              AND YEAR = {year}
        """
        prem_df = _session.sql(query).to_pandas()
        prem_df['PREMIUMRATE'] = pd.to_numeric(prem_df['PREMIUMRATE'], errors='coerce')
        all_premiums[cov_level] = prem_df.set_index('INDEX_INTERVAL_NAME')['PREMIUMRATE'].to_dict()
    return all_premiums


@st.cache_data(ttl=3600)
def load_subsidy(_session, plan_code, coverage_level):
    """Fetches subsidy percentage for a single coverage level."""
    query = f"""
        SELECT SUBSIDY_PERCENT
        FROM SUBSIDYPERCENT_YTD_PLATINUM
        WHERE INSURANCE_PLAN_CODE = {plan_code}
          AND COVERAGE_LEVEL_PERCENT = {coverage_level}
        LIMIT 1
    """
    return float(_session.sql(query).to_pandas().iloc[0, 0])


@st.cache_data(ttl=3600)
def load_subsidies(_session, plan_code, coverage_levels_list):
    """Fetches subsidy percentages for multiple coverage levels."""
    all_subsidies = {}
    for cov_level in coverage_levels_list:
        query = f"""
            SELECT SUBSIDY_PERCENT
            FROM SUBSIDYPERCENT_YTD_PLATINUM
            WHERE INSURANCE_PLAN_CODE = {plan_code}
              AND COVERAGE_LEVEL_PERCENT = {cov_level}
            LIMIT 1
        """
        all_subsidies[cov_level] = float(_session.sql(query).to_pandas().iloc[0, 0])
    return all_subsidies


@st.cache_data(ttl=3600)
def load_all_indices(_session, grid_id):
    """Fetches all historical rainfall data for a single grid, including ENSO phase and Z-scores."""
    numeric_grid_id = extract_numeric_grid_id(grid_id)

    query = f"""
        SELECT
            YEAR, INTERVAL_NAME, INDEX_VALUE, INTERVAL_CODE,
            INTERVAL_MAPPING_TS_TEXT, INTERVAL_MAPPING_TS_NUMBER,
            OPTICAL_MAPPING_CPC, ONI_VALUE,
            SEQUENTIAL_Z_SCORE_HISTORICAL_RECORD,
            SEQUENTIAL_Z_SCORE_5P,
            SEQUENTIAL_Z_SCORE_11P
        FROM CAPITAL_MARKETS_SANDBOX.PUBLIC.RAIN_INDEX_PLATINUM_ENHANCED
        WHERE GRID_ID = {numeric_grid_id}
        ORDER BY YEAR, INTERVAL_CODE
    """
    df = _session.sql(query).to_pandas()
    df['INDEX_VALUE'] = pd.to_numeric(df['INDEX_VALUE'], errors='coerce')
    df['SEQUENTIAL_Z_SCORE_HISTORICAL_RECORD'] = pd.to_numeric(df['SEQUENTIAL_Z_SCORE_HISTORICAL_RECORD'], errors='coerce')
    df['SEQUENTIAL_Z_SCORE_5P'] = pd.to_numeric(df['SEQUENTIAL_Z_SCORE_5P'], errors='coerce')
    df['SEQUENTIAL_Z_SCORE_11P'] = pd.to_numeric(df['SEQUENTIAL_Z_SCORE_11P'], errors='coerce')
    # Filter out rows with no rainfall data
    df = df.dropna(subset=['INDEX_VALUE'])
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def filter_indices_by_scenario(all_indices_df, scenario, start_year=1948, end_year=2024):
    """
    Filter indices dataframe by scenario selection.
    """
    if scenario == 'All Years (except Current Year)':
        return all_indices_df[all_indices_df['YEAR'] < 2025]
    elif scenario == 'ENSO Phase: La Nina':
        if 'OPTICAL_MAPPING_CPC' in all_indices_df.columns:
            return all_indices_df[(all_indices_df['OPTICAL_MAPPING_CPC'] == 'La Nina') & (all_indices_df['YEAR'] < 2025)]
        return all_indices_df[all_indices_df['YEAR'] < 2025]
    elif scenario == 'ENSO Phase: El Nino':
        if 'OPTICAL_MAPPING_CPC' in all_indices_df.columns:
            return all_indices_df[(all_indices_df['OPTICAL_MAPPING_CPC'] == 'El Nino') & (all_indices_df['YEAR'] < 2025)]
        return all_indices_df[all_indices_df['YEAR'] < 2025]
    elif scenario == 'ENSO Phase: Neutral':
        if 'OPTICAL_MAPPING_CPC' in all_indices_df.columns:
            return all_indices_df[(all_indices_df['OPTICAL_MAPPING_CPC'] == 'Neutral') & (all_indices_df['YEAR'] < 2025)]
        return all_indices_df[all_indices_df['YEAR'] < 2025]
    else:  # Select my own interval / Custom Range
        return all_indices_df[(all_indices_df['YEAR'] >= start_year) & (all_indices_df['YEAR'] <= end_year)]


# =============================================================================
# === WEATHER VIEW FILTERING (from Layer 3) ===
# =============================================================================

def filter_years_by_market_view(df, regime, hist_context, trend):
    """
    Filter years (not intervals) based on market view conditions.
    Used for finding analog years matching a specific weather view.
    """
    hist_min, hist_max = HISTORICAL_CONTEXT_MAP[hist_context]
    trend_min, trend_max = TREND_MAP[trend]

    matching_years = []

    for year in df['YEAR'].unique():
        year_data = df[df['YEAR'] == year]

        if len(year_data) < 11:
            continue

        phase_counts = year_data['OPTICAL_MAPPING_CPC'].value_counts()
        dominant_phase = phase_counts.idxmax() if len(phase_counts) > 0 else None
        phase_intervals = phase_counts.max() if len(phase_counts) > 0 else 0

        if dominant_phase != regime or phase_intervals < 5:
            continue

        year_avg_hist_z = year_data['SEQUENTIAL_Z_SCORE_HISTORICAL_RECORD'].mean()

        if pd.isna(year_avg_hist_z) or year_avg_hist_z < hist_min or year_avg_hist_z >= hist_max:
            continue

        first_interval = year_data.iloc[0]
        last_interval = year_data.iloc[-1]

        if 'SEQUENTIAL_Z_SCORE_5P' not in last_interval.index or 'SEQUENTIAL_Z_SCORE_11P' not in first_interval.index:
            continue

        z_5p_end = last_interval['SEQUENTIAL_Z_SCORE_5P']
        z_11p_start = first_interval['SEQUENTIAL_Z_SCORE_11P']

        if pd.isna(z_5p_end) or pd.isna(z_11p_start):
            continue

        delta = z_5p_end - z_11p_start

        if delta < trend_min or delta >= trend_max:
            continue

        matching_years.append({
            'year': year,
            'dominant_phase': dominant_phase,
            'phase_intervals': phase_intervals,
            'year_avg_hist_z': year_avg_hist_z,
            'year_start_11p': z_11p_start,
            'year_end_5p': z_5p_end,
            'trajectory_delta': delta
        })

    return matching_years


@st.cache_data(ttl=3600, show_spinner=False)
def filter_years_by_market_view_cached(_session, grid_id_numeric, regime, hist_context, trend):
    """Cached version of filter_years_by_market_view."""
    df = load_all_indices(_session, grid_id_numeric)
    return filter_years_by_market_view(df, regime, hist_context, trend)


# =============================================================================
# === PORTFOLIO-AGGREGATED ANALOG YEARS ===
# =============================================================================

def find_portfolio_aggregated_analog_years(session, selected_grids, regime, hist_context, trend, cutoff_year=None):
    """
    Find analog years using portfolio-aggregated criteria.

    Instead of finding analog years per-grid independently:
    1. For each historical year (1948-2024):
       - Load Z-score data for ALL selected grids
       - Calculate portfolio-average of SEQUENTIAL_Z_SCORE_HISTORICAL_RECORD
       - Determine dominant ENSO phase across grids (mode)
       - Calculate portfolio-average trajectory (EOY 5P - SOY 11P)
    2. Filter years where portfolio-average matches market view criteria
    3. Output: Single list of analog years
    """
    hist_min, hist_max = HISTORICAL_CONTEXT_MAP[hist_context]
    trend_min, trend_max = TREND_MAP[trend]

    # Collect all grid data
    all_grid_data = {}
    for grid_id in selected_grids:
        grid_numeric = extract_numeric_grid_id(grid_id)
        df = load_all_indices(session, grid_numeric)
        if not df.empty:
            all_grid_data[grid_id] = df

    if not all_grid_data:
        return []

    # Get all unique years across all grids
    all_years = set()
    for df in all_grid_data.values():
        all_years.update(df['YEAR'].unique())

    # Filter to historical years only
    all_years = sorted([y for y in all_years if y < 2025])

    if cutoff_year:
        all_years = [y for y in all_years if y >= cutoff_year]

    matching_years = []

    for year in all_years:
        year_z_scores = []
        year_phases = []
        year_trajectories = []

        for grid_id, df in all_grid_data.items():
            year_data = df[df['YEAR'] == year]

            if len(year_data) < 11:
                continue

            # Get average historical Z-score for this grid-year
            avg_z = year_data['SEQUENTIAL_Z_SCORE_HISTORICAL_RECORD'].mean()
            if not pd.isna(avg_z):
                year_z_scores.append(avg_z)

            # Get dominant ENSO phase for this grid-year
            if 'OPTICAL_MAPPING_CPC' in year_data.columns:
                phase_counts = year_data['OPTICAL_MAPPING_CPC'].value_counts()
                if len(phase_counts) > 0:
                    year_phases.append(phase_counts.idxmax())

            # Calculate trajectory (EOY 5P - SOY 11P)
            first_interval = year_data.iloc[0]
            last_interval = year_data.iloc[-1]

            if 'SEQUENTIAL_Z_SCORE_5P' in last_interval.index and 'SEQUENTIAL_Z_SCORE_11P' in first_interval.index:
                z_5p_end = last_interval['SEQUENTIAL_Z_SCORE_5P']
                z_11p_start = first_interval['SEQUENTIAL_Z_SCORE_11P']

                if not pd.isna(z_5p_end) and not pd.isna(z_11p_start):
                    year_trajectories.append(z_5p_end - z_11p_start)

        # Skip year if insufficient data
        if not year_z_scores or not year_phases:
            continue

        # Calculate portfolio averages
        portfolio_avg_z = np.mean(year_z_scores)
        portfolio_avg_trajectory = np.mean(year_trajectories) if year_trajectories else 0

        # Determine dominant phase (mode across grids)
        from collections import Counter
        phase_counts = Counter(year_phases)
        dominant_phase = phase_counts.most_common(1)[0][0] if phase_counts else None
        phase_count = phase_counts.most_common(1)[0][1] if phase_counts else 0

        # Apply filters
        # 1. ENSO regime filter
        if dominant_phase != regime:
            continue

        # 2. Historical context filter
        if portfolio_avg_z < hist_min or portfolio_avg_z >= hist_max:
            continue

        # 3. Trend filter
        if portfolio_avg_trajectory < trend_min or portfolio_avg_trajectory >= trend_max:
            continue

        matching_years.append({
            'year': year,
            'dominant_phase': dominant_phase,
            'phase_count': phase_count,
            'portfolio_avg_z': portfolio_avg_z,
            'portfolio_trajectory': portfolio_avg_trajectory
        })

    return matching_years


# =============================================================================
# === STYLING HELPERS ===
# =============================================================================

def highlight_greater_than_zero(val):
    """Style helper for DataFrames"""
    if isinstance(val, (int, float)) and val > 0.001:
        return 'background-color: #DFF0D8'
    return ''


def create_download_button(fig, filename, key):
    """Create download button for matplotlib figure with high DPI"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    st.download_button(
        label="Download Chart",
        data=buf,
        file_name=filename,
        mime="image/png",
        key=key
    )


def render_allocation_inputs(key_prefix):
    """Creates the 11-row data editor for interval allocation."""
    st.subheader("Interval Allocation")

    # Check if there's preset allocation data for this key_prefix
    preset_key = f"{key_prefix}_preset_allocation"
    if preset_key in st.session_state:
        preset_alloc = st.session_state[preset_key]
        # Convert from decimal to percentage format - round to whole numbers
        alloc_data = {interval: round(preset_alloc.get(interval, 0.0) * 100) for interval in INTERVAL_ORDER_11}
    else:
        # Default allocation
        alloc_data = {
            'Jan-Feb': 50, 'Feb-Mar': 0, 'Mar-Apr': 50, 'Apr-May': 0,
            'May-Jun': 0, 'Jun-Jul': 0, 'Jul-Aug': 0, 'Aug-Sep': 0,
            'Sep-Oct': 0, 'Oct-Nov': 0, 'Nov-Dec': 0
        }

    df_alloc = pd.DataFrame(list(alloc_data.items()), columns=['Interval', 'Percent of Value'])

    st.caption("Whole numbers only (1% increments). Each interval: 0% OR 10-50%. Total must equal 100%. No adjacent intervals (except Nov-Dec/Jan-Feb wrap).")

    edited_df = st.data_editor(
        df_alloc,
        key=f"{key_prefix}_alloc_editor",
        num_rows="fixed",
        use_container_width=True,
        column_config={
            "Interval": st.column_config.TextColumn("Interval", disabled=True, width="medium"),
            "Percent of Value": st.column_config.NumberColumn("Percent of Value (%)", min_value=0, max_value=50, step=1, format="%d%%")
        }
    )

    # --- Validation ---
    alloc_dict = pd.Series(edited_df['Percent of Value'].values, index=edited_df['Interval']).to_dict()

    # Round to integers to ensure whole numbers
    alloc_dict = {k: round(v) for k, v in alloc_dict.items()}

    total_pct = sum(alloc_dict.values())
    max_pct = max(alloc_dict.values())

    is_valid = True

    # Check for whole numbers
    for interval, pct in alloc_dict.items():
        if pct != int(pct):
            st.error(f"All allocations must be whole numbers. {interval} has {pct}%")
            is_valid = False
            break

    # Check for 0% or 10-50% range
    for interval, pct in alloc_dict.items():
        if pct > 0 and pct < 10:
            st.error(f"Each interval must be 0% OR between 10-50%. {interval} has {pct}% (below 10% minimum)")
            is_valid = False
            break

    if abs(total_pct - 100) > 0.01:
        st.error(f"Allocation must total 100%. Current total: {total_pct:.0f}%")
        is_valid = False

    if max_pct > 50:
        st.error(f"No interval can exceed 50%.")
        is_valid = False

    for i in range(len(INTERVAL_ORDER_11) - 1):  # Stops before Nov-Dec
        if alloc_dict[INTERVAL_ORDER_11[i]] > 0 and alloc_dict[INTERVAL_ORDER_11[i+1]] > 0:
            st.error(f"Cannot allocate to adjacent intervals: {INTERVAL_ORDER_11[i]} and {INTERVAL_ORDER_11[i+1]}")
            is_valid = False
            break

    if is_valid:
        st.success(f"Valid. Total: {total_pct:.0f}%")

    alloc_dict_float = {k: v / 100.0 for k, v in alloc_dict.items()}

    return alloc_dict_float, is_valid


# =============================================================================
# === TAB 1: DECISION SUPPORT (AUDIT) ===
# =============================================================================

def render_decision_support_tab(session, grid_id, intended_use, productivity_factor, total_insured_acres, plan_code):
    """
    Decision Support (Audit) tab - renamed from Layer 1's "Decision Support"
    Single-grid what-if calculations for a specific historical year.
    """
    st.subheader("Decision Support (Audit)")
    st.caption("Calculate ROI for a single grid using a specific historical rainfall year.")

    col1, col2 = st.columns(2)
    sample_year = col1.selectbox("Historical Rainfall Year", list(range(1948, 2026)), index=77, key="ds_year")
    coverage_level = col2.selectbox("Coverage Level", [0.70, 0.75, 0.80, 0.85, 0.90], index=2, format_func=lambda x: f"{x:.0%}", key="ds_coverage")

    with st.expander("Step 2: Set Interval Allocations", expanded=True):
        pct_of_value_alloc, is_valid = render_allocation_inputs("ds")

    st.divider()

    if 'ds_run' not in st.session_state:
        st.session_state.ds_run = False

    if st.button("Run Calculation", key="ds_run_button", disabled=not is_valid):
        st.session_state.ds_run = True
        try:
            with st.spinner("Calculating..."):
                # --- 1. FETCH DATA ---
                subsidy_percent = load_subsidy(session, plan_code, coverage_level)
                county_base_value = load_county_base_value(session, grid_id)
                current_rate_year = get_current_rate_year(session)
                premium_rates_df = load_premium_rates(session, grid_id, intended_use, coverage_level, current_rate_year)

                # Extract numeric grid ID for rainfall data query
                numeric_grid_id = extract_numeric_grid_id(grid_id)
                actuals_query = f"""
                    SELECT INTERVAL_NAME, INDEX_VALUE
                    FROM CAPITAL_MARKETS_SANDBOX.PUBLIC.RAIN_INDEX_PLATINUM_ENHANCED
                    WHERE GRID_ID = {numeric_grid_id} AND YEAR = {sample_year}
                """
                actuals_df = session.sql(actuals_query).to_pandas().set_index('INTERVAL_NAME')
                actuals_df['INDEX_VALUE'] = pd.to_numeric(actuals_df['INDEX_VALUE'], errors='coerce')

                dollar_amount_of_protection = calculate_protection(county_base_value, coverage_level, productivity_factor)
                total_policy_protection = dollar_amount_of_protection * total_insured_acres

                roi_df = pd.DataFrame(index=INTERVAL_ORDER_11)
                roi_df['Percent of Value'] = roi_df.index.map(pct_of_value_alloc)
                roi_df['Policy Protection Per Unit'] = (total_policy_protection * roi_df['Percent of Value']).apply(lambda x: round_half_up(x, 0))
                roi_df = roi_df.join(pd.Series(premium_rates_df, name='PREMIUM_RATE'))
                roi_df = roi_df.join(actuals_df.rename(columns={'INDEX_VALUE': 'Actual Index Value'}))
                roi_df['PREMIUM_RATE'] = pd.to_numeric(roi_df['PREMIUM_RATE'], errors='coerce').fillna(0)
                roi_df['Actual Index Value'] = pd.to_numeric(roi_df['Actual Index Value'], errors='coerce')
                roi_df['Premium Rate Per $100'] = roi_df['PREMIUM_RATE'] * 100
                roi_df['Total Premium'] = (roi_df['Policy Protection Per Unit'] * roi_df['PREMIUM_RATE']).apply(lambda x: round_half_up(x, 0))
                roi_df['Premium Subsidy'] = (roi_df['Total Premium'] * subsidy_percent).apply(lambda x: round_half_up(x, 0))
                roi_df['Producer Premium'] = roi_df['Total Premium'] - roi_df['Premium Subsidy']
                trigger_level = coverage_level * 100
                shortfall_pct = (trigger_level - roi_df['Actual Index Value']) / trigger_level
                roi_df['Estimated Indemnity'] = (shortfall_pct * roi_df['Policy Protection Per Unit']).clip(lower=0).apply(lambda x: round_half_up(x, 0) if abs(x) >= 0.01 else 0.0)
                roi_df['ROI %'] = np.where(roi_df['Producer Premium'] > 0, (roi_df['Estimated Indemnity'] - roi_df['Producer Premium']) / roi_df['Producer Premium'], 0)

            # Save results to session state
            st.session_state.ds_results = {
                "roi_df": roi_df, "grid_id": grid_id, "sample_year": sample_year, "current_rate_year": current_rate_year,
                "intended_use": intended_use, "coverage_level": coverage_level, "productivity_factor": productivity_factor,
                "total_insured_acres": total_insured_acres, "county_base_value": county_base_value,
                "dollar_amount_of_protection": dollar_amount_of_protection, "total_policy_protection": total_policy_protection,
                "subsidy_percent": subsidy_percent
            }

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.exception(e)
            st.session_state.ds_results = None

    # Display results if available
    if 'ds_results' in st.session_state and st.session_state.ds_results:
        try:
            r = st.session_state.ds_results
            st.header(f"ROI Calculation - Grid {r['grid_id']}, Year {r['sample_year']}")
            st.caption(f"Coverage: {r['coverage_level']:.0%} | Productivity: {r['productivity_factor']:.0%} | Use: {r['intended_use']} | Acres: {r['total_insured_acres']:,}")

            c1, c2 = st.columns(2)
            with c1.container(border=True):
                st.subheader("Protection")
                st.text(f"Use: {r['intended_use']}")
                st.text(f"Coverage: {r['coverage_level']:.0%}")
                st.text(f"Productivity: {r['productivity_factor']:.0%}")
                st.text(f"Acres: {r['total_insured_acres']:,}")

            with c2.container(border=True):
                st.subheader("Policy")
                st.text(f"Base Value: ${r['county_base_value']:,.2f}")
                st.text(f"Protection: ${r['dollar_amount_of_protection']:,.2f}")
                st.text(f"Total Protection: ${r['total_policy_protection']:,.0f}")
                st.text(f"Subsidy: {r['subsidy_percent']:.1%}")

            st.subheader("Protection Table")

            # Add CSV download button
            csv_df = r['roi_df'].copy()
            csv_df['Percent of Value'] = csv_df['Percent of Value'] * 100
            csv_columns = ['Percent of Value', 'Policy Protection Per Unit', 'Premium Rate Per $100',
                           'Total Premium', 'Premium Subsidy', 'Producer Premium',
                           'Actual Index Value', 'Estimated Indemnity', 'ROI %']
            csv_export = csv_df[csv_columns].to_csv()

            st.download_button(
                label="Export CSV",
                data=csv_export,
                file_name=f"protection_grid_{extract_numeric_grid_id(r['grid_id'])}_year_{r['sample_year']}.csv",
                mime="text/csv",
            )

            display_df = pd.DataFrame(index=r['roi_df'].index)
            display_df['% Value'] = r['roi_df']['Percent of Value'].apply(lambda x: f"{x*100:.0f}" if x > 0 else 'N/A')
            display_df['Protection'] = r['roi_df']['Policy Protection Per Unit'].apply(lambda x: f"${x:,.0f}" if x > 0 else 'N/A')
            display_df['Rate/$100'] = r['roi_df']['Premium Rate Per $100'].apply(lambda x: f"{x:.2f}" if x > 0 else 'N/A')
            display_df['Premium'] = r['roi_df']['Total Premium'].apply(lambda x: f"${x:,.0f}" if x > 0 else 'N/A')
            display_df['Subsidy'] = r['roi_df']['Premium Subsidy'].apply(lambda x: f"${x:,.0f}" if x > 0 else 'N/A')
            display_df['Producer'] = r['roi_df']['Producer Premium'].apply(lambda x: f"${x:,.0f}" if x > 0 else 'N/A')
            display_df['Index'] = r['roi_df'].apply(
                lambda row: 'N/A' if pd.isna(row['Actual Index Value'])
                else (f"{row['Actual Index Value']:.1f}" if row['Percent of Value'] > 0 or row['Actual Index Value'] > 0
                else 'N/A'),
                axis=1
            )
            display_df['Indemnity'] = r['roi_df']['Estimated Indemnity'].apply(
                lambda x: f"${x:,.0f}" if pd.notna(x) and x > 0 else 'N/A'
            )
            display_df['ROI %'] = r['roi_df'].apply(
                lambda row: 'N/A' if pd.isna(row['Estimated Indemnity']) or row['ROI %'] == 0
                else f"{row['ROI %']:.2%}",
                axis=1
            )
            st.dataframe(display_df, use_container_width=True)

            # Totals
            total_producer_prem = r['roi_df']['Producer Premium'].apply(lambda x: round_half_up(x, 0) if pd.notna(x) else 0).sum()
            total_indemnity = r['roi_df']['Estimated Indemnity'].apply(lambda x: round_half_up(x, 0) if pd.notna(x) else 0).sum(skipna=True)
            net_return = total_indemnity - total_producer_prem

            st.subheader("Totals")

            has_missing_data = r['roi_df']['Actual Index Value'].isna().any()
            if has_missing_data:
                st.info("Some intervals have incomplete data")

            c1, c2, c3 = st.columns(3)
            c1.metric("Producer Premium", f"${total_producer_prem:,.0f}")
            c2.metric("Total Indemnity", f"${total_indemnity:,.0f}")
            c3.metric("Net Return", f"${net_return:,.0f}")

            if total_producer_prem > 0:
                st.metric("ROI", f"{net_return / total_producer_prem:.2%}")
        except Exception as e:
            st.error(f"Error displaying results: {e}")
            st.session_state.ds_results = None
    elif not st.session_state.ds_run:
        st.info("Select parameters and click 'Run Calculation'")


# =============================================================================
# === TAB 2: PORTFOLIO BACKTEST (AUDIT) ===
# =============================================================================

def render_portfolio_backtest_tab(session, grid_id, intended_use, productivity_factor, total_insured_acres, plan_code):
    """
    Portfolio Backtest (Audit) tab - renamed from Layer 1's "Portfolio Backtest"
    Multi-grid historical backtests.
    """
    st.subheader("Portfolio Backtest (Audit)")
    st.caption("Run historical backtests for a portfolio of grids.")

    # === PRESET LOADING BUTTON ===
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("Load King Ranch", key="pb_load_kr"):
            try:
                all_grids_for_preset = load_distinct_grids(session)

                # Build mapping of numeric IDs to their proper county names from preset
                target_grid_mapping = {}
                for county, grid_ids in KING_RANCH_PRESET['counties'].items():
                    for grid_id_num in grid_ids:
                        target_grid_mapping[grid_id_num] = f"{grid_id_num} ({county} - TX)"

                # Match grids in the order from preset
                preset_grid_ids = []
                for numeric_id in KING_RANCH_PRESET['grids']:
                    target_str = target_grid_mapping.get(numeric_id, "")
                    # Try exact match first
                    if target_str in all_grids_for_preset:
                        preset_grid_ids.append(target_str)
                    else:
                        # Fallback: find any grid with this numeric ID
                        for grid_option in all_grids_for_preset:
                            if extract_numeric_grid_id(grid_option) == numeric_id:
                                preset_grid_ids.append(grid_option)
                                break

                st.session_state.pb_grids = preset_grid_ids
                st.session_state.pb_use_custom_acres = True

                # Set acres and allocations for each grid
                for gid in preset_grid_ids:
                    numeric_id = extract_numeric_grid_id(gid)
                    st.session_state[f"pb_acres_{gid}"] = KING_RANCH_PRESET['acres'].get(numeric_id, total_insured_acres)
                    alloc = KING_RANCH_PRESET['allocations'].get(numeric_id, {})
                    alloc_decimal = {interval: float(alloc.get(interval, 0.0)) / 100.0 for interval in INTERVAL_ORDER_11}
                    st.session_state[f"pb_grid_{gid}_preset_allocation"] = alloc_decimal

                # Set King Ranch specific parameters
                st.session_state.productivity_factor = 1.35
                st.session_state.pb_coverage = 0.75

                st.success("King Ranch loaded! (8 grids, 135% productivity, 75% coverage)")

            except Exception as e:
                st.error(f"Error loading King Ranch: {e}")

    with col2:
        st.caption("Auto-populate King Ranch strategy")

    st.divider()

    try:
        all_grids = load_distinct_grids(session)
    except:
        all_grids = [grid_id]

    default_grids = st.session_state.get('pb_grids', [grid_id])
    default_grids = [g for g in default_grids if g in all_grids]

    selected_grids = st.multiselect(
        "Select Grids",
        options=all_grids,
        default=default_grids,
        max_selections=20,
        key="pb_grids"
    )

    if not selected_grids:
        st.warning("Select at least one grid")
        return

    st.divider()

    # === SCENARIO DEFINITION SECTION ===
    st.markdown("#### Scenario Definition")

    scenario_options = [
        'All Years (except Current Year)',
        'ENSO Phase: La Nina',
        'ENSO Phase: El Nino',
        'ENSO Phase: Neutral',
        'Select my own interval'
    ]

    selected_scenario = st.radio(
        "Select one scenario to backtest:",
        options=scenario_options,
        index=0,
        key='pb_scenario_select'
    )

    # Conditional year range display
    start_year = 1948
    end_year = 2024
    if selected_scenario == 'Select my own interval':
        col1, col2 = st.columns(2)
        start_year = col1.selectbox("Start Year", list(range(1948, 2026)), index=62, key="pb_start")
        end_year = col2.selectbox("End Year", list(range(1948, 2026)), index=76, key="pb_end")

    st.divider()

    # === PARAMETERS SECTION ===
    st.subheader("Parameters")
    coverage_level = st.selectbox(
        "Coverage Level",
        [0.70, 0.75, 0.80, 0.85, 0.90],
        index=2,
        format_func=lambda x: f"{x:.0%}",
        key="pb_coverage"
    )

    st.divider()

    # === ACRE CONFIGURATION SECTION ===
    st.subheader("Acre Configuration")

    use_custom_acres = st.checkbox(
        "Configure acres per grid",
        value=st.session_state.get('pb_use_custom_acres', False),
        key="pb_use_custom_acres"
    )

    grid_acres = {}
    if use_custom_acres:
        cols = st.columns(min(4, len(selected_grids)))
        for idx, gid in enumerate(selected_grids):
            with cols[idx % 4]:
                numeric_id = extract_numeric_grid_id(gid)
                default_acres = st.session_state.get(f"pb_acres_{gid}", KING_RANCH_PRESET['acres'].get(numeric_id, total_insured_acres))
                grid_acres[gid] = st.number_input(
                    f"{gid}",
                    min_value=1,
                    value=default_acres,
                    step=10,
                    key=f"pb_acres_{gid}"
                )
    else:
        acres_per_grid = total_insured_acres // len(selected_grids)
        st.info(f"Using {total_insured_acres:,} acres equally distributed ({acres_per_grid:,} acres per grid)")
        for gid in selected_grids:
            grid_acres[gid] = acres_per_grid

    st.divider()

    # === ALLOCATIONS SECTION ===
    st.subheader(f"Allocations for {len(selected_grids)} Grid(s)")

    grid_allocations = {}
    all_valid = True

    for gid in selected_grids:
        with st.expander(f"{gid} ({grid_acres[gid]:,} acres)", expanded=len(selected_grids) == 1):
            alloc_dict, is_valid = render_allocation_inputs(f"pb_grid_{gid}")
            grid_allocations[gid] = alloc_dict
            if not is_valid:
                all_valid = False

    st.divider()

    # === RUN BACKTEST SECTION ===
    if 'pb_run' not in st.session_state:
        st.session_state.pb_run = False

    if st.button("Run Portfolio Backtest", key="pb_run_button", disabled=not all_valid):
        st.session_state.pb_run = True
        try:
            grid_results = {}
            years_used = []

            with st.spinner(f"Running backtest for {len(selected_grids)} grids..."):
                for gid in selected_grids:
                    try:
                        subsidy_percent = load_subsidy(session, plan_code, coverage_level)
                        county_base_value = load_county_base_value(session, gid)
                        current_rate_year = get_current_rate_year(session)
                        premium_rates_df = load_premium_rates(session, gid, intended_use, coverage_level, current_rate_year)
                        dollar_amount_of_protection = calculate_protection(county_base_value, coverage_level, productivity_factor)
                        total_policy_protection = dollar_amount_of_protection * grid_acres[gid]
                        all_indices_df = load_all_indices(session, gid)

                        # Apply scenario-based year filtering
                        filtered_df = filter_indices_by_scenario(all_indices_df, selected_scenario, start_year, end_year)

                        grid_years = filtered_df['YEAR'].unique()

                        year_results = []
                        for year in sorted(grid_years):
                            actuals_df = filtered_df[filtered_df['YEAR'] == year].set_index('INTERVAL_NAME')
                            if actuals_df.empty:
                                continue

                            roi_df = pd.DataFrame(index=INTERVAL_ORDER_11)
                            roi_df['Percent of Value'] = roi_df.index.map(grid_allocations[gid])
                            roi_df['Policy Protection Per Unit'] = (total_policy_protection * roi_df['Percent of Value']).apply(lambda x: round_half_up(x, 0))
                            roi_df = roi_df.join(pd.Series(premium_rates_df, name='PREMIUM_RATE'))
                            roi_df = roi_df.join(actuals_df.rename(columns={'INDEX_VALUE': 'Actual Index Value'}))
                            roi_df['PREMIUM_RATE'] = pd.to_numeric(roi_df['PREMIUM_RATE'], errors='coerce').fillna(0)
                            roi_df['Actual Index Value'] = pd.to_numeric(roi_df['Actual Index Value'], errors='coerce').fillna(0)
                            roi_df['Total Premium'] = (roi_df['Policy Protection Per Unit'] * roi_df['PREMIUM_RATE']).apply(lambda x: round_half_up(x, 0))
                            roi_df['Premium Subsidy'] = (roi_df['Total Premium'] * subsidy_percent).apply(lambda x: round_half_up(x, 0))
                            roi_df['Producer Premium'] = roi_df['Total Premium'] - roi_df['Premium Subsidy']
                            trigger_level = coverage_level * 100
                            shortfall_pct = (trigger_level - roi_df['Actual Index Value']) / trigger_level
                            roi_df['Estimated Indemnity'] = (shortfall_pct * roi_df['Policy Protection Per Unit']).clip(lower=0).apply(lambda x: round_half_up(x, 0) if abs(x) >= 0.01 else 0.0)
                            total_indemnity = roi_df['Estimated Indemnity'].sum()
                            total_producer_prem = roi_df['Producer Premium'].sum()
                            year_roi = (total_indemnity - total_producer_prem) / total_producer_prem if total_producer_prem > 0 else 0.0

                            year_results.append({
                                'Year': year, 'Total Indemnity': total_indemnity, 'Producer Premium': total_producer_prem,
                                'Net Return': total_indemnity - total_producer_prem, 'Total ROI': year_roi
                            })
                            if year not in years_used:
                                years_used.append(year)

                        results_df = pd.DataFrame(year_results)
                        grid_results[gid] = {
                            'results_df': results_df,
                            'allocation': grid_allocations[gid]
                        }

                    except Exception as e:
                        st.error(f"Grid {gid}: {str(e)}")

            # Determine display year range
            if years_used:
                display_start = min(years_used)
                display_end = max(years_used)
            else:
                display_start = start_year
                display_end = end_year

            st.session_state.pb_results = {
                "grid_results": grid_results,
                "selected_grids": selected_grids,
                "grid_acres": grid_acres,
                "grid_allocations": grid_allocations,
                "start_year": display_start,
                "end_year": display_end,
                "coverage_level": coverage_level,
                "productivity_factor": productivity_factor,
                "intended_use": intended_use,
                "total_insured_acres": total_insured_acres,
                "current_rate_year": current_rate_year,
                "scenario": selected_scenario,
                "years_used": sorted(years_used)
            }

        except Exception as e:
            st.error(f"Error: {e}")
            st.exception(e)
            st.session_state.pb_results = None

    if 'pb_results' in st.session_state and st.session_state.pb_results:
        try:
            r = st.session_state.pb_results

            st.header(f"Portfolio Results ({r['start_year']}-{r['end_year']})")
            st.caption(f"Scenario: {r.get('scenario', 'All Years')} | Coverage: {r['coverage_level']:.0%} | Productivity: {r['productivity_factor']:.0%} | Use: {r['intended_use']}")

            if r.get('years_used'):
                st.caption(f"Years included: {len(r['years_used'])} ({min(r['years_used'])}-{max(r['years_used'])})")

            # === PORTFOLIO COVERAGE TABLE ===
            if len(r['selected_grids']) > 1:
                st.subheader("Portfolio Coverage")

                coverage_data = []
                total_coverage = {interval: 0 for interval in INTERVAL_ORDER_11}
                total_portfolio_acres = sum(r['grid_acres'].values())

                for gid in r['selected_grids']:
                    if gid in r['grid_results']:
                        allocation = r['grid_results'][gid]['allocation']
                        row = {'Grid': str(gid)[:20]}
                        row_sum = 0

                        for interval in INTERVAL_ORDER_11:
                            pct = allocation.get(interval, 0) * 100
                            row_sum += pct
                            total_coverage[interval] += pct
                            row[interval] = f"{pct:.0f}%" if pct > 0 else "--"

                        row['Row Sum'] = f"{row_sum:.0f}%"
                        row['Acres'] = f"{r['grid_acres'].get(gid, 0):,.0f}"
                        coverage_data.append(row)

                # Add average row
                avg_row = {'Grid': 'AVERAGE'}
                avg_row_sum = 0
                valid_grids_count = len([gid for gid in r['selected_grids'] if gid in r['grid_results']])
                for interval in INTERVAL_ORDER_11:
                    pct = total_coverage[interval] / valid_grids_count if valid_grids_count > 0 else 0
                    avg_row_sum += pct
                    avg_row[interval] = f"{pct:.0f}%" if pct > 0.5 else "--"
                avg_row['Row Sum'] = f"{avg_row_sum:.0f}%"
                avg_row['Acres'] = f"{total_portfolio_acres:,.0f}"
                coverage_data.append(avg_row)

                coverage_df = pd.DataFrame(coverage_data)

                csv_coverage = coverage_df.to_csv(index=False)
                st.download_button(
                    label="Download Coverage CSV",
                    data=csv_coverage,
                    file_name=f"portfolio_coverage_{r['start_year']}-{r['end_year']}.csv",
                    mime="text/csv",
                    key="pb_coverage_csv"
                )

                st.dataframe(coverage_df, use_container_width=True, hide_index=True)

                st.divider()

            st.subheader("Cumulative Results by Grid")

            combined_data = []
            portfolio_total_premium = 0
            portfolio_total_indemnity = 0
            portfolio_total_net_return = 0
            year_rois_all_grids = []

            for gid in r['selected_grids']:
                if gid in r['grid_results']:
                    results_df = r['grid_results'][gid]['results_df']

                    total_indemnity = results_df['Total Indemnity'].sum()
                    total_premium = results_df['Producer Premium'].sum()
                    net_return = results_df['Net Return'].sum()
                    cumulative_roi = net_return / total_premium if total_premium > 0 else 0

                    year_rois = results_df['Total ROI'].values
                    std_dev = np.std(year_rois) if len(year_rois) > 0 else 0
                    risk_adj_ret = cumulative_roi / std_dev if std_dev > 0 else 0

                    portfolio_total_premium += total_premium
                    portfolio_total_indemnity += total_indemnity
                    portfolio_total_net_return += net_return
                    year_rois_all_grids.extend(year_rois)

                    grid_acres_val = r['grid_acres'].get(gid, 0)

                    combined_data.append({
                        'Grid': str(gid)[:20],
                        'Acres': grid_acres_val,
                        'Total Premium': total_premium,
                        'Total Indemnity': total_indemnity,
                        'Net Return': net_return,
                        'Cumulative ROI': cumulative_roi,
                        'Std Dev': std_dev,
                        'Risk-Adj Return': risk_adj_ret
                    })

            # CSV Download Button
            csv_df = pd.DataFrame(combined_data)
            csv_export = csv_df.to_csv(index=False)

            st.download_button(
                label="Download Results CSV",
                data=csv_export,
                file_name=f"portfolio_results_{r['start_year']}-{r['end_year']}.csv",
                mime="text/csv",
                key="pb_results_csv"
            )

            # Display results table
            st.dataframe(
                pd.DataFrame(combined_data).style.format({
                    'Acres': '{:,.0f}',
                    'Total Premium': '${:,.0f}',
                    'Total Indemnity': '${:,.0f}',
                    'Net Return': '${:,.0f}',
                    'Cumulative ROI': '{:.2%}',
                    'Std Dev': '{:.2%}',
                    'Risk-Adj Return': '{:.2f}'
                }),
                use_container_width=True,
                hide_index=True
            )

            st.divider()
            st.subheader("Portfolio Metrics")

            portfolio_roi = portfolio_total_net_return / portfolio_total_premium if portfolio_total_premium > 0 else 0

            if len(year_rois_all_grids) > 0:
                portfolio_std_dev = np.std(year_rois_all_grids)
                portfolio_risk_adj = portfolio_roi / portfolio_std_dev if portfolio_std_dev > 0 else 0
            else:
                portfolio_std_dev = 0
                portfolio_risk_adj = 0

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Portfolio ROI", f"{portfolio_roi:.2%}")
            c2.metric("Risk-Adjusted Return", f"{portfolio_risk_adj:.2f}")
            c3.metric("Total Premium", f"${portfolio_total_premium:,.0f}")
            c4.metric("Total Net Return", f"${portfolio_total_net_return:,.0f}")

            st.divider()
            st.subheader("Details by Grid")

            all_grids_detail_data = []

            for gid in r['selected_grids']:
                if gid in r['grid_results']:
                    with st.expander(f"{gid} ({r['grid_acres'].get(gid, 0):,} acres)"):
                        results_df = r['grid_results'][gid]['results_df']
                        allocation = r['grid_results'][gid]['allocation']

                        alloc_display = {k: f"{v*100:.0f}%" for k, v in allocation.items() if v > 0}
                        st.text(f"Allocation: {', '.join([f'{k}: {v}' for k, v in alloc_display.items()])}")

                        st.dataframe(results_df.style.format({
                            'Year': '{:.0f}', 'Total Indemnity': '${:,.0f}',
                            'Producer Premium': '${:,.0f}', 'Net Return': '${:,.0f}', 'Total ROI': '{:.2%}'
                        }), use_container_width=True)

                    # Accumulate data for aggregate view
                    grid_df = results_df.copy()
                    grid_df.insert(0, 'Grid ID', gid)
                    all_grids_detail_data.append(grid_df)

            # Render Aggregate Portfolio Details Panel
            if all_grids_detail_data:
                master_audit_df = pd.concat(all_grids_detail_data, ignore_index=True)

                cols = master_audit_df.columns.tolist()
                priority_cols = ['Grid ID', 'Year']
                other_cols = [c for c in cols if c not in priority_cols]
                master_audit_df = master_audit_df[priority_cols + other_cols]

                with st.expander("Aggregate Portfolio Details (Audit View)"):
                    st.caption(f"Complete backtesting data for all {len(r['selected_grids'])} grids across all years.")

                    st.dataframe(master_audit_df.style.format({
                        'Year': '{:.0f}', 'Total Indemnity': '${:,.0f}',
                        'Producer Premium': '${:,.0f}', 'Net Return': '${:,.0f}', 'Total ROI': '{:.2%}'
                    }), use_container_width=True, height=400)

                    csv_data = master_audit_df.to_csv(index=False)
                    st.download_button(
                        label="Download Master Audit CSV",
                        data=csv_data,
                        file_name="portfolio_audit_details.csv",
                        mime="text/csv",
                        key="download_master_audit"
                    )

        except Exception as e:
            st.error(f"Error: {e}")
            st.exception(e)
            st.session_state.pb_results = None
    elif not st.session_state.pb_run:
        st.info("Configure grids and click 'Run Portfolio Backtest'")


# =============================================================================
# === HELPER FUNCTIONS FOR CHAMPION VS CHALLENGER ===
# =============================================================================

def run_portfolio_backtest(session, grids, allocations, acres, start_year, end_year,
                           coverage_level, productivity_factor, intended_use, plan_code, scenario='All Years'):
    """
    Run a complete portfolio backtest and return results dataframe, grid results, and metrics.
    """
    grid_results = {}
    all_year_data = []
    years_used = []

    for gid in grids:
        try:
            subsidy_percent = load_subsidy(session, plan_code, coverage_level)
            county_base_value = load_county_base_value(session, gid)
            premium_rates_df = load_premium_rates(session, gid, intended_use, coverage_level, get_current_rate_year(session))
            dollar_amount_of_protection = calculate_protection(county_base_value, coverage_level, productivity_factor)
            total_policy_protection = dollar_amount_of_protection * acres.get(gid, 1000)
            all_indices_df = load_all_indices(session, gid)

            # Apply scenario filter
            filtered_df = filter_indices_by_scenario(all_indices_df, scenario, start_year, end_year)

            year_results = []
            for year in sorted(filtered_df['YEAR'].unique()):
                actuals_df = filtered_df[filtered_df['YEAR'] == year].set_index('INTERVAL_NAME')
                if actuals_df.empty:
                    continue

                allocation = allocations.get(gid, {})
                roi_df = pd.DataFrame(index=INTERVAL_ORDER_11)
                roi_df['Percent of Value'] = roi_df.index.map(allocation)
                roi_df['Policy Protection Per Unit'] = (total_policy_protection * roi_df['Percent of Value']).apply(lambda x: round_half_up(x, 0))
                roi_df = roi_df.join(pd.Series(premium_rates_df, name='PREMIUM_RATE'))
                roi_df = roi_df.join(actuals_df.rename(columns={'INDEX_VALUE': 'Actual Index Value'}))
                roi_df['PREMIUM_RATE'] = pd.to_numeric(roi_df['PREMIUM_RATE'], errors='coerce').fillna(0)
                roi_df['Actual Index Value'] = pd.to_numeric(roi_df['Actual Index Value'], errors='coerce').fillna(0)
                roi_df['Total Premium'] = (roi_df['Policy Protection Per Unit'] * roi_df['PREMIUM_RATE']).apply(lambda x: round_half_up(x, 0))
                roi_df['Premium Subsidy'] = (roi_df['Total Premium'] * subsidy_percent).apply(lambda x: round_half_up(x, 0))
                roi_df['Producer Premium'] = roi_df['Total Premium'] - roi_df['Premium Subsidy']
                trigger_level = coverage_level * 100
                shortfall_pct = (trigger_level - roi_df['Actual Index Value']) / trigger_level
                roi_df['Estimated Indemnity'] = (shortfall_pct * roi_df['Policy Protection Per Unit']).clip(lower=0).apply(lambda x: round_half_up(x, 0) if abs(x) >= 0.01 else 0.0)

                total_indemnity = roi_df['Estimated Indemnity'].sum()
                total_producer_prem = roi_df['Producer Premium'].sum()
                year_roi = (total_indemnity - total_producer_prem) / total_producer_prem if total_producer_prem > 0 else 0.0

                year_results.append({
                    'Year': year, 'Grid': gid,
                    'Total Indemnity': total_indemnity,
                    'Producer Premium': total_producer_prem,
                    'Net Return': total_indemnity - total_producer_prem,
                    'ROI': year_roi
                })
                if year not in years_used:
                    years_used.append(year)

            grid_results[gid] = pd.DataFrame(year_results)
            all_year_data.extend(year_results)

        except Exception as e:
            st.error(f"Error processing {gid}: {e}")

    # Aggregate results
    results_df = pd.DataFrame(all_year_data)

    # Calculate portfolio metrics
    if not results_df.empty:
        total_premium = results_df['Producer Premium'].sum()
        total_indemnity = results_df['Total Indemnity'].sum()
        total_net = results_df['Net Return'].sum()
        cumulative_roi = total_net / total_premium if total_premium > 0 else 0

        # Calculate by year for volatility
        year_agg = results_df.groupby('Year').agg({
            'Producer Premium': 'sum',
            'Total Indemnity': 'sum',
            'Net Return': 'sum'
        }).reset_index()
        year_agg['ROI'] = year_agg['Net Return'] / year_agg['Producer Premium']
        year_agg['ROI'] = year_agg['ROI'].replace([np.inf, -np.inf], 0).fillna(0)

        std_dev = year_agg['ROI'].std() if len(year_agg) > 0 else 0
        risk_adj_return = cumulative_roi / std_dev if std_dev > 0 else 0
        profitable_years = len(year_agg[year_agg['ROI'] > 0])
        profitable_pct = profitable_years / len(year_agg) if len(year_agg) > 0 else 0

        metrics = {
            'cumulative_roi': cumulative_roi,
            'avg_annual_premium': total_premium / len(year_agg) if len(year_agg) > 0 else 0,
            'std_dev': std_dev,
            'risk_adj_return': risk_adj_return,
            'profitable_pct': profitable_pct,
            'total_premium': total_premium,
            'total_indemnity': total_indemnity,
            'total_net_return': total_net,
            'years_analyzed': len(year_agg)
        }
    else:
        metrics = {
            'cumulative_roi': 0, 'avg_annual_premium': 0, 'std_dev': 0,
            'risk_adj_return': 0, 'profitable_pct': 0, 'total_premium': 0,
            'total_indemnity': 0, 'total_net_return': 0, 'years_analyzed': 0
        }

    return results_df, grid_results, metrics


def create_performance_comparison_table(champ_metrics, chall_metrics):
    """Create a comparison table between Champion and Challenger metrics."""
    data = []
    metric_labels = {
        'cumulative_roi': ('Cumulative ROI', '{:.2%}'),
        'risk_adj_return': ('Risk-Adjusted Return', '{:.2f}'),
        'avg_annual_premium': ('Avg Annual Premium', '${:,.0f}'),
        'profitable_pct': ('Win Rate', '{:.1%}'),
        'std_dev': ('Volatility (Std Dev)', '{:.2%}'),
        'years_analyzed': ('Years Analyzed', '{:.0f}')
    }

    for key, (label, fmt) in metric_labels.items():
        champ_val = champ_metrics.get(key, 0)
        chall_val = chall_metrics.get(key, 0)
        diff = chall_val - champ_val

        data.append({
            'Metric': label,
            'Champion': fmt.format(champ_val),
            'Challenger': fmt.format(chall_val),
            'Difference': fmt.format(diff) if key != 'avg_annual_premium' else f'${diff:,.0f}'
        })

    return pd.DataFrame(data)


def create_optimized_allocation_table(allocations, grids, grid_acres=None, label="AVERAGE"):
    """Create a styled allocation table for display."""
    data = []

    for gid in grids:
        alloc = allocations.get(gid, {})
        row = {'Grid': str(gid)[:20]}
        row_sum = 0
        for interval in INTERVAL_ORDER_11:
            pct = alloc.get(interval, 0) * 100
            row_sum += pct
            row[interval] = pct
        row['Row Sum'] = row_sum
        if grid_acres:
            row['Acres'] = grid_acres.get(gid, 0)
        data.append(row)

    # Add average row
    avg_row = {'Grid': label}
    for interval in INTERVAL_ORDER_11:
        avg_pct = sum(allocations.get(gid, {}).get(interval, 0) for gid in grids) / len(grids) * 100 if grids else 0
        avg_row[interval] = avg_pct
    avg_row['Row Sum'] = sum(avg_row[interval] for interval in INTERVAL_ORDER_11)
    if grid_acres:
        avg_row['Acres'] = sum(grid_acres.values())
    data.append(avg_row)

    df = pd.DataFrame(data)

    # Create styled version
    def style_allocation(val):
        if isinstance(val, (int, float)) and val > 0:
            return 'background-color: #d4edda'
        return ''

    styled = df.style.applymap(style_allocation, subset=INTERVAL_ORDER_11)
    styled = styled.format({interval: '{:.0f}%' for interval in INTERVAL_ORDER_11})
    styled = styled.format({'Row Sum': '{:.0f}%'})
    if grid_acres:
        styled = styled.format({'Acres': '{:,.0f}'})

    return styled, df


def create_change_analysis_table(champ_alloc, chall_alloc, champ_acres, chall_acres, grids):
    """Create a table showing changes between Champion and Challenger allocations."""
    data = []

    for gid in grids:
        row = {'Grid': str(gid)[:20]}

        champ_a = champ_alloc.get(gid, {})
        chall_a = chall_alloc.get(gid, {})

        for interval in INTERVAL_ORDER_11:
            champ_pct = champ_a.get(interval, 0) * 100
            chall_pct = chall_a.get(interval, 0) * 100
            diff = chall_pct - champ_pct

            if abs(diff) < 0.5:
                row[interval] = '--'
            elif diff > 0:
                row[interval] = f'+{diff:.0f}%'
            else:
                row[interval] = f'{diff:.0f}%'

        # Acres change
        champ_ac = champ_acres.get(gid, 0)
        chall_ac = chall_acres.get(gid, 0)
        acre_diff = chall_ac - champ_ac
        row['Acre Change'] = f'{acre_diff:+,.0f}' if acre_diff != 0 else '--'

        data.append(row)

    # Add totals row
    total_row = {'Grid': 'PORTFOLIO TOTALS'}
    for interval in INTERVAL_ORDER_11:
        champ_avg = sum(champ_alloc.get(gid, {}).get(interval, 0) for gid in grids) / len(grids) * 100 if grids else 0
        chall_avg = sum(chall_alloc.get(gid, {}).get(interval, 0) for gid in grids) / len(grids) * 100 if grids else 0
        diff = chall_avg - champ_avg
        if abs(diff) < 0.5:
            total_row[interval] = '--'
        elif diff > 0:
            total_row[interval] = f'+{diff:.0f}%'
        else:
            total_row[interval] = f'{diff:.0f}%'

    total_acre_diff = sum(chall_acres.values()) - sum(champ_acres.values())
    total_row['Acre Change'] = f'{total_acre_diff:+,.0f}' if total_acre_diff != 0 else '--'
    data.append(total_row)

    df = pd.DataFrame(data)

    def style_change(val):
        if isinstance(val, str):
            if val.startswith('+'):
                return 'color: green; font-weight: bold'
            elif val.startswith('-'):
                return 'color: red; font-weight: bold'
        return ''

    styled = df.style.applymap(style_change)

    return styled, df


# =============================================================================
# === TAB 3: CHAMPION VS CHALLENGER(S) ===
# =============================================================================

def render_champion_vs_challenger_tab(session, grid_id, intended_use, productivity_factor, total_insured_acres, plan_code):
    """
    Champion vs Challenger(s) tab - Enhanced from Layer 1's "Portfolio Strategy"
    Now includes weather view challengers (from Layer 3).
    """
    st.subheader("Champion vs Challenger(s)")
    st.caption("Compare your baseline strategy against optimized alternatives, including weather-conditioned challengers.")

    # === PRESET LOADING BUTTON ===
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("Load King Ranch", key="cvc_load_kr"):
            try:
                all_grids_for_preset = load_distinct_grids(session)

                target_grid_mapping = {}
                for county, grid_ids in KING_RANCH_PRESET['counties'].items():
                    for grid_id_num in grid_ids:
                        target_grid_mapping[grid_id_num] = f"{grid_id_num} ({county} - TX)"

                preset_grid_ids = []
                for numeric_id in KING_RANCH_PRESET['grids']:
                    target_str = target_grid_mapping.get(numeric_id, "")
                    if target_str in all_grids_for_preset:
                        preset_grid_ids.append(target_str)
                    else:
                        for grid_option in all_grids_for_preset:
                            if extract_numeric_grid_id(grid_option) == numeric_id:
                                preset_grid_ids.append(grid_option)
                                break

                st.session_state.cvc_grids = preset_grid_ids
                st.session_state.cvc_use_custom_acres = True

                for gid in preset_grid_ids:
                    numeric_id = extract_numeric_grid_id(gid)
                    st.session_state[f"cvc_champ_acres_{gid}"] = KING_RANCH_PRESET['acres'].get(numeric_id, total_insured_acres)
                    alloc = KING_RANCH_PRESET['allocations'].get(numeric_id, {})
                    alloc_decimal = {interval: float(alloc.get(interval, 0.0)) / 100.0 for interval in INTERVAL_ORDER_11}
                    st.session_state[f"cvc_champ_{gid}_preset_allocation"] = alloc_decimal

                st.session_state.productivity_factor = 1.35
                st.session_state.cvc_coverage = 0.75

                st.success("King Ranch loaded! (8 grids, 135% productivity, 75% coverage)")

            except Exception as e:
                st.error(f"Error loading King Ranch: {e}")

    with col2:
        st.caption("Auto-populate King Ranch strategy")

    st.divider()

    try:
        all_grids = load_distinct_grids(session)
    except:
        all_grids = [grid_id]

    default_grids = st.session_state.get('cvc_grids', [grid_id])
    default_grids = [g for g in default_grids if g in all_grids]

    selected_grids = st.multiselect(
        "Select Grids",
        options=all_grids,
        default=default_grids,
        max_selections=20,
        key="cvc_grids"
    )

    if not selected_grids:
        st.warning("Select at least one grid")
        return

    st.divider()

    # === SCENARIO AND PARAMETERS ===
    st.markdown("### Scenario and Parameters")

    col1, col2, col3 = st.columns(3)

    with col1:
        scenario_options = [
            'All Years (except Current Year)',
            'ENSO Phase: La Nina',
            'ENSO Phase: El Nino',
            'ENSO Phase: Neutral',
            'Custom Range'
        ]
        selected_scenario = st.selectbox("Scenario", scenario_options, key="cvc_scenario")

    with col2:
        coverage_level = st.selectbox(
            "Coverage Level",
            [0.70, 0.75, 0.80, 0.85, 0.90],
            index=2,
            format_func=lambda x: f"{x:.0%}",
            key="cvc_coverage"
        )

    with col3:
        st.metric("Productivity Factor", f"{productivity_factor:.0%}")

    if selected_scenario == 'Custom Range':
        col1, col2 = st.columns(2)
        start_year = col1.selectbox("Start Year", list(range(1948, 2026)), index=62, key="cvc_start")
        end_year = col2.selectbox("End Year", list(range(1948, 2026)), index=76, key="cvc_end")
    else:
        start_year = 1948
        end_year = 2024

    st.divider()

    # ==========================================================================
    # THE CHAMPION (BASELINE)
    # ==========================================================================
    st.markdown("### The Champion (Baseline)")
    st.caption("Define your baseline strategy. This is what the Challenger will try to beat.")

    # Champion Acreage Configuration
    with st.expander("Champion Acreage per Grid", expanded=True):
        champion_acres = {}
        cols = st.columns(min(4, len(selected_grids)))
        for idx, gid in enumerate(selected_grids):
            with cols[idx % 4]:
                numeric_id = extract_numeric_grid_id(gid)
                default_acres = st.session_state.get(f"cvc_champ_acres_{gid}", KING_RANCH_PRESET['acres'].get(numeric_id, total_insured_acres))
                champion_acres[gid] = st.number_input(
                    f"{gid}",
                    min_value=1,
                    value=default_acres,
                    step=10,
                    key=f"cvc_champ_acres_{gid}"
                )

    # Champion Interval Allocations
    champion_allocations = {}
    champion_all_valid = True

    with st.expander("Champion Interval Allocations", expanded=False):
        for gid in selected_grids:
            st.markdown(f"**{gid}**")
            alloc_dict, is_valid = render_allocation_inputs(f"cvc_champ_{gid}")
            champion_allocations[gid] = alloc_dict
            if not is_valid:
                champion_all_valid = False

    # Champion Run Button
    if st.button("Run Champion Backtest", key="cvc_run_champion", disabled=not champion_all_valid):
        with st.spinner("Running Champion backtest..."):
            champion_df, champion_grid_results, champion_metrics = run_portfolio_backtest(
                session, selected_grids, champion_allocations, champion_acres,
                start_year, end_year, coverage_level, productivity_factor,
                intended_use, plan_code, selected_scenario
            )

            st.session_state.champion_results = {
                'df': champion_df,
                'grid_results': champion_grid_results,
                'metrics': champion_metrics,
                'allocations': champion_allocations,
                'acres': champion_acres,
                'grids': selected_grids
            }
            st.success("Champion backtest complete!")

    # Display Champion Results (if available)
    if 'champion_results' in st.session_state and st.session_state.champion_results:
        champ = st.session_state.champion_results
        metrics = champ.get('metrics', {})

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Cumulative ROI", f"{metrics.get('cumulative_roi', 0):.1%}")
        col2.metric("Risk-Adj Return", f"{metrics.get('risk_adj_return', 0):.2f}")
        col3.metric("Avg Annual Premium", f"${metrics.get('avg_annual_premium', 0):,.0f}")
        col4.metric("Win Rate", f"{metrics.get('profitable_pct', 0):.0%}")

    st.divider()

    # ==========================================================================
    # WEATHER VIEW CHALLENGERS (FROM LAYER 3)
    # ==========================================================================
    st.markdown("### Weather View Challengers")
    st.caption("Generate challengers based on climate regime filtering using analog years.")

    with st.expander("Configure Weather View", expanded=False):
        wv_col1, wv_col2 = st.columns(2)

        with wv_col1:
            historical_context = st.selectbox(
                "Historical Context (Z-Score)",
                list(HISTORICAL_CONTEXT_MAP.keys()),
                index=1,  # Default to "Normal (-0.25 to 0.25)"
                key="cvc_historical_context"
            )

        with wv_col2:
            trend = st.selectbox(
                "Expected Trend",
                list(TREND_MAP.keys()),
                index=1,  # Default to "Stay Stable"
                key="cvc_trend"
            )

        st.caption("This will filter historical years to find analog years matching your weather view, then backtest using only those years.")

        if st.button("Generate Weather View Challenger", key="cvc_gen_weather_challenger"):
            if 'champion_results' not in st.session_state or not st.session_state.champion_results:
                st.warning("Please run the Champion backtest first!")
            else:
                with st.spinner("Finding analog years and running backtest..."):
                    champ = st.session_state.champion_results

                    # Find portfolio-aggregated analog years
                    analog_years = find_portfolio_aggregated_analog_years(
                        session=session,
                        grid_ids=selected_grids,
                        historical_context=historical_context,
                        trend=trend,
                        min_year=start_year,
                        max_year=end_year
                    )

                    if not analog_years:
                        st.warning(f"No analog years found for {historical_context} + {trend}")
                    else:
                        st.info(f"Found {len(analog_years)} analog years: {sorted(analog_years)}")

                        # Run backtest on analog years only using Custom Range
                        weather_df, weather_grid_results, weather_metrics = run_portfolio_backtest(
                            session, selected_grids, champ['allocations'], champ['acres'],
                            min(analog_years), max(analog_years), coverage_level, productivity_factor,
                            intended_use, plan_code, 'Custom Range'
                        )

                        # Filter to only include analog years
                        if not weather_df.empty:
                            weather_df = weather_df[weather_df['Year'].isin(analog_years)]

                        st.session_state.weather_challenger_results = {
                            'df': weather_df,
                            'grid_results': weather_grid_results,
                            'metrics': weather_metrics,
                            'allocations': champ['allocations'],
                            'acres': champ['acres'],
                            'grids': selected_grids,
                            'analog_years': analog_years,
                            'view': f"{historical_context} + {trend}"
                        }
                        st.success(f"Weather View Challenger complete! ({len(analog_years)} analog years)")

    # Display Weather Challenger Results
    if 'weather_challenger_results' in st.session_state and st.session_state.weather_challenger_results:
        wc = st.session_state.weather_challenger_results
        wc_metrics = wc.get('metrics', {})

        st.markdown(f"**Weather View: {wc.get('view', 'N/A')}** ({len(wc.get('analog_years', []))} analog years)")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Cumulative ROI", f"{wc_metrics.get('cumulative_roi', 0):.1%}")
        col2.metric("Risk-Adj Return", f"{wc_metrics.get('risk_adj_return', 0):.2f}")
        col3.metric("Avg Annual Premium", f"${wc_metrics.get('avg_annual_premium', 0):,.0f}")
        col4.metric("Win Rate", f"{wc_metrics.get('profitable_pct', 0):.0%}")

    st.divider()

    # ==========================================================================
    # COMPARISON OUTPUT
    # ==========================================================================
    if 'champion_results' in st.session_state and st.session_state.champion_results:
        st.markdown("### Results Comparison")

        champ = st.session_state.champion_results
        champ_metrics = champ.get('metrics', {})

        # Compare with Weather Challenger if available
        if 'weather_challenger_results' in st.session_state and st.session_state.weather_challenger_results:
            wc = st.session_state.weather_challenger_results
            wc_metrics = wc.get('metrics', {})

            st.markdown("#### Champion vs Weather View Challenger")
            comparison_df = create_performance_comparison_table(champ_metrics, wc_metrics)
            st.table(comparison_df.set_index('Metric'))

            # Winner Banner
            wc_roi = wc_metrics.get('cumulative_roi', 0)
            champ_roi = champ_metrics.get('cumulative_roi', 0)

            if wc_roi > champ_roi:
                improvement = ((wc_roi - champ_roi) / abs(champ_roi) * 100) if champ_roi != 0 else 0
                st.success(f"**Weather View performs better** under {wc.get('view', 'selected conditions')}! ROI differs by {improvement:.1f}%")
            elif wc_roi < champ_roi:
                st.warning(f"**Champion performs better** than Weather View under {wc.get('view', 'selected conditions')}")
            else:
                st.info("**TIE!** Both perform equally.")

            st.caption(f"Note: Weather View Challenger uses {len(wc.get('analog_years', []))} analog years vs Champion's full historical range.")

        st.divider()

        # === ALLOCATION DISPLAY ===
        st.markdown("#### Champion Allocation Summary")

        champ_styled, champ_df = create_optimized_allocation_table(
            champ['allocations'], champ['grids'], grid_acres=champ['acres'],
            label="CHAMPION AVERAGE"
        )
        st.dataframe(champ_styled, use_container_width=True, hide_index=True)

        st.download_button(
            label="Download Champion CSV",
            data=champ_df.to_csv(index=False),
            file_name="champion_allocations.csv",
            mime="text/csv",
            key="download_champion_alloc_csv"
        )


# =============================================================================
# === MAIN APPLICATION ===
# =============================================================================

def main():
    st.title("PRF Analysis Tool")
    st.caption("Consolidated PRF Insurance Analysis - Decision Support, Backtesting, and Strategy Optimization")

    session = get_active_session()

    # Load available grids
    try:
        valid_grids = load_distinct_grids(session)
    except Exception as e:
        st.sidebar.error("Fatal Error: Could not load Grid ID list")
        st.error(f"Could not load Grid ID list: {e}")
        st.stop()

    # === SIDEBAR: COMMON PARAMETERS ===
    st.sidebar.header("Common Parameters")

    # Initialize session state
    if 'grid_id' not in st.session_state:
        st.session_state.grid_id = valid_grids[0] if valid_grids else "7928 (Nueces - TX)"
    if 'productivity_factor' not in st.session_state:
        st.session_state.productivity_factor = 1.0
    if 'total_insured_acres' not in st.session_state:
        st.session_state.total_insured_acres = 1000
    if 'intended_use' not in st.session_state:
        st.session_state.intended_use = 'Grazing'
    if 'insurance_plan_code' not in st.session_state:
        st.session_state.insurance_plan_code = 13

    # Grid selection
    try:
        default_grid_index = valid_grids.index(st.session_state.grid_id)
    except (ValueError, AttributeError):
        default_grid_index = 0

    grid_id = st.sidebar.selectbox(
        "Grid ID",
        options=valid_grids,
        index=default_grid_index,
        key="sidebar_grid_id"
    )

    # Productivity factor
    prod_options = list(range(60, 151))
    prod_options_formatted = [f"{x}%" for x in prod_options]
    try:
        current_prod_index = prod_options.index(int(st.session_state.productivity_factor * 100))
    except ValueError:
        current_prod_index = 40  # Default to 100%

    selected_prod_str = st.sidebar.selectbox(
        "Productivity Factor",
        options=prod_options_formatted,
        index=current_prod_index,
        key="sidebar_prod_factor"
    )
    productivity_factor = int(selected_prod_str.replace('%', '')) / 100.0

    # Acres
    total_insured_acres = st.sidebar.number_input(
        "Total Insured Acres",
        value=st.session_state.total_insured_acres,
        step=10,
        key="sidebar_acres"
    )

    # Intended use
    intended_use = st.sidebar.selectbox(
        "Intended Use",
        ['Grazing', 'Haying'],
        index=0 if st.session_state.intended_use == 'Grazing' else 1,
        key="sidebar_use"
    )

    # Plan code (disabled)
    plan_code = st.sidebar.number_input(
        "Insurance Plan Code",
        value=st.session_state.insurance_plan_code,
        disabled=True
    )

    # Update session state
    st.session_state.grid_id = grid_id
    st.session_state.productivity_factor = productivity_factor
    st.session_state.total_insured_acres = total_insured_acres
    st.session_state.intended_use = intended_use

    st.sidebar.divider()
    st.sidebar.caption("*2025 Rates are used for this application")
    st.sidebar.caption("*Common Parameters are secondary to tab-specific parameters")

    # Z-Score Translation Key (from Layer 3)
    with st.sidebar.expander("Z-Score Translation Key"):
        st.markdown("**Historical Context (Year Avg Z-Score):**")
        st.markdown("- **Dry:** Z < -0.25 (~30th percentile)")
        st.markdown("- **Normal:** Z -0.25 to +0.25 (~30th-70th)")
        st.markdown("- **Wet:** Z > +0.25 (~70th percentile)")
        st.markdown("---")
        st.markdown("**Trajectory (EOY 5P - SOY 11P):**")
        st.markdown("- **Get Drier:** Delta < -0.2")
        st.markdown("- **Stay Stable:** Delta -0.2 to +0.2")
        st.markdown("- **Get Wetter:** Delta > +0.2")

    # === MAIN CONTENT: THREE TABS ===
    tab_decision, tab_backtest, tab_challenger = st.tabs([
        "Decision Support (Audit)",
        "Portfolio Backtest (Audit)",
        "Champion vs Challenger(s)"
    ])

    with tab_decision:
        render_decision_support_tab(session, grid_id, intended_use, productivity_factor, total_insured_acres, plan_code)

    with tab_backtest:
        render_portfolio_backtest_tab(session, grid_id, intended_use, productivity_factor, total_insured_acres, plan_code)

    with tab_challenger:
        render_champion_vs_challenger_tab(session, grid_id, intended_use, productivity_factor, total_insured_acres, plan_code)


if __name__ == "__main__":
    main()
