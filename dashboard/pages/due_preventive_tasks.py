"""
Due Preventive Maintenance Tasks Dashboard Page.

This page displays FAA AC 43-12C–aligned preventive maintenance tasks that are
due based on model predictions and FMEA/RPN thresholds. It provides:
- List of all pending preventive tasks
- Filtering by system and tail number
- Task details including pilot-allowed vs A&P-required
- CSV export capability

This demonstrates the FAA compliance and maintenance recommendation capabilities.

Author: General Aviation PdM Team
Date: 2025
"""

import streamlit as st
import pandas as pd
from typing import List

from config.db_utils import execute_query
from config.logging_config import setup_logger

# Initialize logger
logger = setup_logger(__name__)


# ============================================================================
# DATA LOADING
# ============================================================================

@st.cache_data(ttl=60)
def load_due_tasks() -> pd.DataFrame:
    """
    Load due preventive tasks from the database view.

    Returns:
        pd.DataFrame: Due preventive tasks ordered by timestamp
    """
    query = """
        SELECT * FROM due_preventive_tasks
        ORDER BY timestamp DESC
    """
    try:
        return execute_query(query)
    except Exception as e:
        logger.error(f"Failed to load due tasks: {e}")
        st.error("Failed to load due preventive tasks.")
        return pd.DataFrame()


# ============================================================================
# FILTERING
# ============================================================================

def get_unique_values(df: pd.DataFrame, column: str) -> List[str]:
    """
    Get sorted unique values from a DataFrame column.

    Args:
        df: DataFrame to extract values from
        column: Column name

    Returns:
        List[str]: Sorted list of unique values with "All" prepended
    """
    if column not in df.columns or df.empty:
        return ["All"]

    values = df[column].dropna().unique().tolist()
    return ["All"] + sorted(values)


def apply_filters(
    df: pd.DataFrame,
    system_filter: str,
    tail_filter: str
) -> pd.DataFrame:
    """
    Apply system and tail number filters to tasks DataFrame.

    Args:
        df: Tasks DataFrame
        system_filter: Selected system filter
        tail_filter: Selected tail number filter

    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    filtered = df.copy()

    if system_filter != "All":
        filtered = filtered[filtered['system'] == system_filter]

    if tail_filter != "All":
        filtered = filtered[filtered['tail_number'] == tail_filter]

    logger.debug(f"Applied filters: system={system_filter}, tail={tail_filter}. "
                 f"Result: {len(filtered)} tasks")

    return filtered


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """
    Main application for displaying due preventive maintenance tasks.

    Provides filtering, sorting, and export capabilities for FAA-compliant
    maintenance task management.
    """
    logger.info("Due Preventive Tasks page loaded")

    st.title("🛠 Due Preventive Maintenance Tasks (FAA-Aligned)")

    # Load tasks
    tasks_df = load_due_tasks()

    if tasks_df.empty:
        st.info("✅ No pending preventive maintenance tasks at this time.")
        logger.info("No pending preventive tasks")
        return

    # Display summary
    st.write(f"### 🔧 {len(tasks_df):,} Task(s) Requiring Attention")

    # Sidebar filters
    st.sidebar.subheader("🔍 Filter Tasks")

    systems = get_unique_values(tasks_df, 'system')
    selected_system = st.sidebar.selectbox("System", systems)

    tails = get_unique_values(tasks_df, 'tail_number')
    selected_tail = st.sidebar.selectbox("Tail Number", tails)

    # Apply filters
    filtered_df = apply_filters(tasks_df, selected_system, selected_tail)

    # Display full dataset
    st.write("### All Tasks")
    st.dataframe(tasks_df, use_container_width=True)

    # Display filtered view if filters are active
    if selected_system != "All" or selected_tail != "All":
        st.write("### Filtered View")
        st.dataframe(filtered_df, use_container_width=True)

        if filtered_df.empty:
            st.info("No tasks match the selected filters.")
    else:
        filtered_df = tasks_df  # Use full dataset for export

    # Task breakdown by system (if system column exists)
    if 'system' in tasks_df.columns and not tasks_df.empty:
        st.write("### Tasks by System")
        system_counts = tasks_df['system'].value_counts()
        st.bar_chart(system_counts)

    # Download button
    if not filtered_df.empty:
        st.markdown("---")
        st.download_button(
            label="📥 Download Filtered Tasks as CSV",
            data=filtered_df.to_csv(index=False).encode(),
            file_name="due_preventive_tasks.csv",
            mime="text/csv"
        )
        logger.info(f"Prepared download for {len(filtered_df)} tasks")


if __name__ == "__main__":
    main()
