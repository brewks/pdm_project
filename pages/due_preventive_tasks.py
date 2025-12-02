import streamlit as st
import pandas as pd
import sqlite3

DB_PATH = "ga_maintenance.db"  # local SQLite database for the PdM system

# Page title
st.title("🛠 Due Preventive Maintenance Tasks (FAA-Aligned)")

# Pull due tasks from the database view
@st.cache_data  # cache so we don't hit the DB on every interaction
def load_due_tasks():
    with sqlite3.connect(DB_PATH) as conn:
        query = """
        SELECT * FROM due_preventive_tasks
        ORDER BY timestamp DESC
        """
        # Bring the SQL result straight into a DataFrame
        return pd.read_sql_query(query, conn)

# Load tasks once
tasks_df = load_due_tasks()

# If there are no open items, show a simple “all clear” message
if tasks_df.empty:
    st.info("✅ No pending preventive maintenance tasks at this time.")
else:
    # Show how many tasks need attention
    st.write(f"### 🔧 {len(tasks_df)} Task(s) Requiring Attention")
    st.dataframe(tasks_df)

    # Sidebar filters so a mechanic can slice by system or tail number
    st.sidebar.subheader("🔍 Filter Tasks")
    systems = ["All"] + sorted(tasks_df['system'].dropna().unique().tolist())
    selected_system = st.sidebar.selectbox("System", systems)

    tails = ["All"] + sorted(tasks_df['tail_number'].dropna().unique().tolist())
    selected_tail = st.sidebar.selectbox("Tail Number", tails)

    # Apply filters if the user selects a specific system or aircraft
    if selected_system != "All":
        tasks_df = tasks_df[tasks_df['system'] == selected_system]
    if selected_tail != "All":
        tasks_df = tasks_df[tasks_df['tail_number'] == selected_tail]

    st.write("### Filtered View")
    st.dataframe(tasks_df)

    # Allow download of the filtered list for shop planning or records
    st.download_button(
        label="📥 Download CSV",
        data=tasks_df.to_csv(index=False).encode(),
        file_name="due_preventive_tasks.csv",
        mime="text/csv"
    )
