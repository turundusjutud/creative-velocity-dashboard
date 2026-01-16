import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Turundusjutud | Creative Analytics", 
    page_icon="📊",
    layout="wide"
)

# --- CUSTOM BRANDING CSS (FIXED FOR VISIBILITY) ---
st.markdown("""
    <style>
        /* 1. Force Background & Text Colors */
        .stApp {
            background-color: #FAFAFA;
        }
        
        p, div, label, span {
            color: #052623 !important; /* Force Dark Green Text */
        }
        
        /* 2. Headings */
        h1, h2, h3, h4, h5, h6 {
            color: #052623 !important;
            font-family: 'Helvetica', 'Arial', sans-serif;
            font-weight: 700;
        }
        
        /* 3. Metrics Cards (The big numbers) */
        div[data-testid="stMetric"] {
            background-color: #FFFFFF;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            border: 1px solid #E5E7EB;
            border-left: 5px solid #1A776F; /* Brand Teal Accent */
        }
        
        div[data-testid="stMetric"] label {
            color: #1A776F !important; /* Label in Teal */
            font-size: 14px;
        }
        
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
            color: #052623 !important; /* Value in Dark Green */
        }

        /* 4. Buttons (Brand Orange) */
        div.stButton > button {
            background-color: #FF7F40;
            color: white !important;
            border-radius: 8px;
            border: none;
            padding: 0.5rem 1rem;
            font-weight: bold;
        }
        div.stButton > button:hover {
            background-color: #E66A2E; /* Darker Orange on hover */
            color: white !important;
            border-color: #E66A2E;
        }

        /* 5. Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #FFFFFF;
            border-right: 1px solid #E5E7EB;
        }
        
        /* 6. Success/Info Boxes */
        div.stAlert {
            background-color: #F0FDFA; /* Very light teal */
            border: 1px solid #1A776F;
            color: #052623;
        }
        
        /* 7. Charts Background */
        .js-plotly-plot .plotly .main-svg {
            background: rgba(0,0,0,0) !important;
        }
    </style>
""", unsafe_allow_html=True)

# --- PRIVACY NOTICE ---
st.info("""
    **🔒 Privacy & Data Security:** This tool processes data **locally in your browser session**. 
    No data is saved to any server. Closing this tab deletes all data permanently.
""")

st.title("Creative Strategy Analytics")

# --- INSTRUCTIONS ---
with st.expander("📝 How to export data from Meta Ads (Click to read)", expanded=False):
    st.markdown("""
    1. Go to **Ads Manager** -> **Reports** -> **Export Table Data**.
    2. **Breakdown:** Select **"Day"** (Crucial!).
    3. **Level:** Select **"Ad"**.
    4. **Columns:** `Ad ID`, `Reporting Starts`, `Amount Spent`, `Impressions`, `Link Clicks`, `Purchases` (or your key metric).
    5. **Export:** Save as .csv and upload on the left.
    """)

# --- HELPER FUNCTIONS ---
def load_data(file):
    df = pd.read_csv(file)
    df.columns = [c.lower().strip() for c in df.columns]
    
    # Auto-detect date
    date_cols = [c for c in df.columns if 'date' in c or 'start' in c]
    if not date_cols:
        return None, None, None, None, None, None, None, None
    date_col = date_cols[0]
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Auto-detect metrics
    spend_col = next((c for c in df.columns if 'amount' in c or 'spend' in c), None)
    imps_col = next((c for c in df.columns if 'impression' in c), None)
    clicks_col = next((c for c in df.columns if 'link click' in c or 'clicks' in c), None)
    ad_id_col = next((c for c in df.columns if 'ad id' in c), None)
    installs_col = next((c for c in df.columns if 'install' in c), None)
    if not installs_col:
        installs_col = next((c for c in df.columns if 'result' in c), None)
    value_col = next((c for c in df.columns if 'value' in c or 'revenue' in c), None)
    
    return df, date_col, spend_col, imps_col, installs_col, clicks_col, value_col, ad_id_col

# --- MAIN APP ---
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=['csv'])

if uploaded_file is not None:
    raw_df, date_col, spend_col, imps_col, installs_col, clicks_col, value_col, ad_id_col = load_data(uploaded_file)
    
    if raw_df is not None and spend_col and ad_id_col:
        st.sidebar.success("✅ Data Loaded Successfully")
        
        st.sidebar.markdown("---")
        st.sidebar.header("⚙️ Settings")
        min_spend = st.sidebar.number_input("Min. Spend per Creative", value=0)
        
        # --- PROCESSING ---
        creative_agg = raw_df.groupby(ad_id_col).agg({date_col: 'min', spend_col: 'sum'}).reset_index()
        creative_agg.columns = [ad_id_col, 'launch_date', 'lifetime_spend']
        valid_ads = creative_agg[creative_agg['lifetime_spend'] >= min_spend][ad_id_col]
        raw_df = raw_df[raw_df[ad_id_col].isin(valid_ads)]
        creative_birthdays = creative_agg[creative_agg[ad_id_col].isin(valid_ads)][[ad_id_col, 'launch_date']]
        
        raw_df['week_start'] = raw_df[date_col].dt.to_period('W-MON').apply(lambda r: r.start_time)
        numeric_cols = raw_df.select_dtypes(include=[np.number]).columns.tolist()
        weekly_metrics = raw_df.groupby('week_start')[numeric_cols].sum().reset_index()
        
        creative_birthdays['launch_week'] = creative_birthdays['launch_date'].dt.to_period('W-MON').apply(lambda r: r.start_time)
        weekly_new_creatives = creative_birthdays.groupby('launch_week')[ad_id_col].count().reset_index()
        weekly_new_creatives.columns = ['week_start', 'new_creatives_count']
        
        analysis_df = pd.merge(weekly_metrics, weekly_new_creatives, on='week_start', how='left').fillna(0)
        
        if installs_col and imps_col:
            analysis_df['Calculated_IPM'] = (analysis_df[installs_col] / analysis_df[imps_col]) * 1000
            analysis_df['Calculated_CPA'] = analysis_df[spend_col] / analysis_df[installs_col]
        
        analysis_df['Calculated_CPM'] = (analysis_df[spend_col] / analysis_df[imps_col]) * 1000
        
        if clicks_col and imps_col:
            analysis_df['Calculated_CTR'] = (analysis_df[clicks_col] / analysis_df[imps_col]) * 100
        if value_col and spend_col:
            analysis_df['Calculated_ROAS'] = analysis_df[value_col] / analysis_df[spend_col]

        analysis_df = analysis_df[analysis_df[spend_col] > 0]

        # --- 1. VELOCITY ---
        st.header("1. Velocity Impact")
        available_metrics = list(analysis_df.select_dtypes(include=[np.number]).columns)
        available_metrics = [m for m in available_metrics if 'id' not in m and 'week' not in m]
        
        c1, c2 = st.columns([2,1])
        metric_choice = c1.selectbox("KPI:", available_metrics, index=0)
        lag_weeks = c2.slider("Lag (Weeks):", 0, 8, 0)

        analysis_df['lagged_uploads'] = analysis_df['new_creatives_count'].shift(lag_weeks)
        valid_data = analysis_df.dropna(subset=['lagged_uploads', metric_choice])

        if len(valid_data) > 2:
            corr = valid_data['lagged_uploads'].corr(valid_data[metric_choice])
            st.metric("Correlation", f"{corr:.3f}")

        fig = go.Figure()
        fig.add_trace(go.Bar(x=valid_data['week_start'], y=valid_data['lagged_uploads'], name='New Creatives', marker_color='rgba(5, 38, 35, 0.2)', yaxis='y'))
        fig.add_trace(go.Scatter(x=valid_data['week_start'], y=valid_data[metric_choice], name=metric_choice, mode='lines+markers', line=dict(color='#1A776F', width=3), yaxis='y2'))
        
        fig.update_layout(
            title=f'New Creatives vs {metric_choice}',
            xaxis=dict(title='Week'),
            yaxis=dict(title='Count', side='left', showgrid=False),
            yaxis2=dict(title=metric_choice, side='right', overlaying='y', showgrid=False),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#052623')
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- 2. FRESH VS FATIGUED ---
        st.markdown("---")
        st.header("2. Fresh vs. Fatigued")
        
        raw_with_birthdays = pd.merge(raw_df, creative_birthdays, on=ad_id_col, how='left')
        raw_with_birthdays['age'] = (raw_with_birthdays[date_col] - raw_with_birthdays['launch_date']).dt.days
        raw_with_birthdays['Status'] = raw_with_birthdays['age'].apply(lambda x: 'Fresh (<21 Days)' if x < 21 else 'Fatigued (21+ Days)')

        metric_options = {}
        if installs_col and imps_col: metric_options['IPM'] = 'calc_ipm'; metric_options['CPA'] = 'calc_cpa'
        if clicks_col and imps_col: metric_options['CTR'] = 'calc_ctr'
        if value_col and spend_col: metric_options['ROAS'] = 'calc_roas'
        metric_options['CPM'] = 'calc_cpm'
        
        s2_choice = st.selectbox("Compare Metric:", list(metric_options.keys()), index=0)
        
        eff_comp = raw_with_birthdays.groupby('Status')[numeric_cols].sum().reset_index()
        sel_logic = metric_options[s2_choice]
        
        if sel_logic == 'calc_cpa': eff_comp['val'] = eff_comp[spend_col] / eff_comp[installs_col]
        elif sel_logic == 'calc_ipm': eff_comp['val'] = (eff_comp[installs_col] / eff_comp[imps_col]) * 1000
        elif sel_logic == 'calc_ctr': eff_comp['val'] = (eff_comp[clicks_col] / eff_comp[imps_col]) * 100
        elif sel_logic == 'calc_roas': eff_comp['val'] = eff_comp[value_col] / eff_comp[spend_col]
        elif sel_logic == 'calc_cpm': eff_comp['val'] = (eff_comp[spend_col] / eff_comp[imps_col]) * 1000
        
        if len(eff_comp) >= 2:
            fresh_val = eff_comp.loc[eff_comp['Status'].str.contains('Fresh'), 'val'].values[0]
            old_val = eff_comp.loc[eff_comp['Status'].str.contains('Fatigued'), 'val'].values[0]
            if old_val == 0: old_val = 0.0001
            diff = ((fresh_val - old_val) / old_val) * 100
            
            c1, c2 = st.columns(2)
            c1.metric(f"Fresh {s2_choice}", f"{fresh_val:,.2f}", f"{diff:.1f}%")
            
            fig_comp = px.bar(
                eff_comp, x='Status', y='val', 
                color='Status', 
                color_discrete_map={'Fresh (<21 Days)': '#1A776F', 'Fatigued (21+ Days)': '#FF7F40'},
                text_auto='.2f'
            )
            fig_comp.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#052623'))
            st.plotly_chart(fig_comp, use_container_width=True)

        # --- 3. LIFECYCLE ---
        st.markdown("---")
        st.header("3. Decay Curve")
        life_df = raw_with_birthdays.groupby('age')[numeric_cols].sum().reset_index()
        
        if sel_logic == 'calc_cpa': life_df['y'] = life_df[spend_col] / life_df[installs_col]
        elif sel_logic == 'calc_ipm': life_df['y'] = (life_df[installs_col] / life_df[imps_col]) * 1000
        elif sel_logic == 'calc_ctr': life_df['y'] = (life_df[clicks_col] / life_df[imps_col]) * 100
        elif sel_logic == 'calc_roas': life_df['y'] = life_df[value_col] / life_df[spend_col]
        elif sel_logic == 'calc_cpm': life_df['y'] = (life_df[spend_col] / life_df[imps_col]) * 1000
        
        life_df = life_df[life_df['age'] <= 60]
        
        fig_life = px.line(life_df, x='age', y='y', title=f"{s2_choice} by Day Since Launch", markers=True)
        fig_life.update_traces(line_color='#052623')
        fig_life.add_vline(x=21, line_dash="dash", line_color="#FF7F40")
        fig_life.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#052623'))
        st.plotly_chart(fig_life, use_container_width=True)

        # --- DOWNLOAD ---
        csv = analysis_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download CSV", data=csv, file_name="data.csv", mime="text/csv")

    else:
        st.error("Error: Check your CSV columns.")
else:
    st.info("👈 Upload CSV to start.")
