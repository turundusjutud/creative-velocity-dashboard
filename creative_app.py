import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Turundusjutud | Creative Analytics Dashboard", 
    page_icon="📊",
    layout="wide"
)

# --- CUSTOM BRANDING CSS ---
st.markdown("""
    <style>
        /* 1. Main Layout & Colors */
        .stApp {
            background-color: #FAFAFA;
            color: #052623;
        }
        
        p, div, label, span, li {
            color: #052623;
            font-family: 'Helvetica', 'Arial', sans-serif;
        }
        
        /* 2. Headings */
        h1, h2, h3, h4 {
            color: #052623 !important;
            font-weight: 700;
        }
        
        /* 3. Metric Cards */
        div[data-testid="stMetric"] {
            background-color: #FFFFFF;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            border: 1px solid #E5E7EB;
            border-left: 5px solid #1A776F; /* Brand Teal */
        }
        div[data-testid="stMetric"] label {
            color: #1A776F !important;
        }
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
            color: #052623 !important;
        }

        /* 4. Radio Buttons (Segmented Control) */
        div.row-widget.stRadio > div {
            flex-direction: row;
            gap: 10px;
        }
        div.row-widget.stRadio > div > label {
            background-color: #FFFFFF;
            border: 1px solid #1A776F;
            padding: 5px 15px;
            border-radius: 5px;
            color: #052623;
            cursor: pointer;
        }
        div.row-widget.stRadio > div > label[data-baseweb="radio"] {
            background-color: #1A776F;
            color: white;
        }

        /* 5. Buttons */
        div.stButton > button {
            background-color: #FF7F40;
            color: white !important;
            border-radius: 8px;
            border: none;
            font-weight: bold;
        }
        div.stButton > button:hover {
            background-color: #E66A2E;
            color: white !important;
        }
    </style>
""", unsafe_allow_html=True)

# --- PRIVACY NOTICE ---
st.info("""
    **🔒 Privacy & Data Security:** This tool processes data **locally in your browser session**. 
    No data is saved to any server. Closing this tab deletes all data permanently.
""")

st.title("Creative Strategy Analytics")

# --- INSTRUCTIONS EXPANDER ---
with st.expander("📝 How to export data from Meta Ads (Read this first!)", expanded=False):
    st.markdown("""
    To ensure the analysis works, your CSV must be formatted exactly like this:
    
    1. Go to **Ads Manager** -> **Reports** -> **Export Table Data**.
    2. **Breakdown:** Select **"Day"** (Under Time).
       * ⚠️ **IMPORTANT:** Do NOT select any other breakdowns. The data must be **ungrouped**.
    3. **Level:** Select **"Ad"**.
    4. **Columns:** Ensure you have: `Ad ID`, `Reporting Starts`, `Amount Spent`, `Impressions`, `Link Clicks`, `Purchases` (or your main result).
    5. **Export:** Save as .csv and upload on the left.
    """)

# --- HELPER FUNCTIONS ---
def load_data(file):
    df = pd.read_csv(file)
    df.columns = [c.lower().strip() for c in df.columns]
    
    date_cols = [c for c in df.columns if 'date' in c or 'start' in c]
    if not date_cols: return None, None, None, None, None, None, None, None
    date_col = date_cols[0]
    df[date_col] = pd.to_datetime(df[date_col])
    
    spend_col = next((c for c in df.columns if 'amount' in c or 'spend' in c), None)
    imps_col = next((c for c in df.columns if 'impression' in c), None)
    clicks_col = next((c for c in df.columns if 'link click' in c or 'clicks' in c), None)
    ad_id_col = next((c for c in df.columns if 'ad id' in c), None)
    installs_col = next((c for c in df.columns if 'install' in c), None)
    if not installs_col: installs_col = next((c for c in df.columns if 'result' in c), None)
    value_col = next((c for c in df.columns if 'value' in c or 'revenue' in c), None)
    
    return df, date_col, spend_col, imps_col, installs_col, clicks_col, value_col, ad_id_col

def categorize_age(days):
    if days <= 30: return '1. New (<1 Mo)'
    elif days <= 90: return '2. Recent (1-3 Mo)'
    elif days <= 180: return '3. Mature (3-6 Mo)'
    elif days <= 365: return '4. Vintage (6 Mo - 1 Yr)'
    else: return '5. Legacy (1 Yr+)'

# --- TOOLTIP DICTIONARY ---
tooltips = {
    'win_rate': "Win Rate is the % of new ads that spent more than your 'Meaningful Spend' threshold.",
    'slop': "Creative Slop is the number of ads that failed to launch (spent almost €0).",
    'ipm': "IPM (Installs Per Mille) measures creative 'hook' power. Formula: (Installs / Impressions) * 1000.",
    'cpa': "CPA (Cost Per Action).",
    'cpm': "CPM (Cost Per Mille). Cost of 1,000 impressions.",
    'ctr': "CTR (Click Through Rate).",
    'roas': "ROAS (Return on Ad Spend). Revenue / Spend.",
    'correlation': "Score from -1 to +1. (+1: More ads = Higher Metric). (-1: More ads = Lower Metric)."
}

# --- MAIN APP ---
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=['csv'])

if uploaded_file is not None:
    raw_df, date_col, spend_col, imps_col, installs_col, clicks_col, value_col, ad_id_col = load_data(uploaded_file)
    
    if raw_df is not None and spend_col and ad_id_col:
        st.sidebar.success("✅ Data Loaded Successfully")
        
        # --- SIDEBAR SETTINGS ---
        st.sidebar.markdown("---")
        st.sidebar.header("⚙️ Settings & Guardrails")
        
        min_spend_global = st.sidebar.number_input("Global Min Spend Filter (€)", value=0, help="Exclude rows with 0 spend.")
        
        st.sidebar.subheader("🏆 Win Rate Logic")
        meaningful_spend = st.sidebar.number_input("Threshold for 'Meaningful Spend' (€)", value=50, help="An ad is only considered a 'Player' if it spends at least this much.")
        
        # --- PROCESSING ---
        # 1. Lifetime Stats
        creative_agg = raw_df.groupby(ad_id_col).agg({date_col: 'min', spend_col: 'sum'}).reset_index()
        creative_agg.columns = [ad_id_col, 'launch_date', 'lifetime_spend']
        
        # 2. Filter Global Min Spend
        raw_df = raw_df[raw_df[spend_col] > min_spend_global]
        
        # 3. Age Buckets
        last_date = raw_df[date_col].max()
        creative_agg['age_days'] = (last_date - creative_agg['launch_date']).dt.days
        creative_agg['Age_Bucket'] = creative_agg['age_days'].apply(categorize_age)
        
        # 4. Weekly Data
        raw_df['week_start'] = raw_df[date_col].dt.to_period('W-MON').apply(lambda r: r.start_time)
        numeric_cols = raw_df.select_dtypes(include=[np.number]).columns.tolist()
        weekly_metrics = raw_df.groupby('week_start')[numeric_cols].sum().reset_index()
        
        # 5. New Creative Counts (Velocity)
        creative_agg['launch_week'] = creative_agg['launch_date'].dt.to_period('W-MON').apply(lambda r: r.start_time)
        weekly_new_creatives = creative_agg.groupby('launch_week')[ad_id_col].count().reset_index()
        weekly_new_creatives.columns = ['week_start', 'new_creatives_count']
        
        # 6. Merge Metrics + Velocity
        analysis_df = pd.merge(weekly_metrics, weekly_new_creatives, on='week_start', how='left').fillna(0)
        
        # 7. CLEAN METRIC NAMES (No "Calculated_" prefix)
        if installs_col and imps_col:
            analysis_df['IPM'] = (analysis_df[installs_col] / analysis_df[imps_col]) * 1000
            analysis_df['CPA'] = analysis_df[spend_col] / analysis_df[installs_col]
        
        analysis_df['CPM'] = (analysis_df[spend_col] / analysis_df[imps_col]) * 1000
        analysis_df['CPC'] = analysis_df[spend_col] / analysis_df[clicks_col]
        
        if clicks_col and imps_col:
            analysis_df['CTR'] = (analysis_df[clicks_col] / analysis_df[imps_col]) * 100
        if value_col and spend_col:
            analysis_df['ROAS'] = analysis_df[value_col] / analysis_df[spend_col]

        analysis_df = analysis_df[analysis_df[spend_col] > 0]
        
        # --- CALCULATOR STATS ---
        avg_cpm = analysis_df['CPM'].mean()
        st.sidebar.markdown("---")
        st.sidebar.subheader("🧮 Budget vs. Velocity Calculator")
        user_budget = st.sidebar.number_input("Monthly Budget (€)", value=5000)
        test_imps = st.sidebar.number_input("Impressions to test 1 Ad", value=4000)
        
        cost_to_test_one = (test_imps / 1000) * avg_cpm
        max_tests = user_budget / cost_to_test_one if cost_to_test_one > 0 else 0
        
        st.sidebar.info(f"At your CPM of €{avg_cpm:.2f}, it costs **€{cost_to_test_one:.2f}** to test one ad.\n\n👉 **Capacity:** You can test ~{int(max_tests)} ads/month.")

        # --- SECTION 1: LAUNCH RHYTHM ---
        st.header("1. Creative Pulse & Consistency")
        st.caption("Are we launching consistently, or do we have long periods of inactivity?")

        launch_dates = sorted(creative_agg['launch_date'].unique())
        if len(launch_dates) > 1:
            date_diffs = pd.Series(launch_dates).diff().dt.days.dropna()
            avg_launch_gap = date_diffs.mean()
            max_launch_gap = date_diffs.max()
        else:
            avg_launch_gap = 0
            max_launch_gap = 0

        c1, c2, c3 = st.columns(3)
        c1.metric("Unique Launch Days", len(launch_dates))
        c2.metric("Avg Days Between Launches", f"{avg_launch_gap:.1f} days")
        c3.metric("Longest Drought", f"{max_launch_gap:.0f} days")
        
        if max_launch_gap > 30:
            st.warning(f"⚠️ **Stability Risk:** You went **{max_launch_gap:.0f} days** without launching a single ad.")

        # --- SECTION 2: COMPOSITION CHART ---
        st.markdown("---")
        st.subheader("Spend Composition: Fresh vs. Fatigued")
        st.caption("Green = Spend on ads <21 days old. Red = Spend on ads >21 days old. Black Line = Active Fresh Ads.")

        raw_w_launch = pd.merge(raw_df, creative_agg[[ad_id_col, 'launch_date']], on=ad_id_col, how='left')
        raw_w_launch['spend_age_days'] = (raw_w_launch[date_col] - raw_w_launch['launch_date']).dt.days
        raw_w_launch['Freshness'] = raw_w_launch['spend_age_days'].apply(lambda x: 'Fresh (<21d)' if x < 21 else 'Fatigued (>21d)')
        
        comp_df = raw_w_launch.groupby(['week_start', 'Freshness'])[spend_col].sum().reset_index()
        fresh_only = raw_w_launch[raw_w_launch['Freshness'] == 'Fresh (<21d)']
        active_fresh_count = fresh_only.groupby('week_start')[ad_id_col].nunique().reset_index()
        
        fig_dual = go.Figure()
        for status, color in [('Fatigued (>21d)', '#FF7F40'), ('Fresh (<21d)', '#1A776F')]:
            subset = comp_df[comp_df['Freshness'] == status]
            fig_dual.add_trace(go.Bar(
                x=subset['week_start'].astype(str),
                y=subset[spend_col],
                name=f"Spend: {status}",
                marker_color=color
            ))

        if not active_fresh_count.empty:
            fig_dual.add_trace(go.Scatter(
                x=active_fresh_count['week_start'].astype(str),
                y=active_fresh_count[ad_id_col],
                name="Active Fresh Ads (Count)",
                yaxis='y2',
                mode='lines+markers',
                line=dict(color='black', width=3)
            ))

        fig_dual.update_layout(
            barmode='stack',
            title='Weekly Spend Composition + Fresh Ad Volume',
            yaxis=dict(title='Spend (€)'),
            yaxis2=dict(title='Count of Active Fresh Ads', overlaying='y', side='right', showgrid=False),
            legend=dict(orientation="h", y=1.1),
            hovermode="x unified"
        )
        st.plotly_chart(fig_dual, use_container_width=True)

        # --- SECTION 3: VELOCITY IMPACT ---
        st.markdown("---")
        st.header("2. How Velocity Impacts Metrics")
        st.caption("How does adding (or not adding) creatives affect key results?")
        
        # Clean metrics list (excluding helper cols)
        available_metrics = [c for c in analysis_df.columns if c in ['IPM', 'CPA', 'CPM', 'CPC', 'CTR', 'ROAS']]
        
        c1, c2 = st.columns([1, 1])
        with c1:
            metric_choice = st.selectbox("Select KPI to Analyze:", available_metrics, index=0)
        with c2:
            st.write("Time Lag (Weeks):")
            lag_weeks = st.radio("Lag", options=[0, 1, 2, 3, 4, 5, 6, 7, 8], horizontal=True, label_visibility="collapsed", index=1)
            st.caption(f"Analyzing how uploads today affect results **{lag_weeks} weeks later**.")

        analysis_df['lagged_uploads'] = analysis_df['new_creatives_count'].shift(lag_weeks)
        valid_data = analysis_df.dropna(subset=['lagged_uploads', metric_choice])

        # Correlation Box
        if len(valid_data) > 2:
            corr = valid_data['lagged_uploads'].corr(valid_data[metric_choice])
            st.metric("Correlation Score", f"{corr:.2f}", help=tooltips['correlation'])
            
            is_good_metric = 'CPA' not in metric_choice and 'CPC' not in metric_choice
            if abs(corr) < 0.25: st.info("⚪ **Neutral:** Velocity has no strong impact on this metric right now.")
            elif corr < -0.3:
                if not is_good_metric: st.success(f"🟢 **Good:** Higher velocity reduces {metric_choice} (Cheaper Results).")
                else: st.warning(f"🔴 **Warning:** Higher velocity is lowering your {metric_choice} (Bad).")
            elif corr > 0.3:
                if is_good_metric: st.success(f"🟢 **Good:** Higher velocity increases {metric_choice} (Better Results).")
                else: st.warning(f"🔴 **Warning:** Higher velocity is increasing costs ({metric_choice}).")

        # THE CHART (Restored)
        fig_vel = go.Figure()
        fig_vel.add_trace(go.Bar(x=valid_data['week_start'], y=valid_data['lagged_uploads'], name='New Creatives', marker_color='rgba(5, 38, 35, 0.2)', yaxis='y'))
        fig_vel.add_trace(go.Scatter(x=valid_data['week_start'], y=valid_data[metric_choice], name=metric_choice, mode='lines+markers', line=dict(color='#1A776F', width=3), yaxis='y2'))
        fig_vel.update_layout(title=f'Velocity vs {metric_choice}', yaxis=dict(title='New Ads'), yaxis2=dict(title=metric_choice, overlaying='y', side='right'))
        st.plotly_chart(fig_vel, use_container_width=True)

        # --- SECTION 4: WIN RATE ---
        st.markdown("---")
        st.header("3. Creative Efficiency & Win Rate")
        st.caption(f"Are we launching 'Slop'? (Ads with < €{meaningful_spend} spend)")
        
        recent_ads = creative_agg[creative_agg['age_days'] <= 90]
        total_recent = len(recent_ads)
        
        if total_recent > 0:
            winners = recent_ads[recent_ads['lifetime_spend'] >= meaningful_spend]
            slop = recent_ads[recent_ads['lifetime_spend'] < meaningful_spend]
            win_rate = (len(winners) / total_recent) * 100
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Ads Launched (Last 90d)", total_recent)
            c2.metric("Win Rate %", f"{win_rate:.1f}%", help=tooltips['win_rate'])
            c3.metric("Creative Slop", len(slop), help=tooltips['slop'])
            
            if win_rate < 10:
                st.warning("⚠️ **Low Win Rate:** <10% of new ads get meaningful spend. Focus on quality.")
            elif win_rate > 50:
                st.success("✅ **High Win Rate:** Your ads are launching successfully.")
        else:
            st.info("No new ads found in last 90 days.")

        # --- SECTION 5: AGE DISTRIBUTION ---
        st.markdown("---")
        st.header("4. Ad Age Distribution")
        st.caption("How old are the ads that are spending your money?")
        
        raw_with_bucket = pd.merge(raw_df, creative_agg[[ad_id_col, 'Age_Bucket']], on=ad_id_col, how='left')
        bucket_spend = raw_with_bucket.groupby('Age_Bucket')[spend_col].sum().reset_index()
        bucket_spend['% Spend'] = (bucket_spend[spend_col] / bucket_spend[spend_col].sum()) * 100
        
        c1, c2 = st.columns([1, 2])
        with c1:
            st.dataframe(bucket_spend.style.format({spend_col: '€{:.0f}', '% Spend': '{:.1f}%'}), use_container_width=True)
        with c2:
            fig_pie = px.pie(bucket_spend, values=spend_col, names='Age_Bucket', title="Share of Spend by Ad Age", color_discrete_sequence=px.colors.sequential.Teal)
            st.plotly_chart(fig_pie, use_container_width=True)

        # --- SECTION 6: FRESH VS FATIGUED ---
        st.markdown("---")
        st.header("5. Performance: Old vs. New")
        
        s5_choice = st.selectbox("Compare Metric:", available_metrics, index=0)
        
        eff_comp = raw_w_launch.groupby('Freshness')[numeric_cols].sum().reset_index()
        
        if s5_choice == 'IPM': eff_comp['val'] = (eff_comp[installs_col] / eff_comp[imps_col]) * 1000
        elif s5_choice == 'CPA': eff_comp['val'] = eff_comp[spend_col] / eff_comp[installs_col]
        elif s5_choice == 'CTR': eff_comp['val'] = (eff_comp[clicks_col] / eff_comp[imps_col]) * 100
        elif s5_choice == 'CPC': eff_comp['val'] = eff_comp[spend_col] / eff_comp[clicks_col]
        elif s5_choice == 'CPM': eff_comp['val'] = (eff_comp[spend_col] / eff_comp[imps_col]) * 1000
        elif s5_choice == 'ROAS': eff_comp['val'] = eff_comp[value_col] / eff_comp[spend_col]
        else: eff_comp['val'] = 0

        fig_comp = px.bar(
            eff_comp, x='Freshness', y='val', color='Freshness', 
            title=f"{s5_choice} Comparison",
            color_discrete_map={'Fresh (<21d)': '#1A776F', 'Fatigued (>21d)': '#FF7F40'},
            text_auto='.2f'
        )
        st.plotly_chart(fig_comp, use_container_width=True)

        # --- SECTION 7: DECAY CURVE ---
        st.markdown("---")
        st.header("6. The Decay Curve")
        
        raw_w_launch['absolute_age'] = (raw_w_launch[date_col] - raw_w_launch['launch_date']).dt.days
        life_df = raw_w_launch.groupby('absolute_age')[numeric_cols].sum().reset_index()
        
        if s5_choice == 'IPM': life_df['y'] = (life_df[installs_col] / life_df[imps_col]) * 1000
        elif s5_choice == 'CPA': life_df['y'] = life_df[spend_col] / life_df[installs_col]
        elif s5_choice == 'CTR': life_df['y'] = (life_df[clicks_col] / life_df[imps_col]) * 100
        elif s5_choice == 'CPC': life_df['y'] = life_df[spend_col] / life_df[clicks_col]
        elif s5_choice == 'CPM': life_df['y'] = (life_df[spend_col] / life_df[imps_col]) * 1000
        elif s5_choice == 'ROAS': life_df['y'] = life_df[value_col] / life_df[spend_col]
        
        life_df = life_df[life_df['absolute_age'] <= 60]
        
        fig_life = px.line(
            life_df, x='absolute_age', y='y', title=f"{s5_choice} by Day Since Launch", markers=True,
            labels={"absolute_age": "Days Since Launch", "y": s5_choice}
        )
        fig_life.add_vline(x=21, line_dash="dash", line_color="#FF7F40", annotation_text="Fatigue (21d)")
        fig_life.update_traces(line_color='#052623')
        st.plotly_chart(fig_life, use_container_width=True)

        # --- GLOSSARY ---
        with st.expander("📚 Metric Glossary"):
            st.markdown(f"""
            * **Win Rate:** {tooltips['win_rate']}
            * **IPM:** {tooltips['ipm']}
            * **Creative Slop:** {tooltips['slop']}
            * **Correlation Score:** {tooltips['correlation']}
            """)

    else:
        st.error("Error: Check CSV columns.")
else:
    st.info("👈 Upload CSV to start.")
