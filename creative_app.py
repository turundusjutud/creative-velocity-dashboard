import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(page_title="Creative Strategy Dashboard", layout="wide")

st.title("🎨 Creative Strategy & Velocity Dashboard")
st.markdown("""
**Goal:** Analyze if increasing creative production (Velocity) actually drives business results.
**Instructions:** Upload your Facebook Ads Manager CSV (Day + Ad Level).
""")

# --- HELPER FUNCTIONS ---
def load_data(file):
    df = pd.read_csv(file)
    df.columns = [c.lower().strip() for c in df.columns]
    
    # Auto-detect date
    date_cols = [c for c in df.columns if 'date' in c or 'start' in c]
    if not date_cols:
        st.error("Could not find a 'Date' or 'Reporting Starts' column.")
        return None, None, None, None, None, None
    date_col = date_cols[0]
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Auto-detect key metrics
    spend_col = next((c for c in df.columns if 'amount' in c or 'spend' in c), None)
    imps_col = next((c for c in df.columns if 'impression' in c), None)
    clicks_col = next((c for c in df.columns if 'link click' in c or 'clicks' in c), None)
    ad_id_col = next((c for c in df.columns if 'ad id' in c), None)
    
    # Helper: Detect Install/Result column
    installs_col = next((c for c in df.columns if 'install' in c), None)
    if not installs_col:
        installs_col = next((c for c in df.columns if 'result' in c), None)
        
    # Helper: Detect Purchase Value
    value_col = next((c for c in df.columns if 'value' in c or 'revenue' in c), None)
    
    return df, date_col, spend_col, imps_col, installs_col, clicks_col, value_col, ad_id_col

# --- MAIN APP ---
uploaded_file = st.sidebar.file_uploader("Upload FB Ads CSV", type=['csv'])

if uploaded_file is not None:
    # Load Data
    raw_df, date_col, spend_col, imps_col, installs_col, clicks_col, value_col, ad_id_col = load_data(uploaded_file)
    
    if raw_df is not None and spend_col and ad_id_col:
        st.sidebar.success("✅ Data Processed Successfully")
        
        # --- SIDEBAR CONTROLS ---
        st.sidebar.markdown("---")
        st.sidebar.header("⚙️ Data Filters")
        min_spend = st.sidebar.number_input("Min Spend per Creative (Lifetime)", value=0, help="Exclude creatives that spent less than this total amount.")
        
        # --- PRE-PROCESSING ---
        # 1. Identify Creative Birthdays & Lifetime Spend
        creative_agg = raw_df.groupby(ad_id_col).agg({date_col: 'min', spend_col: 'sum'}).reset_index()
        creative_agg.columns = [ad_id_col, 'launch_date', 'lifetime_spend']
        
        # FILTER: Exclude low spenders
        valid_ads = creative_agg[creative_agg['lifetime_spend'] >= min_spend][ad_id_col]
        raw_df = raw_df[raw_df[ad_id_col].isin(valid_ads)]
        
        # Re-merge birthdays after filter
        creative_birthdays = creative_agg[creative_agg[ad_id_col].isin(valid_ads)][[ad_id_col, 'launch_date']]
        
        # 2. Weekly Aggregation (for Timeline)
        raw_df['week_start'] = raw_df[date_col].dt.to_period('W-MON').apply(lambda r: r.start_time)
        numeric_cols = raw_df.select_dtypes(include=[np.number]).columns.tolist()
        weekly_metrics = raw_df.groupby('week_start')[numeric_cols].sum().reset_index()
        
        # Count New Creatives per Week
        creative_birthdays['launch_week'] = creative_birthdays['launch_date'].dt.to_period('W-MON').apply(lambda r: r.start_time)
        weekly_new_creatives = creative_birthdays.groupby('launch_week')[ad_id_col].count().reset_index()
        weekly_new_creatives.columns = ['week_start', 'new_creatives_count']
        
        # Merge
        analysis_df = pd.merge(weekly_metrics, weekly_new_creatives, on='week_start', how='left').fillna(0)
        
        # 3. Calculate Derived Metrics for Timeline
        if installs_col and imps_col:
            analysis_df['Calculated_IPM'] = (analysis_df[installs_col] / analysis_df[imps_col]) * 1000
            analysis_df['Calculated_CPA'] = analysis_df[spend_col] / analysis_df[installs_col]
        
        analysis_df['Calculated_CPM'] = (analysis_df[spend_col] / analysis_df[imps_col]) * 1000
        
        if clicks_col and imps_col:
            analysis_df['Calculated_CTR'] = (analysis_df[clicks_col] / analysis_df[imps_col]) * 100
        if value_col and spend_col:
            analysis_df['Calculated_ROAS'] = analysis_df[value_col] / analysis_df[spend_col]

        analysis_df = analysis_df[analysis_df[spend_col] > 0]

        # ==========================================
        # SECTION 1: VELOCITY & CORRELATION
        # ==========================================
        st.header("1. Does Velocity Impact Performance?")
        st.caption("Check if launching more ads correlates with better metrics.")

        available_metrics = list(analysis_df.select_dtypes(include=[np.number]).columns)
        available_metrics = [m for m in available_metrics if 'id' not in m and 'week' not in m]
        
        default_ix = 0
        if 'Calculated_CPA' in available_metrics:
            default_ix = available_metrics.index('Calculated_CPA')
        
        col_sel, col_lag = st.columns([2, 1])
        with col_sel:
            metric_choice = st.selectbox("Select Business KPI to analyze (Timeline):", available_metrics, index=default_ix)
        with col_lag:
            lag_weeks = st.slider("Time Lag (Weeks):", 0, 8, 0, help="Shift 'New Uploads' forward.")

        analysis_df['lagged_uploads'] = analysis_df['new_creatives_count'].shift(lag_weeks)
        valid_data = analysis_df.dropna(subset=['lagged_uploads', metric_choice])

        if len(valid_data) > 2:
            correlation = valid_data['lagged_uploads'].corr(valid_data[metric_choice])
            col_metric, col_text = st.columns([1, 3])
            with col_metric:
                st.metric(label="Correlation Coefficient", value=f"{correlation:.3f}")
            with col_text:
                if abs(correlation) < 0.25:
                    st.info("⚪ **Neutral:** No clear correlation found.")
                elif correlation < -0.3:
                    st.success(f"🟢 **Negative Correlation:** As Uploads increase, {metric_choice} tends to **DECREASE**.")
                elif correlation > 0.3:
                    st.info(f"🔵 **Positive Correlation:** As Uploads increase, {metric_choice} tends to **INCREASE**.")
        
        # Dual Axis Chart
        fig = go.Figure()
        fig.add_trace(go.Bar(x=valid_data['week_start'], y=valid_data['lagged_uploads'], name='New Creatives', marker_color='rgba(200, 200, 200, 0.5)', yaxis='y'))
        fig.add_trace(go.Scatter(x=valid_data['week_start'], y=valid_data[metric_choice], name=metric_choice, mode='lines+markers', line=dict(color='#2E86C1', width=3), yaxis='y2'))
        fig.update_layout(title=f'Timeline: Creative Velocity vs {metric_choice}', xaxis=dict(title='Week'), yaxis=dict(title='New Creatives', side='left', showgrid=False), yaxis2=dict(title=metric_choice, side='right', overlaying='y', showgrid=False), legend=dict(x=0, y=1.1, orientation='h'))
        st.plotly_chart(fig, use_container_width=True)

        # ==========================================
        # SECTION 2: FRESH VS FATIGUED
        # ==========================================
        st.markdown("---")
        st.header("2. Fresh vs. Fatigued Performance")
        st.caption("Compare how 'Fresh' ads (< 21 days old) perform vs 'Fatigued' ads.")

        raw_with_birthdays = pd.merge(raw_df, creative_birthdays, on=ad_id_col, how='left')
        raw_with_birthdays['ad_age_days'] = (raw_with_birthdays[date_col] - raw_with_birthdays['launch_date']).dt.days
        raw_with_birthdays['Status'] = raw_with_birthdays['ad_age_days'].apply(lambda x: 'Fresh (<21 Days)' if x < 21 else 'Fatigued (21+ Days)')

        # Metric Logic
        metric_options = {}
        if installs_col and imps_col: metric_options['IPM'] = 'calc_ipm'; metric_options['CPA'] = 'calc_cpa'
        if clicks_col and imps_col: metric_options['CTR'] = 'calc_ctr'; metric_options['CPC'] = 'calc_cpc'
        if value_col and spend_col: metric_options['ROAS'] = 'calc_roas'
        metric_options['CPM'] = 'calc_cpm'
        for col in numeric_cols:
            if col not in [ad_id_col, 'ad_age_days']: metric_options[f"Total {col.title()}"] = col

        s2_choice = st.selectbox("Select Metric to Compare:", list(metric_options.keys()), index=0)
        
        eff_comp = raw_with_birthdays.groupby('Status')[numeric_cols].sum().reset_index()
        selected_logic = metric_options[s2_choice]
        plot_col = "Value"
        is_lower_better = False

        # Calc Logic
        if selected_logic == 'calc_cpa': eff_comp[plot_col] = eff_comp[spend_col] / eff_comp[installs_col]; is_lower_better = True
        elif selected_logic == 'calc_ipm': eff_comp[plot_col] = (eff_comp[installs_col] / eff_comp[imps_col]) * 1000
        elif selected_logic == 'calc_ctr': eff_comp[plot_col] = (eff_comp[clicks_col] / eff_comp[imps_col]) * 100
        elif selected_logic == 'calc_cpc': eff_comp[plot_col] = eff_comp[spend_col] / eff_comp[clicks_col]; is_lower_better = True
        elif selected_logic == 'calc_cpm': eff_comp[plot_col] = (eff_comp[spend_col] / eff_comp[imps_col]) * 1000; is_lower_better = True
        elif selected_logic == 'calc_roas': eff_comp[plot_col] = eff_comp[value_col] / eff_comp[spend_col]
        else: eff_comp[plot_col] = eff_comp[selected_logic]; is_lower_better = True if 'spend' in s2_choice.lower() else False

        if len(eff_comp) >= 2:
            fresh_val = eff_comp.loc[eff_comp['Status'].str.contains('Fresh'), plot_col].values[0]
            old_val = eff_comp.loc[eff_comp['Status'].str.contains('Fatigued'), plot_col].values[0]
            if old_val == 0: old_val = 0.0001
            diff_pct = ((fresh_val - old_val) / old_val) * 100
            
            delta_color = "normal" if (is_lower_better and diff_pct < 0) or (not is_lower_better and diff_pct > 0) else "inverse"
            c1, c2 = st.columns(2)
            with c1: st.metric(f"Fresh Ads: {s2_choice}", f"{fresh_val:,.2f}", f"{diff_pct:.1f}%", delta_color=delta_color)
            with c2: st.metric("Spend on Fresh Ads", f"${eff_comp.loc[eff_comp['Status'].str.contains('Fresh'), spend_col].values[0]:,.0f}")
            
            color_map = {'Fresh (<21 Days)': '#2ECC71' if delta_color=="normal" else '#E74C3C', 'Fatigued (21+ Days)': '#E74C3C' if delta_color=="normal" else '#2ECC71'}
            fig_comp = px.bar(eff_comp, x='Status', y=plot_col, title=f"Comparison: {s2_choice}", color='Status', color_discrete_map=color_map, text_auto='.2f')
            st.plotly_chart(fig_comp, use_container_width=True)

        # ==========================================
        # SECTION 3: LIFECYCLE DECAY
        # ==========================================
        st.markdown("---")
        st.header("3. The 'Death Curve' (Lifecycle Analysis)")
        st.caption("How does performance degrade day-by-day? Use this to find the exact day ads usually fatigue.")

        # Aggregate by 'Day Since Launch'
        lifecycle_df = raw_with_birthdays.groupby('ad_age_days')[numeric_cols].sum().reset_index()
        
        # Recalculate Metric for the Lifecycle Curve
        if selected_logic == 'calc_cpa': lifecycle_df['y_axis'] = lifecycle_df[spend_col] / lifecycle_df[installs_col]
        elif selected_logic == 'calc_ipm': lifecycle_df['y_axis'] = (lifecycle_df[installs_col] / lifecycle_df[imps_col]) * 1000
        elif selected_logic == 'calc_ctr': lifecycle_df['y_axis'] = (lifecycle_df[clicks_col] / lifecycle_df[imps_col]) * 100
        elif selected_logic == 'calc_roas': lifecycle_df['y_axis'] = lifecycle_df[value_col] / lifecycle_df[spend_col]
        elif selected_logic == 'calc_cpm': lifecycle_df['y_axis'] = (lifecycle_df[spend_col] / lifecycle_df[imps_col]) * 1000
        else: lifecycle_df['y_axis'] = lifecycle_df[selected_logic]

        # Filter to first 60 days to keep chart readable
        lifecycle_df = lifecycle_df[lifecycle_df['ad_age_days'] <= 60]
        lifecycle_df = lifecycle_df[lifecycle_df['ad_age_days'] >= 0]

        fig_life = px.line(lifecycle_df, x='ad_age_days', y='y_axis', title=f"Average {s2_choice} by Days Since Launch", markers=True)
        fig_life.add_vline(x=21, line_dash="dash", line_color="red", annotation_text="Fatigue Threshold (21 Days)")
        fig_life.update_layout(xaxis_title="Days Since Launch", yaxis_title=s2_choice)
        st.plotly_chart(fig_life, use_container_width=True)

        # ==========================================
        # SECTION 4: VINTAGE
        # ==========================================
        st.markdown("---")
        st.header("4. Account Decay (Vintage)")
        raw_with_birthdays['vintage_month'] = raw_with_birthdays['launch_date'].dt.to_period('M').astype(str)
        vintage_analysis = raw_with_birthdays.groupby(['week_start', 'vintage_month'])[spend_col].sum().reset_index().sort_values('vintage_month')
        fig_vintage = px.area(vintage_analysis, x='week_start', y=spend_col, color='vintage_month', title="Spend by Launch Month", labels={spend_col: 'Spend'})
        st.plotly_chart(fig_vintage, use_container_width=True)

        # ==========================================
        # SECTION 5: METHODOLOGY (For MarTech)
        # ==========================================
        st.markdown("---")
        with st.expander("📊 Methodology & Calculation Formulas"):
            st.markdown("""
            **1. Data Filtering:**
            * **Min Spend:** Creatives with lifetime spend below the sidebar threshold are excluded to remove statistical noise.
            
            **2. Creative Launch Date:** * Defined as the **earliest date** an `Ad ID` appeared in the uploaded dataset. 

            **3. Fresh vs. Fatigued Status:**
            * **Fresh:** Ad Age (Current Date - Launch Date) is **< 21 Days**.
            * **Fatigued:** Ad Age is **21 Days or older**.
            
            **4. Calculated Metrics (Weighted Averages):**
            * **IPM (Installs Per Mille):** `(Total Installs / Total Impressions) * 1000`
            * **CPA (Cost Per Action):** `Total Spend / Total Installs`
            * **CTR (Click-Through Rate):** `(Total Clicks / Total Impressions) * 100`
            * **ROAS (Return on Ad Spend):** `Total Purchase Value / Total Spend`
            * *Note: We sum the numerators and denominators first, then divide. We do NOT average the daily CPAs, as that would be mathematically incorrect.*

            **5. Correlation Coefficient:**
            * Uses **Pearson Correlation** ($r$) between `Weekly New Creatives` and the selected KPI.
            * If a **Time Lag** is selected, the `New Creatives` column is shifted forward by $N$ weeks before the correlation is calculated.
            """)

        # --- DATA DOWNLOAD ---
        st.markdown("---")
        csv = analysis_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Processed Data (CSV)", data=csv, file_name="creative_velocity_analysis.csv", mime="text/csv")

    else:
        st.error("Error processing file. Please ensure you have 'Amount Spent' and 'Ad ID' columns.")
else:
    st.info("👈 Waiting for file upload...")
