import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Creative Analytics Dashboard", 
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

        /* 4. Buttons */
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
        
        /* 5. Summary Box & Insight Box */
        .summary-box {
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .good-job {
            background-color: #F0FDFA;
            border: 1px solid #1A776F;
        }
        .bad-job {
            background-color: #FFF7ED;
            border: 1px solid #FF7F40;
        }
        .insight-box {
            background-color: #E6FFFA;
            border-left: 4px solid #1A776F;
            padding: 15px;
            margin-top: 10px;
            margin-bottom: 10px;
            border-radius: 4px;
            font-size: 1.05em;
        }
        
        /* 6. Table Centering */
        .dataframe {
            text-align: center !important;
        }
        th {
            text-align: center !important;
        }
    </style>
""", unsafe_allow_html=True)

# --- GLOBAL TOOLTIPS ---
tooltips = {
    'launch_days': "Total number of unique days where at least one new ad was launched.",
    'avg_gap': "The average number of days between two creative launches.",
    'drought': "The longest single period of time where NO new ads were launched.",
    'winning_creatives': "Number of ads that graduated from 'Testing' (Spent > Threshold).",
    'slop': "Number of ads that failed to launch or barely spent (Spent < Threshold). Wasted production effort.",
    'ipm': "Installs Per Mille. A measure of creative 'hook' power. (Installs / Impressions * 1000).",
    'cpa': "Cost Per Action. Spend / Conversions.",
    'cpm': "Cost Per Mille. Cost of 1,000 impressions.",
    'ctr': "Click Through Rate.",
    'cpc': "Cost Per Click.",
    'roas': "Return on Ad Spend.",
    'correlation': "Score from -1 to +1. \n+1: Velocity Improves Metric. \n-1: Velocity Hurts Metric.",
    'lifespan': "The number of days between an ad's first impression and its last impression.",
    'retention': "The percentage of ads that are still running X days after they were launched."
}

# --- PRIVACY NOTICE ---
st.info("""
    **🔒 Privacy & Data Security:** This tool processes data **locally in your browser session**. 
    No data is saved to any server. Closing this tab deletes all data permanently.
""")

st.title("Creative Strategy Analytics")

# --- INSTRUCTIONS EXPANDER ---
with st.expander("📝 How to export data from Meta Ads (Read this first!)", expanded=False):
    st.markdown("""
    1. Go to **Ads Manager** -> **Reports** -> **Export Table Data**.
    2. **Breakdown:** Select **"Day"** (Under Time).
    3. **Level:** Select **"Ad"**.
    4. **Columns:** Ad ID, Reporting Starts, Amount Spent, Impressions, Link Clicks + All KPIs.
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
    
    exclude = [date_col, spend_col, imps_col, clicks_col, ad_id_col, installs_col, value_col, 'reporting starts', 'reporting ends', 'ad name', 'ad set name', 'campaign name']
    numeric_candidates = df.select_dtypes(include=[np.number]).columns.tolist()
    extra_metrics = [c for c in numeric_candidates if c not in exclude and 'id' not in c and 'date' not in c]

    # Determine Main Conversion Name for Text Logic
    conversion_name = "Action"
    if installs_col: conversion_name = "Install"
    elif value_col: conversion_name = "Purchase"
    elif len(extra_metrics) > 0: conversion_name = extra_metrics[0].replace('_', ' ').title()

    return df, date_col, spend_col, imps_col, installs_col, clicks_col, value_col, ad_id_col, extra_metrics, conversion_name

def categorize_age(days):
    if days <= 30: return '1. New (<1 Mo)'
    elif days <= 90: return '2. Recent (1-3 Mo)'
    elif days <= 180: return '3. Mature (3-6 Mo)'
    elif days <= 365: return '4. Vintage (6 Mo - 1 Yr)'
    else: return '5. Legacy (1 Yr+)'

def color_delta(val, metric_name):
    if pd.isna(val): return ''
    lower_is_better = any(x in metric_name.upper() for x in ['CPA', 'CPC', 'CPM', 'COST', 'CP_'])
    color = '#1A776F' if (val < 0 and lower_is_better) or (val > 0 and not lower_is_better) else '#FF7F40'
    return f'color: {color}; font-weight: bold;'

# --- MAIN APP ---
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=['csv'])

if uploaded_file is not None:
    raw_df, date_col, spend_col, imps_col, installs_col, clicks_col, value_col, ad_id_col, extra_metrics, main_conv_name = load_data(uploaded_file)
    
    if raw_df is not None and spend_col and ad_id_col:
        st.sidebar.success("✅ Data Loaded Successfully")
        
        # --- SIDEBAR SETTINGS ---
        st.sidebar.markdown("---")
        st.sidebar.header("⚙️ Settings & Guardrails")
        min_spend_global = st.sidebar.number_input("Global Min Spend Filter (€)", value=0)
        meaningful_spend = st.sidebar.number_input("Meaningful Spend Threshold (€)", value=50)
        
        # --- PROCESSING ---
        numeric_cols_all = raw_df.select_dtypes(include=[np.number]).columns.tolist()
        if ad_id_col in numeric_cols_all: numeric_cols_all.remove(ad_id_col)
            
        creative_agg = raw_df.groupby(ad_id_col).agg({date_col: ['min', 'max'], **{c:'sum' for c in numeric_cols_all}}).reset_index()
        creative_agg.columns = [col[0] if col[1] == '' else f"{col[0]}_{col[1]}" if col[0] == date_col else col[0] for col in creative_agg.columns]
        creative_agg.rename(columns={f'{date_col}_min': 'launch_date', f'{date_col}_max': 'last_date', spend_col: 'lifetime_spend'}, inplace=True)
        creative_agg['lifespan_days'] = (creative_agg['last_date'] - creative_agg['launch_date']).dt.days + 1
        
        raw_df = raw_df[raw_df[spend_col] > min_spend_global]
        raw_df['week_start'] = raw_df[date_col].dt.to_period('W-MON').apply(lambda r: r.start_time)
        weekly_metrics = raw_df.groupby('week_start')[numeric_cols_all].sum().reset_index()
        
        creative_agg['launch_week'] = creative_agg['launch_date'].dt.to_period('W-MON').apply(lambda r: r.start_time)
        weekly_new_creatives = creative_agg.groupby('launch_week')[ad_id_col].count().reset_index()
        weekly_new_creatives.columns = ['week_start', 'new_creatives_count']
        
        analysis_df = pd.merge(weekly_metrics, weekly_new_creatives, on='week_start', how='left').fillna(0)
        
        # --- BUILD METRICS ---
        analysis_df['CPM'] = (analysis_df[spend_col] / analysis_df[imps_col]) * 1000
        analysis_df['CPC'] = analysis_df[spend_col] / analysis_df[clicks_col]
        if clicks_col and imps_col: analysis_df['CTR'] = (analysis_df[clicks_col] / analysis_df[imps_col]) * 100
        available_metrics = ['CPM', 'CPC', 'CTR']
        if installs_col:
            analysis_df['CPA'] = analysis_df[spend_col] / analysis_df[installs_col]
            analysis_df['IPM'] = (analysis_df[installs_col] / analysis_df[imps_col]) * 1000
            available_metrics.extend(['CPA', 'IPM'])
        if value_col:
            analysis_df['ROAS'] = analysis_df[value_col] / analysis_df[spend_col]
            available_metrics.append('ROAS')
        for m in extra_metrics:
            clean = m.replace('_', ' ').title()
            analysis_df[clean] = analysis_df[m]
            available_metrics.append(clean)
            cp_name = f"Cost Per {clean}"
            analysis_df[cp_name] = analysis_df.apply(lambda r: r[spend_col] / r[m] if r[m] > 0 else 0, axis=1)
            available_metrics.append(cp_name)

        analysis_df = analysis_df[analysis_df[spend_col] > 0]
        avg_cpm = analysis_df['CPM'].mean()

        # --- SIDEBAR CALCULATOR ---
        st.sidebar.markdown("---")
        st.sidebar.subheader("🧮 Budget vs. Velocity Calculator")
        user_budget = st.sidebar.number_input("Monthly Budget (€)", value=5000)
        test_imps = st.sidebar.number_input("Impressions to test 1 Ad", value=4000)
        cost_to_test_one = (test_imps / 1000) * avg_cpm
        max_tests = user_budget / cost_to_test_one if cost_to_test_one > 0 else 0
        st.sidebar.info(f"At your CPM of €{avg_cpm:.2f}, it costs **€{cost_to_test_one:.2f}** to test one ad.\n\n👉 **Capacity:** You can test ~{int(max_tests)} ads/month.")

        # --- 1. RHYTHM ---
        st.header("1. Creative Pulse & Consistency")
        launch_dates = sorted(creative_agg['launch_date'].unique())
        drought_start, drought_end, max_launch_gap = None, None, 0
        if len(launch_dates) > 1:
            date_diffs = pd.Series(launch_dates).diff().dt.days.dropna()
            avg_launch_gap = date_diffs.mean()
            diff_values = date_diffs.values
            max_idx = np.argmax(diff_values)
            max_launch_gap = diff_values[max_idx]
            drought_end = launch_dates[max_idx + 1]
            drought_start = launch_dates[max_idx]
        else: avg_launch_gap = 0

        c1, c2, c3 = st.columns(3)
        c1.metric("Unique Launch Days", len(launch_dates), help=tooltips['launch_days'])
        c2.metric("Avg Days Between Launches", f"{avg_launch_gap:.1f} days", help=tooltips['avg_gap'])
        c3.metric("Longest Drought", f"{max_launch_gap:.0f} days", help=tooltips['drought'])
        
        # New "Always On" Analyst Note for Rhythm
        txt_rhythm = f"💡 <b>Analyst Note:</b> Your average launch gap is <b>{avg_launch_gap:.1f} days</b>."
        if avg_launch_gap <= 7:
            txt_rhythm += " 🏆 <b>Elite Consistency:</b> You are launching weekly. This is the #1 driver of account growth."
        elif avg_launch_gap <= 14:
            txt_rhythm += " ✅ <b>Good Rhythm:</b> Launching every two weeks is a healthy, sustainable cadence."
        else:
            txt_rhythm += " ⚠️ <b>Inconsistent:</b> Irregular testing gaps make performance unpredictable. Try to launch at least every 14 days."
        
        if max_launch_gap > 30:
            st.warning(f"⚠️ Stability Risk: You went {max_launch_gap:.0f} days without launching a single ad. \n\n**Why this matters:** Ad performance naturally decays over time. If you pause testing for too long, you have no new winners to replace the dying ones, leading to a sudden performance crash. \n\n**Target State:** Aim to launch at least one new test batch every 7 days.")
        
        st.markdown(f"<div class='insight-box'>{txt_rhythm}</div>", unsafe_allow_html=True)

        # --- 2. COMPOSITION ---
        st.markdown("---")
        st.header("2. Spend Composition: Fresh vs. Fatigued")
        raw_w_launch = pd.merge(raw_df, creative_agg[[ad_id_col, 'launch_date']], on=ad_id_col, how='left')
        raw_w_launch['spend_age_days'] = (raw_w_launch[date_col] - raw_w_launch['launch_date']).dt.days
        raw_w_launch['Freshness'] = raw_w_launch['spend_age_days'].apply(lambda x: 'Fresh (<21d)' if x < 21 else 'Fatigued (>21d)')
        
        comp_df = raw_w_launch.groupby(['week_start', 'Freshness'])[spend_col].sum().reset_index()
        fresh_only = raw_w_launch[raw_w_launch['Freshness'] == 'Fresh (<21d)']
        
        active_fresh_count = fresh_only.groupby('week_start')[ad_id_col].nunique().reset_index()
        all_weeks = pd.DataFrame({'week_start': comp_df['week_start'].unique()})
        active_fresh_count = pd.merge(all_weeks, active_fresh_count, on='week_start', how='left').fillna(0).sort_values('week_start')

        fig_dual = go.Figure()
        for status, color in [('Fatigued (>21d)', '#FF7F40'), ('Fresh (<21d)', '#1A776F')]:
            subset = comp_df[comp_df['Freshness'] == status]
            fig_dual.add_trace(go.Bar(x=subset['week_start'].astype(str), y=subset[spend_col], name=f"Spend: {status}", marker_color=color))
        fig_dual.add_trace(go.Scatter(x=active_fresh_count['week_start'].astype(str), y=active_fresh_count[ad_id_col], name="Active Fresh Ads (Count)", yaxis='y2', mode='lines+markers', line=dict(color='black', width=3)))
        if drought_start and drought_end and max_launch_gap > 14:
            fig_dual.add_vrect(x0=drought_start, x1=drought_end, fillcolor="red", opacity=0.15, layer="below", line_width=0, annotation_text="Longest Drought", annotation_position="top left")
        fig_dual.update_layout(barmode='stack', title='Weekly Spend Composition + Fresh Ad Volume', yaxis=dict(title='Spend (€)'), yaxis2=dict(title='Count of Active Fresh Ads', overlaying='y', side='right', showgrid=False), legend=dict(orientation="h", y=1.1), hovermode="x unified")
        st.plotly_chart(fig_dual, config={'displayModeBar': False, 'responsive': True})

        fresh_spend_share = raw_w_launch[raw_w_launch['Freshness'] == 'Fresh (<21d)'][spend_col].sum() / raw_w_launch[spend_col].sum() * 100
        drought_msg = ""
        if drought_start and drought_end and max_launch_gap > 14:
            mask_during = (raw_df[date_col] >= drought_start) & (raw_df[date_col] < drought_end)
            mask_after = (raw_df[date_col] >= drought_end) & (raw_df[date_col] <= (drought_end + pd.Timedelta(days=14)))
            data_during, data_after = raw_df[mask_during], raw_df[mask_after]
            if not data_during.empty and not data_after.empty:
                metric_to_check = f"Cost Per {main_conv_name}"
                def safe_calc(d, m):
                    if m == 'CPA' and d[installs_col].sum() > 0: return d[spend_col].sum() / d[installs_col].sum()
                    if m == 'CPM' and d[imps_col].sum() > 0: return (d[spend_col].sum() / d[imps_col].sum()) * 1000
                    return 0
                
                # Manual Check based on available columns
                def get_cost_per_main(d):
                    convs = 0
                    if installs_col: convs = d[installs_col].sum()
                    elif value_col: convs = d[value_col].sum() > 0
                    elif len(extra_metrics) > 0: convs = d[extra_metrics[0]].sum()
                    if convs > 0: return d[spend_col].sum() / convs
                    return 0

                val_d = get_cost_per_main(data_during)
                val_a = get_cost_per_main(data_after)
                
                if val_d > 0:
                    diff_pct = ((val_a - val_d) / val_d) * 100
                    direction = "improved (dropped)" if diff_pct < 0 else "worsened (increased)"
                    drought_msg = f"<br>⚠️ <b>Drought Analysis:</b> During the drought ({drought_start.strftime('%b %d')} - {drought_end.strftime('%b %d')}), your <b>{metric_to_check}</b> was <b>{val_d:.2f}</b>. After launches resumed, it <b>{direction} by {abs(diff_pct):.1f}%</b> to <b>{val_a:.2f}</b>."
        
        st.markdown(f"<div class='insight-box'>💡 <b>Analyst Note:</b> Over the selected period, <b>{fresh_spend_share:.1f}%</b> of your total spend went to Fresh ads. {drought_msg}</div>", unsafe_allow_html=True)

        # --- 3. VELOCITY ---
        st.markdown("---")
        st.header("3. How Velocity Impacts Metrics")
        c1, c2 = st.columns([2, 1])
        with c1: st.caption("This matrix ranks which metrics are most sensitive to new launches. Green = Launching helps.")
        with c2: lag_weeks = st.radio("Lag", options=[0, 1, 2, 3, 4, 5, 6, 7, 8], horizontal=True, label_visibility="collapsed", index=1, key="vel_lag")

        corr_data = []
        for m in available_metrics:
            tmp = analysis_df.dropna(subset=['new_creatives_count', m]).copy()
            tmp['lag'] = tmp['new_creatives_count'].shift(lag_weeks)
            tmp = tmp.dropna()
            if len(tmp) > 2:
                corr = tmp['lag'].corr(tmp[m])
                lower_is_better = any(x in m.upper() for x in ['CPA', 'CPC', 'CPM', 'COST', 'CP_'])
                impact = "Good" if (corr < 0 and lower_is_better) or (corr > 0 and not lower_is_better) else "Bad"
                corr_data.append({'Metric': m, 'Correlation': corr, 'Impact': impact})
        
        if corr_data:
            corr_df = pd.DataFrame(corr_data).sort_values('Correlation', ascending=False)
            fig_corr = px.bar(corr_df, x='Correlation', y='Metric', color='Impact', title="Velocity Impact Matrix", color_discrete_map={'Good': '#1A776F', 'Bad': '#FF7F40'}, orientation='h')
            fig_corr.add_vline(x=0, line_width=1, line_color="black")
            st.plotly_chart(fig_corr, config={'displayModeBar': False, 'responsive': True})
            
            # New "Always On" Velocity Insight
            best_corr = corr_df[corr_df['Impact'] == 'Good'].iloc[0] if not corr_df[corr_df['Impact'] == 'Good'].empty else None
            txt_vel = "💡 <b>Analyst Note:</b> "
            if best_corr is not None and abs(best_corr['Correlation']) > 0.3:
                txt_vel += f"Velocity is a strong lever for <b>{best_corr['Metric']}</b> (Corr: {best_corr['Correlation']:.2f}). Launching more ads tends to improve it."
            else:
                txt_vel += "⚖️ <b>Velocity Neutral:</b> Increasing launch volume doesn't drastically swing your metrics right now. Your account is stable."
            st.markdown(f"<div class='insight-box'>{txt_vel}</div>", unsafe_allow_html=True)
        
        st.subheader("Deep Dive")
        vel_metric_choice = st.selectbox("Select Metric to Visualize:", available_metrics, index=0)
        valid_vel = analysis_df.dropna(subset=['new_creatives_count', vel_metric_choice]).copy()
        valid_vel['lag'] = valid_vel['new_creatives_count'].shift(lag_weeks)
        valid_vel = valid_vel.dropna()
        fig_v = go.Figure()
        fig_v.add_trace(go.Bar(x=valid_vel['week_start'], y=valid_vel['lag'], name='New Creatives', marker_color='rgba(5, 38, 35, 0.2)', yaxis='y'))
        fig_v.add_trace(go.Scatter(x=valid_vel['week_start'], y=valid_vel[vel_metric_choice], name=vel_metric_choice, mode='lines+markers', line=dict(color='#1A776F', width=3), yaxis='y2'))
        fig_v.update_layout(title=f'Velocity vs {vel_metric_choice}', yaxis=dict(title='New Ads'), yaxis2=dict(title=vel_metric_choice, overlaying='y', side='right'))
        st.plotly_chart(fig_v, config={'displayModeBar': False, 'responsive': True})

        # --- 4. COST OF INACTION ---
        st.markdown("---")
        st.header("4. The Cost of Inaction")
        valid_vel['Velocity_Bucket'] = valid_vel['lag'].apply(lambda x: 'Active' if x > 0 else 'Quiet')
        impact_grp = valid_vel.groupby('Velocity_Bucket')[available_metrics].mean().reset_index()
        
        if len(impact_grp) == 2:
            act_row = impact_grp[impact_grp['Velocity_Bucket'] == 'Active'].iloc[0]
            quiet_row = impact_grp[impact_grp['Velocity_Bucket'] == 'Quiet'].iloc[0]
            
            for i in range(0, len(available_metrics), 5):
                cols = st.columns(len(available_metrics[i:i+5]))
                for j, m in enumerate(available_metrics[i:i+5]):
                    av, qv = act_row[m], quiet_row[m]
                    diff = ((av - qv) / qv) * 100 if qv != 0 else 0
                    lower = any(x in m.upper() for x in ['CPA', 'CPC', 'CPM', 'COST'])
                    d_col = "inverse" if lower else "normal"
                    with cols[j]: st.metric(label=m, value=f"{av:,.2f}", delta=f"{diff:+.1f}% vs Quiet", delta_color=d_col)
            
            main_cost_metric = f"Cost Per {main_conv_name}" if f"Cost Per {main_conv_name}" in available_metrics else 'CPM'
            if main_cost_metric in available_metrics:
                act_m = act_row[main_cost_metric]
                quiet_m = quiet_row[main_cost_metric]
                if act_m > 0:
                    diff_m = ((quiet_m - act_m) / act_m) * 100
                    msg = f"📉 <b>The Price of Silence:</b> When you stop launching, your <b>{main_cost_metric}</b> increases by <b>{diff_m:.1f}%</b>." if quiet_m > act_m else f"⚖️ <b>Stability:</b> Your {main_cost_metric} remains stable during quiet weeks."
                    st.markdown(f"<div class='insight-box'>{msg}</div>", unsafe_allow_html=True)

        # --- 5. WINNING CREATIVES ---
        st.markdown("---")
        st.header("5. Winning Creatives & Efficiency")
        total_ads = len(creative_agg)
        if total_ads > 0:
            winners = creative_agg[creative_agg['lifetime_spend'] >= meaningful_spend]
            slop = creative_agg[creative_agg['lifetime_spend'] < meaningful_spend]
            win_pct = (len(winners)/total_ads)*100
            slop_pct = (len(slop)/total_ads)*100
            slop_spend = slop['lifetime_spend'].sum()
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Ads", total_ads)
            c2.metric("Winners", f"{len(winners)} ({win_pct:.1f}%)", help=tooltips['winning_creatives'])
            
            slop_delta_color = "inverse" # Red if high
            c3.metric("Slop", f"{len(slop)}", delta=f"{slop_pct:.1f}% Rate", delta_color=slop_delta_color, help=tooltips['slop'])
            
            txt = f"💡 <b>Analyst Note:</b> Your Win Rate is {win_pct:.1f}%."
            txt += " High failure rate." if win_pct < 10 else " Efficient testing." if win_pct > 30 else ""
            txt += f" <br>💸 <b>Wasted Spend:</b> <b>€{slop_spend:,.0f}</b> spent on 'Slop'."
            st.markdown(f"<div class='insight-box'>{txt}</div>", unsafe_allow_html=True)

        # --- 6. AGE DISTRIBUTION ---
        st.markdown("---")
        st.header("6. Ad Age Distribution")
        raw_w_launch['Dynamic_Age_Bucket'] = raw_w_launch['spend_age_days'].apply(categorize_age)
        bucket_spend = raw_w_launch.groupby('Dynamic_Age_Bucket')[spend_col].sum().reset_index()
        total_spend = bucket_spend[spend_col].sum()
        bucket_spend['% Spend'] = (bucket_spend[spend_col] / total_spend) * 100
        
        c1, c2 = st.columns([1, 2])
        with c1: st.dataframe(bucket_spend.style.format({spend_col: '€{:.0f}', '% Spend': '{:.1f}%'}))
        with c2: st.plotly_chart(px.pie(bucket_spend, values=spend_col, names='Dynamic_Age_Bucket', title="Share of Spend by Age", color_discrete_sequence=px.colors.sequential.Teal), config={'displayModeBar': False, 'responsive': True})

        vintage_spend = bucket_spend[bucket_spend['Dynamic_Age_Bucket'].str.contains('Vintage|Legacy', regex=True)]['% Spend'].sum()
        new_spend = bucket_spend[bucket_spend['Dynamic_Age_Bucket'].str.contains('New', regex=True)]['% Spend'].sum()
        
        txt_age = f"💡 <b>Analyst Note:</b>"
        
        # IMPROVED LOGIC
        if vintage_spend > 50: 
            txt_age += f" <b>Zombie Alert:</b> {vintage_spend:.1f}% of spend is on ads older than 6 months. This is risky; you rely too much on legacy winners."
        elif new_spend > 60: 
            txt_age += f" <b>Heavy Testing:</b> {new_spend:.1f}% of spend is on New ads (<1mo). This causes high volatility."
        elif new_spend > 40: 
            txt_age += f" <b>Aggressive Rotation:</b> {new_spend:.1f}% of spend is on New ads. This is higher than the ideal 20%, suggesting you are actively fighting creative fatigue or scaling hard."
        elif vintage_spend < 15:
            txt_age += f" <b>Low Stability:</b> Only {vintage_spend:.1f}% of spend is on ads >6 months old. You lack long-term 'Evergreen' assets."
        else: 
            txt_age += f" <b>Portfolio is Balanced:</b> You have a healthy mix of New Ads ({new_spend:.1f}% for testing) and Vintage Ads ({vintage_spend:.1f}% for stability)."
            
        st.markdown(f"<div class='insight-box'>{txt_age}</div>", unsafe_allow_html=True)

        # --- 7. OLD VS NEW ---
        st.markdown("---")
        st.header("7. Performance: Old vs. New (Simpson's Paradox)")
        fresh_grp = raw_w_launch.groupby('Freshness')[numeric_cols_all].sum().reset_index()
        
        def get_val(df, m):
            if m == 'CPM': return (df[spend_col]/df[imps_col])*1000
            if m == 'CPC': return df[spend_col]/df[clicks_col]
            if m == 'CTR': return (df[clicks_col]/df[imps_col])*100
            if m == 'CPA' and installs_col: return df[spend_col]/df[installs_col]
            if m == 'ROAS' and value_col: return df[value_col]/df[spend_col]
            raw_match = next((x for x in extra_metrics if x.replace('_', ' ').title() == m), None)
            if raw_match: return df[raw_match]
            if "Cost Per" in m:
                base = m.replace("Cost Per ", "")
                raw = next((x for x in extra_metrics if x.replace('_', ' ').title() == base), None)
                if raw: return df[spend_col]/df[raw]
            return 0

        deltas = []
        for m in available_metrics:
            try:
                fr = float(get_val(fresh_grp[fresh_grp['Freshness'] == 'Fresh (<21d)'], m))
                fa = float(get_val(fresh_grp[fresh_grp['Freshness'] == 'Fatigued (>21d)'], m))
                if fa != 0:
                    d = ((fr - fa)/fa)*100
                    deltas.append({'Metric': m, 'Fresh': fr, 'Fatigued': fa, 'Delta %': d})
            except: pass
        
        delta_df = pd.DataFrame(deltas)
        st.dataframe(delta_df.style.format({'Fresh': '{:.2f}', 'Fatigued': '{:.2f}', 'Delta %': '{:+.1f}%'}).apply(lambda x: [color_delta(v, delta_df.iloc[i]['Metric']) for i, v in enumerate(x)], subset=['Delta %'], axis=0).set_table_styles([dict(selector="th", props=[("text-align", "center")]), dict(selector="td", props=[("text-align", "center")])]), use_container_width=True)

        st.markdown("""
        <div class='insight-box'>
            💡 <b>The "Simpson's Paradox" Explained:</b> You might see that launching new ads improves overall account performance (Section 3), yet this table shows that "Fresh" ads perform worse than "Fatigued" ads.
            <br><br>
            This is because <b>"Fatigued" ads are a list of Survivors.</b> They are the best creatives you've ever made. <b>"Fresh" ads are a mix of winners and losers.</b> You must accept lower average efficiency in the "Fresh" bucket to find the one unicorn that will eventually join the "Fatigued" bucket.
        </div>
        """, unsafe_allow_html=True)
        
        # Enhanced Hidden Insights (Always On)
        if not delta_df.empty:
            delta_df['Abs_Delta'] = delta_df['Delta %'].abs()
            sig_devs = delta_df[delta_df['Abs_Delta'] > 20]
            
            insights_list = []
            if not sig_devs.empty:
                for _, row in sig_devs.iterrows():
                    is_cost = any(x in row['Metric'].upper() for x in ['CPA','COST','CPC','CPM'])
                    direction = "better" if (row['Delta %'] < 0 and is_cost) or (row['Delta %'] > 0 and not is_cost) else "worse"
                    insights_list.append(f"<li><b>{row['Metric']}</b> is <b>{abs(row['Delta %']):.1f}% {direction}</b> on fresh ads.</li>")
                st.markdown(f"<div class='insight-box'>👀 <b>Hidden Insights (Significant Deviations):</b><ul>{''.join(insights_list)}</ul></div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='insight-box'>✅ <b>Consistent Quality:</b> Fresh ads perform similarly to old ads across all metrics. No hidden volatility detected.</div>", unsafe_allow_html=True)

        ovn_metric_choice = st.selectbox("Compare Metric:", available_metrics, index=0, key="ovn_select")
        eff_comp = raw_w_launch.groupby('Freshness')[numeric_cols_all].sum().reset_index()
        eff_comp['val'] = eff_comp.apply(lambda row: float(get_val(pd.DataFrame([row]), ovn_metric_choice)), axis=1)
        st.plotly_chart(px.bar(eff_comp, x='Freshness', y='val', color='Freshness', title=f"{ovn_metric_choice} Comparison", color_discrete_map={'Fresh (<21d)': '#1A776F', 'Fatigued (>21d)': '#FF7F40'}, text_auto='.2f'), config={'displayModeBar': False, 'responsive': True})

        # --- 8. DECAY ---
        st.markdown("---")
        st.header("8. The Decay Curve")
        decay_choice = st.selectbox("Select Metric:", available_metrics, index=0, key="decay_select")
        raw_w_launch['absolute_age'] = (raw_w_launch[date_col] - raw_w_launch['launch_date']).dt.days
        life_df = raw_w_launch.groupby('absolute_age')[numeric_cols_all].sum().reset_index()
        life_df['y'] = life_df.apply(lambda r: float(get_val(pd.DataFrame([r]), decay_choice)), axis=1)
        life_df = life_df[life_df['absolute_age'] <= 60]
        
        # Max Drop Logic
        drop_week = 21
        max_drop_val = 0
        if len(life_df) > 28:
            for w in [7, 14, 21, 28, 35, 42]:
                pre = life_df[(life_df['absolute_age'] >= w-7) & (life_df['absolute_age'] < w)]['y'].mean()
                post = life_df[(life_df['absolute_age'] >= w) & (life_df['absolute_age'] < w+7)]['y'].mean()
                if pre != 0:
                    change = ((post - pre) / pre) * 100
                    is_bad = (change > 0) if any(x in decay_choice.upper() for x in ['CPA','COST']) else (change < 0)
                    if is_bad and abs(change) > max_drop_val:
                        max_drop_val = abs(change)
                        drop_week = w

        early = life_df[life_df['absolute_age'] <= 7]['y'].mean()
        mid = life_df[(life_df['absolute_age'] > 14) & (life_df['absolute_age'] <= 28)]['y'].mean()
        late = life_df[(life_df['absolute_age'] > 30) & (life_df['absolute_age'] <= 60)]['y'].mean()
        
        lower_is_better = any(x in decay_choice.upper() for x in ['CPA', 'CPC', 'CPM', 'COST', 'CP_'])
        analysis_txt = f"💡 <b>Curve Analysis for {decay_choice}:</b><br>"
        
        if pd.notnull(early) and pd.notnull(mid) and early != 0:
            early_change = ((mid - early) / early) * 100
            is_early_bad = (early_change > 15) if lower_is_better else (early_change < -15)
            if is_early_bad: analysis_txt += f"• <b>Early Crash:</b> Performance degrades by <b>{abs(early_change):.1f}%</b> in the first month.<br>"
            else: analysis_txt += f"• <b>Solid Start:</b> Stable performance in the first month.<br>"

        if pd.notnull(mid) and pd.notnull(late) and mid != 0:
            late_change = ((late - mid) / mid) * 100
            is_late_bad = (late_change > 10) if lower_is_better else (late_change < -10)
            if is_late_bad: 
                analysis_txt += f"• <b>Late Fatigue:</b> Performance worsens by <b>{abs(late_change):.1f}%</b> after Day 30."
            elif (late_change < -10 and lower_is_better) or (late_change > 10 and not lower_is_better):
                analysis_txt += f"• <b>Survivor Bias:</b> Metrics improve by <b>{abs(late_change):.1f}%</b> late-stage."
            else:
                analysis_txt += f"• <b>High Endurance:</b> Performance holds steady late-stage."

        st.markdown(f"<div class='insight-box'>{analysis_txt}</div>", unsafe_allow_html=True)
        st.plotly_chart(px.line(life_df, x='absolute_age', y='y', title=f"{decay_choice} by Day", markers=True).add_vline(x=drop_week, line_dash="dash", line_color="#FF7F40", annotation_text=f"Max Drop (Day {drop_week})").update_traces(line_color='#052623'), config={'displayModeBar': False, 'responsive': True})

        # --- 9. LIFESPAN ---
        st.markdown("---")
        st.header("9. Lifespan & Retention")
        avg_lifespan = creative_agg['lifespan_days'].mean()
        winners_agg = creative_agg[creative_agg['lifetime_spend'] >= meaningful_spend]
        avg_lifespan_winners = winners_agg['lifespan_days'].mean() if not winners_agg.empty else 0
        active_now = len(creative_agg[creative_agg['last_date'] >= (raw_df[date_col].max() - pd.Timedelta(days=7))])
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Avg Lifespan (All)", f"{avg_lifespan:.1f} days")
        c2.metric("Avg Lifespan (Winners)", f"{avg_lifespan_winners:.1f} days")
        c3.metric("Active Now", f"{active_now} / {len(creative_agg)}")
        
        ret_data = []
        for t in range(61):
            pct = (len(creative_agg[creative_agg['lifespan_days'] >= t])/len(creative_agg))*100 if len(creative_agg) > 0 else 0
            ret_data.append({'Day': t, '%': pct})
        ret_df = pd.DataFrame(ret_data)
        st.plotly_chart(px.line(ret_df, x='Day', y='%', title="Survival Curve").update_traces(line_color='#1A776F', fill='tozeroy'), config={'displayModeBar': False, 'responsive': True})

        day_7 = ret_df[ret_df['Day'] == 7]['%'].values[0] if not ret_df.empty else 0
        txt_ret = f"💡 <b>Analyst Note:</b> By Day 7, <b>{day_7:.1f}%</b> of your creatives are still running."
        if day_7 < 30: txt_ret += " This indicates a <b>Fail Fast</b> strategy (High Churn)."
        elif day_7 > 50: txt_ret += " This indicates <b>Strong Retention</b>."
        else: txt_ret += " ⚖️ <b>Normal Churn:</b> Your retention is average."
        
        st.markdown(f"<div class='insight-box'>{txt_ret}</div>", unsafe_allow_html=True)

        # --- 10. EXECUTIVE SUMMARY ---
        st.markdown("---")
        st.header("10. Executive Summary & Report Card")
        score = 0
        good, bad = [], []
        
        if avg_launch_gap <= 10: 
            score += 1
            good.append(f"<b>High Velocity:</b> Launches every {avg_launch_gap:.1f} days.")
        else:
            bad.append("<b>Low Velocity:</b> Launches are too rare.")
            
        if 'win_pct' in locals() and win_pct >= 20:
            score += 1
            good.append(f"<b>High Quality:</b> {win_pct:.1f}% win rate.")
        else:
            bad.append("<b>Low Quality:</b> High creative failure rate.")
            
        if not delta_df.empty:
            best_m = delta_df.sort_values('Delta %', key=abs, ascending=False).iloc[0]
            is_cost = any(x in best_m['Metric'].upper() for x in ['CPA','COST'])
            is_better = (best_m['Delta %'] < 0 and is_cost) or (best_m['Delta %'] > 0 and not is_cost)
            if is_better:
                score += 1
                good.append(f"<b>Testing Works:</b> Fresh ads beat old ads on {best_m['Metric']}.")
            else:
                bad.append(f"<b>Testing Struggle:</b> Fresh ads perform worse on {best_m['Metric']}.")

        verdict = "💎 Elite" if score >= 3 else "🚀 Strong" if score == 2 else "🚨 Needs Focus"
        css = "good-job" if score >= 2 else "bad-job"
        st.markdown(f"""<div class="summary-box {css}"><h3>Verdict: {verdict}</h3><p><strong>Strengths:</strong></p><ul>{"".join([f"<li>{x}</li>" for x in good])}</ul><p><strong>Action Items:</strong></p><ul>{"".join([f"<li>{x}</li>" for x in bad])}</ul></div>""", unsafe_allow_html=True)

    else:
        st.error("Error: Check CSV columns.")
else:
    st.info("👈 Upload CSV to start.")
