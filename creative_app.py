import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Creative Velocity Dashboard", 
    page_icon="🚀",
    layout="wide"
)

# --- CUSTOM BRANDING CSS ---
st.markdown("""
    <style>
        .stApp { background-color: #FAFAFA; color: #052623; }
        p, div, label, span, li { color: #052623; font-family: 'Helvetica', 'Arial', sans-serif; }
        h1, h2, h3, h4 { color: #052623 !important; font-weight: 700; }
        
        /* Metric Cards */
        div[data-testid="stMetric"] {
            background-color: #FFFFFF; padding: 15px; border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05); border: 1px solid #E5E7EB;
            border-left: 5px solid #1A776F;
        }
        
        /* Analyst Note Box */
        .insight-box {
            background-color: #E6FFFA; border-left: 4px solid #1A776F;
            padding: 15px; margin-top: 10px; margin-bottom: 10px; border-radius: 4px;
            font-size: 1.05em; line-height: 1.6;
        }
        
        /* Summary Box */
        .summary-box { padding: 20px; border-radius: 10px; margin-bottom: 20px; }
        .good-job { background-color: #F0FDFA; border: 1px solid #1A776F; }
        .bad-job { background-color: #FFF7ED; border: 1px solid #FF7F40; }
        
        /* Educational Box */
        .edu-box {
            background-color: #F4F4F5; border: 1px solid #E4E4E7;
            padding: 10px; border-radius: 5px; font-size: 0.9em; color: #52525B;
            margin-bottom: 10px;
        }

        .dataframe { text-align: center !important; }
        th { text-align: center !important; }
        
        /* Radio Button Styling */
        div.row-widget.stRadio > div {
            flex-direction: row;
            align-items: stretch;
        }
        div.row-widget.stRadio > div[role="radiogroup"] > label {
            background-color: #FFFFFF;
            border: 1px solid #E5E7EB;
            padding: 8px 16px;
            border-radius: 5px;
            margin-right: 5px;
            transition: all 0.2s;
        }
        div.row-widget.stRadio > div[role="radiogroup"] > label[data-baseweb="radio"] {
            background-color: #E6FFFA;
            border-color: #1A776F;
            color: #1A776F;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# --- SMART DATA LOADER (Meta + TikTok Support) ---
def find_col(columns, candidates):
    # Iterate through candidates (in order of priority)
    for cand in candidates:
        for col in columns:
            # Check for match (cand inside col name)
            if cand in col: 
                return col
    return None

def clean_numeric(x):
    if isinstance(x, str):
        return x.replace(',', '').strip()
    return x

def simplify_media_type(val):
    if pd.isna(val): return "Unknown"
    val_str = str(val).lower()
    if "video" in val_str: return "Video"
    if "image" in val_str or "photo" in val_str: return "Image"
    if "carousel" in val_str: return "Carousel"
    return val # Return original if no keyword match

def load_data(file):
    try:
        df = pd.read_csv(file)
    except:
        st.error("❌ Could not read file. Ensure it is a valid CSV.")
        return None, None, None, None, None, None, None, None, None, None, None, None

    df.columns = [c.lower().strip() for c in df.columns]
    
    # 1. Date Detection
    date_col = find_col(df.columns, ['reporting starts', 'date', 'day', 'time'])
    if not date_col: return None, None, None, None, None, None, None, None, None, None, None, None
    
    # FIX: Handle TikTok Summary Rows ('-' or 'Total')
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce') 
    df = df.dropna(subset=[date_col])
    
    # 2. Column Mapping (UPDATED PRIORITY)
    spend_col = find_col(df.columns, ['amount spent', 'total cost', 'cost', 'spend'])
    imps_col = find_col(df.columns, ['impression'])
    
    # FIX: Prioritize 'clicks (all)' or 'link click' BEFORE generic 'click' to avoid grabbing 'Sound clicks'
    clicks_col = find_col(df.columns, ['clicks (all)', 'total clicks', 'link click', 'click'])
    
    ad_id_col = find_col(df.columns, ['ad id', 'creative id'])
    ad_name_col = find_col(df.columns, ['ad name', 'creative name'])
    media_type_col = find_col(df.columns, ['media type', 'format', 'image/video'])
    
    installs_col = find_col(df.columns, ['install', 'mobile app install'])
    if not installs_col: installs_col = find_col(df.columns, ['result', 'conversion', 'conversions'])
    
    value_col = find_col(df.columns, ['value', 'revenue', 'total value'])
    
    # 3. FIX: Force Numeric Types
    numeric_cols = [spend_col, imps_col, clicks_col, installs_col, value_col]
    for col in numeric_cols:
        if col:
            df[col] = df[col].apply(clean_numeric)
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Extra Metrics
    exclude = [date_col, spend_col, imps_col, clicks_col, ad_id_col, installs_col, value_col, ad_name_col, media_type_col]
    numeric_candidates = df.select_dtypes(include=[np.number]).columns.tolist()
    extra_metrics = [c for c in numeric_candidates if c not in exclude and 'id' not in c and 'date' not in c]

    # Main Conversion Name
    conversion_name = "Action"
    if installs_col: conversion_name = "Install"
    elif value_col: conversion_name = "Purchase"
    elif len(extra_metrics) > 0: conversion_name = extra_metrics[0].replace('_', ' ').title()

    return df, date_col, spend_col, imps_col, installs_col, clicks_col, value_col, ad_id_col, ad_name_col, media_type_col, extra_metrics, conversion_name

def categorize_age_granular(days):
    if days <= 21: return '1. New (0-21d)'
    elif days <= 60: return '2. Recent (22-60d)'
    elif days <= 120: return '3. Mature (2-4 Mo)'
    elif days <= 240: return '4. Vintage (4-8 Mo)'
    else: return '5. Legacy (8 Mo+)'

def calculate_metric(df, metric, spend_col, imps_col, clicks_col, installs_col, value_col, extra_metrics):
    if df.empty: return 0
    if metric == 'CPM': return (df[spend_col].sum() / df[imps_col].sum()) * 1000 if df[imps_col].sum() > 0 else 0
    if metric == 'CPC': return df[spend_col].sum() / df[clicks_col].sum() if df[clicks_col].sum() > 0 else 0
    if metric == 'CTR': return (df[clicks_col].sum() / df[imps_col].sum()) * 100 if df[imps_col].sum() > 0 else 0
    if metric == 'CPA' and installs_col: return df[spend_col].sum() / df[installs_col].sum() if df[installs_col].sum() > 0 else 0
    if metric == 'IPM' and installs_col: return (df[installs_col].sum() / df[imps_col].sum()) * 1000 if df[imps_col].sum() > 0 else 0
    if metric == 'ROAS' and value_col: return df[value_col].sum() / df[spend_col].sum() if df[spend_col].sum() > 0 else 0
    if "Cost Per" in metric:
        base = metric.replace("Cost Per ", "")
        raw = next((x for x in extra_metrics if x.replace('_', ' ').title() == base), None)
        if raw and df[raw].sum() > 0: return df[spend_col].sum() / df[raw].sum()
    raw_match = next((x for x in extra_metrics if x.replace('_', ' ').title() == metric), None)
    if raw_match: return df[raw_match].sum()
    return 0

# --- MAIN APP ---
uploaded_file = st.sidebar.file_uploader("Upload CSV File (Meta, TikTok, etc)", type=['csv'])

if uploaded_file is not None:
    raw_df, date_col, spend_col, imps_col, installs_col, clicks_col, value_col, ad_id_col, ad_name_col, media_type_col, extra_metrics, main_conv_name = load_data(uploaded_file)
    
    if raw_df is not None and spend_col and ad_id_col:
        st.sidebar.success("✅ Data Parsed Successfully")
        
        # --- SETTINGS ---
        st.sidebar.markdown("---")
        st.sidebar.header("⚙️ Settings")
        min_spend_global = st.sidebar.number_input("Global Min Spend Filter (Currency)", value=0)
        meaningful_spend = st.sidebar.number_input("Meaningful Spend Threshold (Currency)", value=50)
        
        # --- PROCESSING ---
        numeric_cols_all = raw_df.select_dtypes(include=[np.number]).columns.tolist()
        if ad_id_col in numeric_cols_all: numeric_cols_all.remove(ad_id_col)
            
        creative_agg = raw_df.groupby(ad_id_col).agg({date_col: ['min', 'max'], **{c:'sum' for c in numeric_cols_all}}).reset_index()
        creative_agg.columns = [col[0] if col[1] == '' else f"{col[0]}_{col[1]}" if col[0] == date_col else col[0] for col in creative_agg.columns]
        creative_agg.rename(columns={f'{date_col}_min': 'launch_date', f'{date_col}_max': 'last_date', spend_col: 'lifetime_spend'}, inplace=True)
        creative_agg['lifespan_days'] = (creative_agg['last_date'] - creative_agg['launch_date']).dt.days + 1
        
        raw_df = raw_df[raw_df[spend_col] > min_spend_global]
        raw_df['week_start'] = raw_df[date_col].dt.to_period('W-MON').apply(lambda r: r.start_time)
        
        available_metrics = ['CPM', 'CPC', 'CTR']
        if installs_col: available_metrics.extend(['CPA', 'IPM'])
        if value_col: available_metrics.append('ROAS')
        for m in extra_metrics:
            clean = m.replace('_', ' ').title()
            available_metrics.append(clean)
            available_metrics.append(f"Cost Per {clean}")

        # --- SIDEBAR INFO ---
        st.sidebar.markdown("---")
        avg_cpm = (raw_df[spend_col].sum() / raw_df[imps_col].sum()) * 1000
        cost_to_test = (4000 / 1000) * avg_cpm
        st.sidebar.info(f"Avg CPM: {avg_cpm:.2f}\n\nEst. cost to test 1 ad (4k imps): **{cost_to_test:.2f}**")

        # --- 1. RHYTHM ---
        st.header("1. Creative Pulse & Consistency")
        st.caption("Tracks your launch cadence. Consistent launching prevents 'Performance Crashes' caused by fatigue.")
        
        launch_dates = sorted(creative_agg['launch_date'].unique())
        avg_gap = pd.Series(launch_dates).diff().dt.days.mean() if len(launch_dates) > 1 else 0
        max_gap = pd.Series(launch_dates).diff().dt.days.max() if len(launch_dates) > 1 else 0
        
        drought_start, drought_end = None, None
        if len(launch_dates) > 1:
            diffs = pd.Series(launch_dates).diff().dt.days.fillna(0).values
            max_idx = np.argmax(diffs)
            if diffs[max_idx] > 0:
                drought_end = launch_dates[max_idx]
                drought_start = launch_dates[max_idx-1]

        c1, c2, c3 = st.columns(3)
        c1.metric("Unique Launch Days", len(launch_dates))
        c2.metric("Avg Launch Gap", f"{avg_gap:.1f} days")
        c3.metric("Longest Drought", f"{max_gap:.0f} days")
        
        txt_rhythm = f"💡 <b>Analyst Note:</b> Your average launch gap is <b>{avg_gap:.1f} days</b>. "
        if avg_gap <= 7: txt_rhythm += "🏆 <b>Elite Consistency:</b> Weekly launches prevent fatigue."
        elif avg_gap <= 14: txt_rhythm += "✅ <b>Good Rhythm:</b> Bi-weekly testing is sustainable."
        else: txt_rhythm += "⚠️ <b>Inconsistent:</b> Gaps >14 days create volatility."
        if max_gap > 30: st.warning(f"⚠️ Stability Risk: You went {max_gap:.0f} days without launching.")
        st.markdown(f"<div class='insight-box'>{txt_rhythm}</div>", unsafe_allow_html=True)

        # --- 2. VOLUME VS PERFORMANCE ---
        st.markdown("---")
        st.subheader("2. Active Creative Volume vs. Performance")
        st.caption("Does running MORE ads simultaneously help or hurt your efficiency? Use this to find your 'Saturation Point'.")
        
        vol_metric = st.radio("Select Performance Metric:", available_metrics, index=0, key='vol_kpi', horizontal=True)
        
        weekly_active = raw_df.groupby('week_start')[ad_id_col].nunique().reset_index(name='Active Creatives')
        weekly_perf = raw_df.groupby('week_start').apply(lambda x: calculate_metric(x, vol_metric, spend_col, imps_col, clicks_col, installs_col, value_col, extra_metrics)).reset_index(name=vol_metric)
        vol_df = pd.merge(weekly_active, weekly_perf, on='week_start')
        
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Bar(x=vol_df['week_start'], y=vol_df['Active Creatives'], name='Active Ads Count', marker_color='rgba(26, 119, 111, 0.3)', yaxis='y'))
        fig_vol.add_trace(go.Scatter(x=vol_df['week_start'], y=vol_df[vol_metric], name=vol_metric, yaxis='y2', line=dict(color='#FF7F40', width=3)))
        fig_vol.update_layout(title="Weekly Active Ads vs KPI", yaxis=dict(title='Active Ads Count'), yaxis2=dict(title=vol_metric, overlaying='y', side='right'), hovermode="x unified")
        st.plotly_chart(fig_vol, use_container_width=True)
        
        corr_vol = vol_df['Active Creatives'].corr(vol_df[vol_metric])
        txt_vol = f"💡 <b>Analyst Note:</b> Correlation is <b>{corr_vol:.2f}</b>. "
        if abs(corr_vol) < 0.2: txt_vol += "Running more ads has <b>no major impact</b> on efficiency."
        else: txt_vol += "Ensure you aren't diluting spend by running too many weak ads."
        st.markdown(f"<div class='insight-box'>{txt_vol}</div>", unsafe_allow_html=True)

        # --- 3. COMPOSITION ---
        st.markdown("---")
        st.header("3. Spend Composition: Fresh vs. Fatigued")
        st.caption("Visualizes budget health. Are you relying on decaying winners (Fatigued) or testing new concepts (Fresh)?")
        
        creative_agg['launch_week'] = creative_agg['launch_date'].dt.to_period('W-MON').apply(lambda r: r.start_time)
        weekly_new_creatives = creative_agg.groupby('launch_week')[ad_id_col].count().reset_index(name='new_creatives_count')
        
        raw_w_launch = pd.merge(raw_df, creative_agg[[ad_id_col, 'launch_date']], on=ad_id_col, how='left')
        raw_w_launch['spend_age_days'] = (raw_w_launch[date_col] - raw_w_launch['launch_date']).dt.days
        raw_w_launch['Freshness'] = raw_w_launch['spend_age_days'].apply(lambda x: 'Fresh (<21d)' if x < 21 else 'Fatigued (>21d)')
        
        comp_df = raw_w_launch.groupby(['week_start', 'Freshness'])[spend_col].sum().reset_index()
        all_weeks = pd.DataFrame({'week_start': comp_df['week_start'].unique()})
        new_counts = pd.merge(all_weeks, weekly_new_creatives, left_on='week_start', right_on='launch_week', how='left').fillna(0).sort_values('week_start')

        fig_dual = go.Figure()
        for status, color in [('Fatigued (>21d)', '#FF7F40'), ('Fresh (<21d)', '#1A776F')]:
            subset = comp_df[comp_df['Freshness'] == status]
            fig_dual.add_trace(go.Bar(
                x=subset['week_start'].astype(str), 
                y=subset[spend_col], 
                name=status, 
                marker_color=color,
                hovertemplate=f"<b>{status}</b><br>Spend: %{{y:,.0f}}<extra></extra>"
            ))
        
        fig_dual.add_trace(go.Scatter(
            x=new_counts['week_start'], 
            y=new_counts['new_creatives_count'], 
            name="New Ads Launched", 
            yaxis='y2', 
            mode='lines', 
            line=dict(color='black', width=2),
            hovertemplate="<b>New Launches</b><br>Count: %{y}<extra></extra>"
        ))
        
        if drought_start and drought_end and max_gap > 14:
            fig_dual.add_vrect(x0=drought_start, x1=drought_end, fillcolor="red", opacity=0.15, layer="below", line_width=0, annotation_text="Longest Drought", annotation_position="top left")
        
        fig_dual.update_layout(barmode='stack', title='Weekly Spend Composition + Launch Volume', yaxis=dict(title='Spend'), yaxis2=dict(title='New Ads Launched', overlaying='y', side='right', showgrid=False), hovermode="x unified")
        st.plotly_chart(fig_dual, use_container_width=True)
        
        fresh_share = raw_w_launch[raw_w_launch['Freshness'] == 'Fresh (<21d)'][spend_col].sum() / raw_w_launch[spend_col].sum() * 100
        
        drought_note = ""
        if drought_start and drought_end and max_gap > 14:
            mask_during = (raw_df[date_col] >= drought_start) & (raw_df[date_col] <= drought_end)
            mask_after = (raw_df[date_col] > drought_end) & (raw_df[date_col] <= drought_end + pd.Timedelta(days=14))
            
            changes = []
            rate_metrics = [m for m in available_metrics if m in ['CPM', 'CPC', 'CTR', 'CPA', 'IPM', 'ROAS'] or m.startswith('Cost Per')]
            
            for m in rate_metrics:
                val_d = calculate_metric(raw_df[mask_during], m, spend_col, imps_col, clicks_col, installs_col, value_col, extra_metrics)
                val_a = calculate_metric(raw_df[mask_after], m, spend_col, imps_col, clicks_col, installs_col, value_col, extra_metrics)
                if val_d > 0:
                    diff = ((val_a - val_d)/val_d)*100
                    if abs(diff) > 1.0:
                        changes.append(f"{m}: {diff:+.1f}%")
            
            if changes:
                drought_note = f"<br>⚠️ <b>Drought Impact:</b> After the drought ended: " + ", ".join(changes) + "."

        st.markdown(f"<div class='insight-box'>💡 <b>Analyst Note:</b> <b>{fresh_share:.1f}%</b> of spend is on Fresh ads. The black line tracks launch volume.{drought_note}</div>", unsafe_allow_html=True)

        # --- 4. CORRELATION ---
        st.markdown("---")
        st.header("4. Correlation Analysis")
        st.caption("Does the act of launching new ads statistically improve your account performance? We apply a lag to account for learning phase.")
        
        st.markdown("""
        <div class='edu-box'>
        <b>📊 How to read Correlation:</b><br>
        • <b>0.7 to 1.0:</b> Very Strong Relationship<br>
        • <b>0.4 to 0.7:</b> Moderate Relationship<br>
        • <b>0.0 to 0.3:</b> No Relationship (Random)<br>
        • <b>Note:</b> Correlation is not Causation, but it's a strong hint.
        </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns([2, 1])
        with c2: lag_weeks = st.slider("Weeks Lag", 0, 8, 1, help="Shift launch data forward to see delayed impact on performance.")
        
        weekly_stats = raw_df.groupby('week_start').apply(lambda x: pd.Series({m: calculate_metric(x, m, spend_col, imps_col, clicks_col, installs_col, value_col, extra_metrics) for m in available_metrics})).reset_index()
        analysis_df = pd.merge(weekly_stats, weekly_new_creatives.rename(columns={'launch_week':'week_start'}), on='week_start', how='left').fillna(0)
        
        corr_data = []
        for m in available_metrics:
            tmp = analysis_df.copy()
            tmp['lag'] = tmp['new_creatives_count'].shift(lag_weeks)
            tmp = tmp.dropna()
            if len(tmp) > 2:
                corr = tmp['lag'].corr(tmp[m])
                lower_good = any(x in m.upper() for x in ['CPA', 'CPC', 'CPM', 'COST'])
                impact = "Good" if (corr < 0 and lower_good) or (corr > 0 and not lower_good) else "Bad"
                corr_data.append({'Metric': m, 'Correlation': corr, 'Impact': impact})
        
        if corr_data:
            corr_df = pd.DataFrame(corr_data).sort_values('Correlation', ascending=False)
            fig_c = px.bar(corr_df, x='Correlation', y='Metric', color='Impact', title=f"Velocity Correlation (Lag: {lag_weeks}w)", color_discrete_map={'Good': '#1A776F', 'Bad': '#FF7F40'}, orientation='h')
            fig_c.add_vline(x=0, line_width=1, line_color="black")
            st.plotly_chart(fig_c, use_container_width=True)
            
            best = corr_df.iloc[0]
            st.markdown(f"<div class='insight-box'>💡 <b>Analyst Note:</b> Strongest correlation is with <b>{best['Metric']}</b> ({best['Correlation']:.2f}). Launching new ads has the biggest impact on this metric.</div>", unsafe_allow_html=True)

        # --- 5. COST OF INACTION ---
        st.markdown("---")
        st.header("5. The Cost of Inaction")
        st.caption("Comparing performance during 'Active' launch weeks vs. 'Quiet' weeks.")
        
        analysis_df['Status'] = analysis_df['new_creatives_count'].shift(lag_weeks).apply(lambda x: 'Active' if x > 0 else 'Quiet')
        impact_grp = analysis_df.groupby('Status')[available_metrics].mean().reset_index()
        
        if len(impact_grp) == 2:
            act = impact_grp[impact_grp['Status'] == 'Active'].iloc[0]
            quiet = impact_grp[impact_grp['Status'] == 'Quiet'].iloc[0]
            
            for i in range(0, len(available_metrics), 4):
                cols = st.columns(4)
                for j, m in enumerate(available_metrics[i:i+4]):
                    av, qv = act[m], quiet[m]
                    diff = ((qv - av) / av) * 100 if av > 0 else 0
                    
                    lower_good = any(x in m.upper() for x in ['CPA', 'CPC', 'CPM', 'COST'])
                    color = "inverse" if lower_good else "normal"
                    with cols[j]: st.metric(m, f"{qv:,.2f}", delta=f"{diff:+.1f}% vs Active", delta_color=color)

            st.markdown(f"<div class='insight-box'>💡 <b>Analyst Note:</b> Values shown are averages during <b>Quiet Weeks</b>. <br>• Red deltas = Metric gets worse when you pause. <br>• Green deltas = Metric improves (or stays stable). <br>• If your CPA/ROAS degrades during Quiet weeks, you are <b>Launch Dependent</b>.</div>", unsafe_allow_html=True)

        # --- 6. WINNERS & SLOP ---
        st.markdown("---")
        st.header("6. Winning Creatives & Efficiency")
        st.caption("Efficiency Audit: Are we finding winners or burning cash on low-spend failures?")
        
        total_ads = len(creative_agg)
        winners = creative_agg[creative_agg['lifetime_spend'] >= meaningful_spend]
        slop = creative_agg[creative_agg['lifetime_spend'] < meaningful_spend]
        win_pct = (len(winners)/total_ads)*100
        slop_pct = (len(slop)/total_ads)*100
        slop_spend = slop['lifetime_spend'].sum()
        total_spend_all = creative_agg['lifetime_spend'].sum()
        slop_spend_share = (slop_spend / total_spend_all) * 100 if total_spend_all > 0 else 0
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Ads", total_ads)
        c2.metric("Winners", f"{len(winners)} ({win_pct:.1f}%)")
        c3.metric("Creative Slop", f"{len(slop)} ({slop_pct:.1f}%)")
        
        st.markdown(f"<div class='insight-box'>💡 <b>Analyst Note:</b> <br>• <b>Win Rate:</b> {win_pct:.1f}%. (Benchmark: >20% for scale, >30% for efficiency). <br>• <b>Wasted Budget:</b> You spent <b>{slop_spend:,.0f} ({slop_spend_share:.1f}%)</b> of your total budget on Slop. (Benchmark: Keep <10%).</div>", unsafe_allow_html=True)

        # --- 7. AD AGE DISTRIBUTION ---
        st.markdown("---")
        st.header("7. Ad Age Distribution")
        st.caption("Portfolio Balance: Do you have a healthy mix of New (Testing), Mature (Scaling), and Legacy (Profit) ads?")
        
        raw_w_launch['Granular_Age'] = raw_w_launch['spend_age_days'].apply(categorize_age_granular)
        
        age_agg = raw_w_launch.groupby('Granular_Age').apply(
            lambda x: pd.Series({
                'Spend': x[spend_col].sum(),
                'CPM': calculate_metric(x, 'CPM', spend_col, imps_col, clicks_col, installs_col, value_col, extra_metrics),
                'CTR': calculate_metric(x, 'CTR', spend_col, imps_col, clicks_col, installs_col, value_col, extra_metrics),
                f'Cost Per {main_conv_name}': calculate_metric(x, f'Cost Per {main_conv_name}' if f'Cost Per {main_conv_name}' in available_metrics else 'CPM', spend_col, imps_col, clicks_col, installs_col, value_col, extra_metrics),
                'ROAS': calculate_metric(x, 'ROAS', spend_col, imps_col, clicks_col, installs_col, value_col, extra_metrics)
            })
        ).reset_index()
        age_agg['Share of Spend'] = (age_agg['Spend'] / age_agg['Spend'].sum()) * 100
        
        c1, c2 = st.columns([2, 1])
        with c1:
            fmt = {'Spend': '{:.0f}', 'Share of Spend': '{:.1f}%', 'CPM': '{:.2f}', 'CTR': '{:.2f}%', f'Cost Per {main_conv_name}': '{:.2f}', 'ROAS': '{:.2f}'}
            st.dataframe(age_agg.style.format(fmt, na_rep="-"), use_container_width=True)
        with c2:
            st.plotly_chart(px.pie(age_agg, values='Spend', names='Granular_Age', title="Portfolio Balance", color_discrete_sequence=px.colors.sequential.Teal), use_container_width=True)
            
        st.markdown(f"<div class='insight-box'>💡 <b>Analyst Note:</b> Compare your <b>New</b> vs <b>Legacy</b> performance. If Legacy ads have significantly better CPA, you are relying on old winners. If New ads are better, your recent creative strategy is working well.</div>", unsafe_allow_html=True)

        # --- 8. DECAY & RETENTION (SUPER CHART) ---
        st.markdown("---")
        st.header("8. The Decay Curve & Retention")
        st.caption("Visualizes Lifecycle. Left Axis = Performance. Right Axis = Retention %. Purple Line = Half-Life.")
        
        decay_metric = st.radio("Select Metric to analyze against Retention:", available_metrics, index=0, key='decay_m_combined', horizontal=True)
        
        raw_w_launch['abs_age'] = (raw_w_launch[date_col] - raw_w_launch['launch_date']).dt.days
        life_df = raw_w_launch.groupby('abs_age').apply(lambda x: calculate_metric(x, decay_metric, spend_col, imps_col, clicks_col, installs_col, value_col, extra_metrics)).reset_index(name='y')
        
        total_creatives = len(creative_agg)
        ret_data = [{'abs_age': t, 'retention': (len(creative_agg[creative_agg['lifespan_days'] >= t])/total_creatives)*100} for t in range(61)]
        ret_df = pd.DataFrame(ret_data)
        
        combo = pd.merge(life_df, ret_df, on='abs_age', how='left')
        combo = combo[combo['abs_age'] <= 60]
        
        # Sensitive Max Drop
        check_limit = len(combo) - 3 # Exclude tail noise
        subset_y = combo['y'].iloc[:check_limit]
        diffs = subset_y.diff()
        lower_bad = any(x in decay_metric.upper() for x in ['CPA','COST', 'CPM', 'CPC'])
        drop_idx = diffs.idxmax() if lower_bad else diffs.idxmin()
        if pd.isna(drop_idx) or drop_idx < 1: drop_idx = 21
        
        fig_combo = go.Figure()
        
        # Area (Retention)
        fig_combo.add_trace(go.Scatter(x=combo['abs_age'], y=combo['retention'], name="Retention %", fill='tozeroy', line=dict(color='rgba(26, 119, 111, 0.2)', width=0), yaxis='y2'))
        
        # Line (Performance)
        fig_combo.add_trace(go.Scatter(x=combo['abs_age'], y=combo['y'], name=decay_metric, mode='lines+markers', line=dict(color='#052623', width=2)))
        
        # Max Drop Line
        fig_combo.add_vline(x=drop_idx, line_dash="dash", line_color="#FF7F40", annotation_text=f"Max Shift (Day {drop_idx})")
        
        # Half Life Line
        under_50 = ret_df[ret_df['retention'] < 50]
        h_life = under_50['abs_age'].min() if not under_50.empty else None
        
        if h_life is not None and not pd.isna(h_life):
             fig_combo.add_vline(x=h_life, line_dash="dot", line_color="purple", annotation_text=f"Half-Life ({int(h_life)}d)", annotation_position="top right")

        fig_combo.update_layout(
            title=f"{decay_metric} Performance vs Retention",
            xaxis=dict(title="Days Since Launch"),
            yaxis=dict(title=decay_metric, side='left'),
            yaxis2=dict(title="Retention %", side='right', overlaying='y', range=[0, 100], showgrid=False),
            hovermode="x unified", legend=dict(orientation="h", y=1.1)
        )
        st.plotly_chart(fig_combo, use_container_width=True)
        
        txt_combo = f"💡 <b>Deep Dive:</b><br>"
        txt_combo += f"• <b>Max Performance Shift:</b> Day {drop_idx}.<br>"
        
        hl_text = f"Day {int(h_life)}" if h_life is not None else "Day 60+"
        txt_combo += f"• <b>Creative Half-Life:</b> {hl_text} (Purple Line). 50% of your ads are dead by this day.<br>"
        
        if h_life is not None and drop_idx < h_life:
             txt_combo += f"⚠️ <b>Reactive Pausing:</b> Performance crashes at Day {drop_idx}, but you wait until {hl_text} to cut ads."
        elif h_life is not None:
             txt_combo += f"✅ <b>Proactive Pausing:</b> You cut ads ({hl_text}) before the major crash (Day {drop_idx})."
             
        st.markdown(f"<div class='insight-box'>{txt_combo}</div>", unsafe_allow_html=True)

        # --- 9. ATTRIBUTES (Multi-Group & Merged) ---
        st.markdown("---")
        st.header("9. Creative Attribute Deep Dive")
        st.caption("Compare performance by Ad Name Tags and Media Type in a single view.")
        
        st.subheader("🅰️ Ad Name Multi-Group + 📷 Media Type")
        search_terms = st.text_area("Enter tags separated by commas (e.g. UGC, Static, Offer):", value="UGC, Static")
        att_metric_name = st.radio("Select Metric:", available_metrics, index=0, key='att_name', horizontal=True)
        
        if search_terms and ad_name_col:
            tags = [t.strip() for t in search_terms.split(',') if t.strip()]
            
            def assign_group(name):
                name_str = str(name).lower()
                for t in tags:
                    if t.lower() in name_str: return t
                return "Other"
            
            raw_df['Ad_Group'] = raw_df[ad_name_col].apply(assign_group)
            
            # Group by Name Group AND Media Type (if exists)
            group_cols = ['Ad_Group']
            color_col = 'Ad_Group'
            if media_type_col:
                # Simplify TikTok/Meta Media Types
                raw_df['Simple_Media_Type'] = raw_df[media_type_col].apply(simplify_media_type)
                group_cols.append('Simple_Media_Type')
                color_col = 'Simple_Media_Type'

            grp = raw_df.groupby(group_cols).apply(lambda x: calculate_metric(x, att_metric_name, spend_col, imps_col, clicks_col, installs_col, value_col, extra_metrics)).reset_index(name='Value')
            
            # Grouped Bar Chart
            fig_att = px.bar(grp, x='Ad_Group', y='Value', color=color_col, barmode='group', title=f"{att_metric_name} by Group & Media Type", text_auto='.2f')
            st.plotly_chart(fig_att, use_container_width=True)
            
            if not media_type_col:
                 st.info("ℹ️ Add a 'Media Type' column to your CSV to see Image vs Video splits within these groups.")

        # --- 10. EXECUTIVE SUMMARY ---
        st.markdown("---")
        st.header("10. Executive Summary")
        st.caption("Strategic Overview & Action Plan")
        
        # Logic for Summary Points
        highlights = []
        warnings = []
        
        if avg_gap <= 10: highlights.append(f"✅ <b>High Velocity:</b> Launching every {avg_gap:.1f} days.")
        else: warnings.append(f"⚠️ <b>Low Velocity:</b> Launching every {avg_gap:.1f} days (Target: <10).")
        
        if win_pct >= 20: highlights.append(f"✅ <b>Strong Win Rate:</b> {win_pct:.1f}% of ads succeed.")
        else: warnings.append(f"⚠️ <b>Low Win Rate:</b> Only {win_pct:.1f}% of ads succeed (Target: >20%).")
        
        if slop_spend_share > 10: warnings.append(f"💸 <b>High Waste:</b> {slop_spend_share:.1f}% budget spent on failed tests (Slop).")
        
        if h_life is not None and drop_idx < h_life: warnings.append(f"📉 <b>Slow Reaction:</b> Ads crash at Day {drop_idx} but you hold them until Day {int(h_life)}.")
        
        if not highlights and not warnings:
            highlights.append("Data loaded successfully. Review sections for specific insights.")

        verdict = "💎 Elite" if len(warnings) == 0 else "🚀 Strong" if len(warnings) <= 2 else "🚨 Needs Focus"
        css = "good-job" if len(warnings) <= 1 else "bad-job"
        
        st.markdown(f"""
        <div class="summary-box {css}">
            <h3>Verdict: {verdict}</h3>
            <p><strong>🌟 Strategic Highlights:</strong></p>
            <ul>{"".join([f"<li>{x}</li>" for x in highlights])}</ul>
            <p><strong>⚡ Critical Actions:</strong></p>
            <ul>{"".join([f"<li>{x}</li>" for x in warnings])}</ul>
        </div>""", unsafe_allow_html=True)

        # --- METHODOLOGY ---
        st.markdown("---")
        with st.expander("🔬 Methodology & Calculation Logic"):
            st.markdown("""
            - **Grouping:** All data is aggregated by `Ad ID`.
            - **Fresh vs Fatigued:** Split at Day 21. Fresh = Testing. Fatigued = Scaling.
            - **Decay Normalization:** All ads aligned to Day 0 (Launch Date). We detect drops based on raw daily changes to find the 'cliff'.
            - **Creative Slop:** Ads that spent < Threshold (Default 50). Represents wasted production effort.
            - **Cost of Inaction:** Compares weeks with 0 launches vs. weeks with >0 launches.
            """)

    else:
        st.error("Error: Check CSV columns.")
else:
    st.info("👈 Upload CSV to begin.")
