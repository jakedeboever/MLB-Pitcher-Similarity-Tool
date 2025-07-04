import streamlit as st
import pandas as pd
import numpy as np
from pybaseball import statcast, playerid_reverse_lookup
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances

# ---- CONFIGURATION ----
PITCH_TYPES = {
    'fast': ['FF', 'FT', 'SI', 'FC'],
    'brk': ['SL', 'CU', 'KC'],
    'off': ['CH', 'SF', 'FS']
}

STAT_FIELDS = [
    'arm_angle', 'extension',
    'fast_usage', 'fast_vel', 'fast_ivb', 'fast_hb',
    'brk_usage', 'brk_vel', 'brk_ivb', 'brk_hb',
    'off_usage', 'off_vel', 'off_ivb', 'off_hb'
]

MANUAL_WEIGHTS = {
    'arm_angle': 1.0,
    'extension': 1.0
}

# ---- FUNCTIONS ----
@st.cache_data(show_spinner=False)
def add_player_info(df):
    pitcher_ids = df['pitcher'].unique().tolist()
    info_list = []
    for pid in pitcher_ids:
        try:
            info = playerid_reverse_lookup(pid)
            if not info.empty:
                info = info.iloc[0]
                info_list.append({
                    'pitcher': pid,
                    'player_name': f"{info['name_first']} {info['name_last']}",
                    'throws': info['throws']
                })
        except Exception:
            # If lookup fails, skip that pitcher
            pass
    info_df = pd.DataFrame(info_list)
    df = df.merge(info_df, how='left', on='pitcher')
    return df

@st.cache_data(show_spinner=False)
def load_data(start_date, end_date):
    df = statcast(start_date, end_date)
    df = df[df['pitcher'].notnull()]
    df = add_player_info(df)
    return df

def aggregate(df_raw):
    aggs = []
    for pid, name, hand in df_raw[['pitcher','player_name','throws']].drop_duplicates().values:
        dd = df_raw[df_raw['pitcher']==pid]
        row = dict(player_id=pid, player_name=name, throws=hand)
        row['arm_angle'] = dd['release_pos_y'].mean()
        row['extension'] = dd['release_extension'].mean()

        for tag, types in PITCH_TYPES.items():
            sub = dd[dd['pitch_type'].isin(types)]
            row[f'{tag}_usage'] = sub.shape[0] / dd.shape[0] if dd.shape[0] > 0 else 0
            row[f'{tag}_vel'] = sub['release_speed'].mean()
            row[f'{tag}_ivb'] = sub['induced_vert_break'].mean()
            row[f'{tag}_hb'] = sub['pfx_x'].mean()

        aggs.append(row)

    return pd.DataFrame(aggs).dropna()

def usage_weight_columns(df, manual_weights):
    df = df.copy()
    for prefix in ['fast', 'brk', 'off']:
        usage_col = f'{prefix}_usage'
        for suffix in ['vel', 'ivb', 'hb']:
            col = f'{prefix}_{suffix}'
            if col in df.columns and usage_col in df.columns:
                df[col] = df[col] * df[usage_col]

    for col, w in manual_weights.items():
        if col in df.columns:
            df[col] = df[col] * w

    return df

def normalize_and_prepare(df, stat_cols, manual_weights):
    df_weighted = usage_weight_columns(df, manual_weights)
    scaler = StandardScaler()
    df_scaled = df_weighted.copy().reset_index(drop=True)
    df_scaled[stat_cols] = scaler.fit_transform(df_scaled[stat_cols])
    return df_scaled, scaler

def find_similar_euclidean(df_scaled, stat_cols, reference, top_n=5):
    ref_vec = np.array([reference[c] for c in stat_cols]).reshape(1, -1)
    distances = pairwise_distances(ref_vec, df_scaled[stat_cols].values, metric='euclidean')[0]
    df_scaled['distance'] = distances
    return df_scaled.sort_values('distance').head(top_n)

# ---- STREAMLIT APP ----
st.title("⚾ Pitcher Similarity Tool (2025 Season & Euclidean Distance)")

col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start Date", value=pd.to_datetime("2025-03-28"))
with col2:
    end_date = st.date_input("End Date", value=pd.to_datetime("2025-10-01"))

if start_date > end_date:
    st.error("End date must be after start date.")
    st.stop()

if st.button("Load Data"):
    with st.spinner("Loading Statcast data and player info..."):
        raw_data = load_data(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        stats_df = aggregate(raw_data)
        stats_scaled, scaler_obj = normalize_and_prepare(stats_df, STAT_FIELDS, MANUAL_WEIGHTS)
        st.session_state.stats_df = stats_df
        st.session_state.stats_scaled = stats_scaled
        st.session_state.scaler = scaler_obj
        st.success(f"Loaded {len(stats_df)} pitchers.")

if 'stats_df' in st.session_state:
    input_mode = st.radio("Input Method", ["Select Pitcher", "Enter Custom Stats"])

    if input_mode == "Select Pitcher":
        pname = st.selectbox("Pitcher", st.session_state.stats_df['player_name'])
        ref_row = st.session_state.stats_df[st.session_state.stats_df['player_name'] == pname].iloc[0]
        hand = ref_row['throws']
    else:
        hand = st.radio("Handedness", ['R', 'L'])
        ref_row = {'player_id': None, 'player_name': 'Manual', 'throws': hand}
        st.markdown("Enter your custom stat line:")
        for c in STAT_FIELDS:
            ref_row[c] = st.number_input(c.replace('_',' ').title(), value=0.0, step=0.1)

    # Filter same-handed pitchers
    df = st.session_state.stats_df
    df_scaled = st.session_state.stats_scaled
    mask = df['throws'] == hand
    df = df[mask].reset_index(drop=True)
    df_scaled = df_scaled[mask].reset_index(drop=True)

    # Similarity
    sim_results = find_similar_euclidean(df_scaled.copy(), STAT_FIELDS, ref_row, top_n=10)
    st.subheader("🎯 Top Similar Pitchers")
    st.dataframe(sim_results[['player_name', 'throws', 'distance'] + STAT_FIELDS])
