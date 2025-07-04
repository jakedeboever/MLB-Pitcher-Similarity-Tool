import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances

CSV_PATH = 'pitchstats.csv'  # your CSV file in repo

# Stat fields (no extension)
STAT_FIELDS = [
    'arm_angle',
    'fast_usage', 'fast_vel', 'fast_ivb', 'fast_hb',
    'brk_usage', 'brk_vel', 'brk_ivb', 'brk_hb',
    'off_usage', 'off_vel', 'off_ivb', 'off_hb'
]

MANUAL_WEIGHTS = {
    'arm_angle': 1.0
}

def preprocess(df):
    # Use your combined name column exactly as-is
    df['player_name'] = df['last_name, first_name']
    
    df['throws'] = df['pitch_hand']
    
    # Calculate total pitches and usage fractions safely
    df['total_pitches'] = (df['n_fastball_formatted'] +
                           df['n_breaking_formatted'] +
                           df['n_offspeed_formatted'])
    df['fast_usage'] = df['n_fastball_formatted'] / df['total_pitches'].replace(0, np.nan)
    df['brk_usage'] = df['n_breaking_formatted'] / df['total_pitches'].replace(0, np.nan)
    df['off_usage'] = df['n_offspeed_formatted'] / df['total_pitches'].replace(0, np.nan)
    
    # Rename columns to match stat names
    df = df.rename(columns={
        'fastball_avg_speed': 'fast_vel',
        'fastball_avg_break_x': 'fast_hb',
        'fastball_avg_break_z_induced': 'fast_ivb',
        'breaking_avg_speed': 'brk_vel',
        'breaking_avg_break_x': 'brk_hb',
        'breaking_avg_break_z_induced': 'brk_ivb',
        'offspeed_avg_speed': 'off_vel',
        'offspeed_avg_break_x': 'off_hb',
        'offspeed_avg_break_z_induced': 'off_ivb'
    })
    
    # Drop rows missing critical stats
    df_clean = df.dropna(subset=['arm_angle'] +
                         ['fast_vel', 'fast_hb', 'fast_ivb',
                          'brk_vel', 'brk_hb', 'brk_ivb',
                          'off_vel', 'off_hb', 'off_ivb'])
    return df_clean

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
    df_scaled['similarity_score'] = 1 / (1 + df_scaled['distance'])
    return df_scaled.sort_values('distance').head(top_n)


# ---- STREAMLIT APP ----
st.title("⚾ Pitcher Similarity Tool (CSV with combined name)")

try:
    df = pd.read_csv(CSV_PATH)
except FileNotFoundError:
    st.error(f"CSV file not found: {CSV_PATH}\nMake sure your CSV is committed to the repo.")
    st.stop()
except Exception as e:
    st.error(f"Failed to load CSV: {e}")
    st.stop()

df = preprocess(df)

input_mode = st.radio("Input Method", ["Select Pitcher", "Enter Custom Stats"])

if input_mode == "Select Pitcher":
    pname = st.selectbox("Select pitcher", df['player_name'].unique())
    ref_row = df[df['player_name'] == pname].iloc[0]
    hand = ref_row['throws']
else:
    hand = st.radio("Handedness", ['R', 'L'])
    ref_row = {'player_id': None, 'player_name': 'Manual', 'throws': hand}
    st.markdown("Enter custom stat line:")
    for c in STAT_FIELDS:
        ref_row[c] = st.number_input(c.replace('_', ' ').title(), value=0.0, step=0.1)

# Filter to same handedness
df_filtered = df[df['throws'] == hand].reset_index(drop=True)

# Normalize and weight stats
df_scaled, scaler = normalize_and_prepare(df_filtered, STAT_FIELDS, MANUAL_WEIGHTS)

# Find top similar pitchers by Euclidean distance
results = find_similar_euclidean(df_scaled.copy(), STAT_FIELDS, ref_row, top_n=10)

st.subheader("🎯 Top Similar Pitchers")
st.dataframe(results[['player_name', 'throws', 'distance'] + STAT_FIELDS])
