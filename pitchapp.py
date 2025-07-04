import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances

CSV_PATH = 'pitchstats.csv'  # Your uploaded CSV file

# Feature columns (NO extension)
STAT_FIELDS = [
    'arm_angle',
    'fast_usage', 'fast_vel', 'fast_ivb', 'fast_hb',
    'brk_usage', 'brk_vel', 'brk_ivb', 'brk_hb',
    'off_usage', 'off_vel', 'off_ivb', 'off_hb'
]

# Arm angle manual weight to make it ~20% of total similarity
MANUAL_WEIGHTS = {
    'arm_angle': 1.0
}

# Movement vs Velocity weights (per pitch group)
PITCH_COMPONENT_WEIGHTS = {
    'vel': 0.5,
    'ivb': 0.25,
    'hb': 0.25
}

def preprocess(df):
    # Use combined name column
    df['player_name'] = df['last_name, first_name']
    df['throws'] = df['pitch_hand']

    # Total pitches thrown
    df['total_pitches'] = (
        df['n_fastball_formatted'] +
        df['n_breaking_formatted'] +
        df['n_offspeed_formatted']
    )

    # Usage fractions
    df['fast_usage'] = df['n_fastball_formatted'] / df['total_pitches'].replace(0, np.nan)
    df['brk_usage'] = df['n_breaking_formatted'] / df['total_pitches'].replace(0, np.nan)
    df['off_usage'] = df['n_offspeed_formatted'] / df['total_pitches'].replace(0, np.nan)

    # Rename stat columns
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

    # Drop rows with missing required values
    df_clean = df.dropna(subset=[
        'arm_angle',
        'fast_vel', 'fast_ivb', 'fast_hb',
        'brk_vel', 'brk_ivb', 'brk_hb',
        'off_vel', 'off_ivb', 'off_hb'
    ])

    return df_clean

def apply_weights(df, manual_weights):
    df = df.copy()
    for prefix in ['fast', 'brk', 'off']:
        usage_col = f'{prefix}_usage'
        for suffix in ['vel', 'ivb', 'hb']:
            stat_col = f'{prefix}_{suffix}'
            if stat_col in df.columns and usage_col in df.columns:
                df[stat_col] = df[stat_col] * df[usage_col] * PITCH_COMPONENT_WEIGHTS[suffix]

    for col, weight in manual_weights.items():
        if col in df.columns:
            df[col] = df[col] * weight

    return df

def normalize(df, stat_cols):
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[stat_cols] = scaler.fit_transform(df_scaled[stat_cols])
    return df_scaled, scaler

def find_similar_euclidean(df_scaled, stat_cols, ref_vector, top_n=10):
    ref_array = np.array([ref_vector[c] for c in stat_cols]).reshape(1, -1)
    distances = pairwise_distances(ref_array, df_scaled[stat_cols].values, metric='euclidean')[0]
    df_scaled = df_scaled.copy()
    df_scaled['distance'] = distances
    df_scaled['similarity_score'] = 1 / (1 + distances)
    return df_scaled.sort_values('distance').head(top_n)

# ---- STREAMLIT APP ----
st.title("⚾ Pitcher Similarity Tool")

# Load CSV
try:
    df = pd.read_csv(CSV_PATH)
except Exception as e:
    st.error(f"Could not load CSV: {e}")
    st.stop()

df = preprocess(df)

# Input mode
input_mode = st.radio("Input Method", ["Select Pitcher", "Enter Custom Stats"])

if input_mode == "Select Pitcher":
    pname = st.selectbox("Select pitcher", df['player_name'].unique())
    ref_row = df[df['player_name'] == pname].iloc[0]
    hand = ref_row['throws']
else:
    hand = st.radio("Handedness", ['R', 'L'])
    st.markdown("Enter your custom stat line below:")
    ref_row = {'player_name': 'Manual', 'throws': hand}
    for stat in STAT_FIELDS:
        ref_row[stat] = st.number_input(stat.replace('_', ' ').title(), value=0.0, step=0.1)

# Filter by same-handed pitchers
df_filtered = df[df['throws'] == hand].reset_index(drop=True)

# Apply usage and manual weights
df_weighted = apply_weights(df_filtered.copy(), MANUAL_WEIGHTS)
df_scaled, scaler = normalize(df_weighted, STAT_FIELDS)

# Normalize custom/manual entry
if input_mode == "Enter Custom Stats":
    ref_df = pd.DataFrame([ref_row])
    ref_weighted = apply_weights(ref_df.copy(), MANUAL_WEIGHTS)
    ref_scaled = ref_weighted.copy()
    ref_scaled[STAT_FIELDS] = scaler.transform(ref_weighted[STAT_FIELDS])
    ref_vector = ref_scaled.iloc[0]
else:
    ref_idx = df_filtered[df_filtered['player_name'] == pname].index[0]
    ref_vector = df_scaled.loc[ref_idx]

# Slider to choose top N
top_n = st.slider("How many similar pitchers to show?", min_value=3, max_value=200, value=10)

# Similarity computation
results_scaled = find_similar_euclidean(df_scaled.copy(), STAT_FIELDS, ref_vector, top_n=top_n)
top_indexes = results_scaled.index
results_original = df_filtered.loc[top_indexes].copy()
results_original['similarity_score'] = results_scaled['similarity_score'].values

# Display
st.subheader("🎯 Top Similar Pitchers")
st.dataframe(
    results_original[['player_name', 'throws', 'similarity_score'] + STAT_FIELDS]
    .style.format({'similarity_score': "{:.3f}"})
)
