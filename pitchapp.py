import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# Load the long-format cleaned data
@st.cache_data
def load_data():
    df = pd.read_csv("stats (2).csv")
    df.rename(columns={"last_name, first_name": "player_name"}, inplace=True)

    pitch_types = ["ff", "sl", "ch", "cu", "si", "fc", "fs", "st", "sv"]
    stat_suffixes = ["avg_speed", "avg_spin", "avg_break_x", "avg_break_z_induced"]

    pitch_rows = []
    for _, row in df.iterrows():
        for pt in pitch_types:
            prefix = f"{pt}_"
            pitch_data = {
                "player_name": row["player_name"],
                "player_id": row["player_id"],
                "year": row["year"],
                "pitch_hand": row["pitch_hand"],
                "pitch_type": pt.upper(),
                "arm_angle": row["arm_angle"]
            }
            has_data = False
            for suffix in stat_suffixes:
                col = f"{prefix}{suffix}"
                if col in row:
                    val = row[col]
                    pitch_data[suffix] = val
                    if pd.notna(val):
                        has_data = True
            if has_data:
                pitch_rows.append(pitch_data)

    df_long = pd.DataFrame(pitch_rows)
    features = ["avg_speed", "avg_break_x", "avg_break_z_induced", "avg_spin", "arm_angle"]
    df_long = df_long.dropna(subset=features)
    return df_long

df_long = load_data()

# Available features and their default weights
all_features = {
    "avg_speed": 1.5,
    "avg_break_x": 1.5,
    "avg_break_z_induced": 1.5,
    "avg_spin": 1.0,
    "arm_angle": 1.0,
}

# Sidebar controls
st.sidebar.title("Pitch Comparison Tool")
mode = st.sidebar.radio("Select Mode", ["Pitcher Lookup", "Manual Input"])
same_hand_only = st.sidebar.checkbox("Same Handedness Only", value=True)
num_results = st.sidebar.slider("# of Similar Pitches", 1, 20, 5)

st.sidebar.markdown("### Select Stats to Include")
enabled_features = []
for feat in all_features.keys():
    if st.sidebar.checkbox(f"Use {feat}", value=True):  # default checked
        enabled_features.append(feat)

if not enabled_features:
    st.error("⚠️ Please select at least one feature.")
    st.stop()

weights = {f: all_features[f] for f in enabled_features}

# Normalize features with weights
scaler = StandardScaler()
X = df_long[enabled_features]
X_scaled = scaler.fit_transform(X) * [weights[f] for f in enabled_features]

def get_similar_pitches(input_vector, input_hand=None):
    input_scaled = scaler.transform([input_vector]) * [weights[f] for f in enabled_features]
    sims = cosine_similarity(input_scaled, X_scaled)[0]
    df_long_copy = df_long.copy()
    df_long_copy["similarity"] = sims
    if same_hand_only and input_hand is not None:
        df_long_copy = df_long_copy[df_long_copy["pitch_hand"] == input_hand]
    return df_long_copy.sort_values("similarity", ascending=False).head(num_results)

if mode == "Pitcher Lookup":
    all_names = sorted(df_long["player_name"].unique())
    selected_name = st.selectbox("Choose a pitcher", all_names)
    matches = df_long[df_long["player_name"] == selected_name]
    pitch_options = matches["pitch_type"].unique()
    selected_pitch = st.selectbox("Select Pitch Type", pitch_options)
    selected = matches[matches["pitch_type"] == selected_pitch].iloc[0]
    st.write("### Selected Pitch Stats")
    st.write(selected[enabled_features])
    results = get_similar_pitches(selected[enabled_features].values, selected["pitch_hand"])
    st.write("### Similar Pitches")
    st.dataframe(results[["player_name", "pitch_type", "pitch_hand"] + enabled_features + ["similarity"]])

else:
    st.write("### Manually Enter Pitch Stats")
    manual_input = {}
    for f in enabled_features:
        manual_input[f] = st.number_input(f, value=float(df_long[f].mean()), step=0.1)
    hand = st.selectbox("Pitcher Handedness", ["R", "L"])
    results = get_similar_pitches([manual_input[f] for f in enabled_features], hand)
    st.write("### Similar Pitches")
    st.dataframe(results[["player_name", "pitch_type", "pitch_hand"] + enabled_features + ["similarity"]])
