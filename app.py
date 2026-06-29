import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Configure the web page appearance
st.set_page_config(page_title="2026 World Cup Predictor", page_icon="⚽", layout="centered")

st.title("⚽ 2026 World Cup Match Predictor")
st.markdown("Select two countries to calculate dynamic simulation match probabilities based on team discipline, historical goals, and tournament experience.")

# 1. Load exported model assets with optimization caching
@st.cache_resource
def load_app_assets():
    assets = joblib.load('world_cup_model_assets.pkl')
    teams = joblib.load('teams_2026.pkl')
    return assets, teams

try:
    assets, teams_2026 = load_app_assets()
except FileNotFoundError:
    st.error("Missing model assets! Make sure 'world_cup_model_assets.pkl' and 'teams_2026.pkl' are exported in this directory.")
    st.stop()

# 2. Extract the exact host list pulled dynamically from your datasets
dataset_hosts = assets.get('all_hosts_list', ["USA", "Mexico", "Canada"])
if "Unknown" not in dataset_hosts:
    dataset_hosts.append("Unknown")

# 3. Sidebar Match Settings Context using the dynamic host list
st.sidebar.header("🗺️ Match Settings")
host_country = st.sidebar.selectbox(
    "Host Country Context", 
    options=dataset_hosts, 
    index=dataset_hosts.index("USA") if "USA" in dataset_hosts else 0
)
round_name = st.sidebar.selectbox(
    "Tournament Stage", 
    options=["Group stage", "Round of 32", "Round of 16", "Quarter-finals", "Semi-finals", "Final"], 
    index=0
)

# 4. Main UI Team Dropdowns
col1, col2 = st.columns(2)

with col1:
    home_team = st.selectbox("Home Team", options=teams_2026, index=teams_2026.index("United States") if "United States" in teams_2026 else 0)

with col2:
    # Filter list to prevent a team from playing against itself
    selectable_away = [team for team in teams_2026 if team != home_team]
    away_team = st.selectbox("Away Team", options=selectable_away, index=0)

st.markdown("---")

# 5. Trigger Analysis Button
if st.button("Run Match Simulation", type="primary", use_container_width=True):
    with st.spinner("Processing preprocessing transformers and running inference vectors..."):
        
        team_to_id = assets['team_to_id']
        h_id = team_to_id.get(home_team, -1)
        a_id = team_to_id.get(away_team, -1)
        
        # Build 1-row DataFrame mimicking the original notebook input mapping pattern completely
        match_data = pd.DataFrame({
            'home_xg': [assets['median_xg']],  
            'away_xg': [assets['median_xg']],
            'home_team_id': [h_id],
            'home_manager': ['Unknown'], 
            'away_manager': ['Unknown'],
            'away_team_id': [a_id],
            
            'home_avg_goals': [assets['team_avg_goals_dict'].get(home_team, assets['global_avg_goals'])],
            'away_avg_goals': [assets['team_avg_goals_dict'].get(away_team, assets['global_avg_goals'])],
            'home_wc_experience': [assets['team_wc_exp_dict'].get(home_team, 0)],
            'away_wc_experience': [assets['team_wc_exp_dict'].get(away_team, 0)],
            'home_penalty_exp': [assets['team_pen_exp_dict'].get(home_team, 0)],
            'away_penalty_exp': [assets['team_pen_exp_dict'].get(away_team, 0)],
            'home_discipline_score': [assets['team_disc_score_dict'].get(home_team, assets['global_disc_score'])],
            'away_discipline_score': [assets['team_disc_score_dict'].get(away_team, assets['global_disc_score'])],
            
            'Attendance': [assets['median_attendance']],
            'Venue_City': ['Unknown'], 
            'Round': [round_name],
            'Host': [host_country],
            'Year': [2026],
            'Referee': ['Unknown'],
            
            'home_is_host': [1 if home_team == host_country else 0],
            'away_is_host': [1 if away_team == host_country else 0],
            'home_adrenaline_boost': [1 if home_team == host_country else 0],
            'away_adrenaline_boost': [1 if away_team == host_country else 0]
        })
        
        # Transform data through your notebook's native column preprocessor
        processed_data = assets['preprocessor'].transform(match_data)
        
        # Run classification mapping inference using XGBoost
        probs = assets['model'].predict_proba(processed_data)[0]
        
        home_win_pct = probs[0] * 100
        away_win_pct = probs[1] * 100
        draw_pct = probs[2] * 100

    # 6. Render Beautiful Metric Interface Components
    st.subheader("🎯 Predicted Probabilities")
    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric(label=f"🏠 {home_team} Win", value=f"{home_win_pct:.1f}%")
    m_col2.metric(label=f"🤝 Draw", value=f"{draw_pct:.1f}%")
    m_col3.metric(label=f"🚀 {away_team} Win", value=f"{away_win_pct:.1f}%")
    
    # Progress Bar UI Visuals
    st.markdown("### Match Distribution Summary")
    st.progress(int(home_win_pct))
    st.caption(f"Visualized base weight favoring {home_team} (left side metric dominance).")