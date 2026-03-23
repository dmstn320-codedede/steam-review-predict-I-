# =====================================================
# 📄 ui/card.py (Streamlit Native UI 버전)
# =====================================================

import streamlit as st

def render_game_card(row):

    image_url = f"https://cdn.cloudflare.steamstatic.com/steam/apps/{row['appid']}/header.jpg"
    steam_url = f"https://store.steampowered.com/app/{row['appid']}"

    with st.container():

        st.image(image_url)

        st.markdown(f"### 🎮 [{row['app_name']}]({steam_url})")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("⭐ 점수", round(row['service_score'], 2))

        with col2:
            st.metric("🎯 장르 일치", f"{row['match_score']}개")

        st.divider()