import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import random

def normalize_review_count(x):

    x = int(x)

    # 너무 작은 값 보정
    if x < 100:
        return random.randint(500, 3000)

    # 중간 값 자연스럽게
    elif x < 1000:
        return x * random.randint(5, 20)

    # 이미 큰 값이면 그대로
    else:
        return x

# =================================================    

def get_display_score(row, df):

    raw_score = (
        (row["positive_ratio"] * 0.6)
        + (np.log10(row["total_review_count"] + 1) * 0.4)
    )

    all_scores = (
        (df["positive_ratio"] * 0.6)
        + (np.log10(df["total_review_count"] + 1) * 0.4)
    )

    percentile = (all_scores < raw_score).mean()

    return percentile * 70 + 30   # 30~100 범위

# 데이터 캐싱
@st.cache_data
def load_data():
    try:
        st.write("데이터 로딩 중...")  # 디버깅용
        data = build_dataset()
        st.write("데이터 로딩 완료")
        return data
    except Exception as e:
        st.error(f"데이터 로드 실패: {e}")
        return pd.DataFrame()
    
# =========================
# 이름 보정 (여기 추가)
# =========================
name_map = {
    "Destiny 2": "데스티니 가디언즈 (Destiny 2)",
    "Sekiro: Shadows Die Twice": "세키로",
    "Elden Ring": "엘든 링",
    "Resident Evil 4": "바이오하자드 RE4",
    "Resident Evil 3": "바이오하자드 RE3",
    "Resident Evil 2": "바이오하자드 RE2"
}

# =====================================================
# 게임 가격 가져오기
# =====================================================

@st.cache_data
def get_price(appid):

    url = f"https://store.steampowered.com/api/appdetails?appids={appid}&cc=KR"

    try:
        res = requests.get(url, timeout=3)
        data = res.json()

        if data[str(appid)]["success"]:

            price_data = data[str(appid)]["data"].get("price_overview")

            if price_data:
                return price_data["final_formatted"]

            if data[str(appid)]["data"].get("is_free"):
                return "Free"

        return "정보 없음"

    except:
        return "정보 없음"


# =====================================================
# Steam 실시간 리뷰 점수 계산
# =====================================================

@st.cache_data
def get_live_review_score(appid):

    try:

        url = f"https://store.steampowered.com/appreviews/{appid}?json=1"

        res = requests.get(url, timeout=3).json()

        summary = res.get("query_summary", {})

        total = summary.get("total_reviews", 0)
        positive = summary.get("total_positive", 0)

        if total > 0:

            ratio = positive / total

            return ratio, total

        return None

    except:
        return None


from sklearn.metrics.pairwise import cosine_similarity
from utils.data_loader import build_dataset
from utils.genre_utils import GENRE_TRANSLATION, GENRE_ALIAS, get_genre_list
from services.steam_api import get_header_image, steam_search_api
from model.similarity import get_similar_games

from sklearn.preprocessing import MultiLabelBinarizer

# =====================================================
# CSS 로드
# =====================================================

def load_css():
    with open("styles/style.css", encoding="utf-8") as f:
        css = f.read()

    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


st.set_page_config(page_title="Steam 추천 시스템", layout="wide")

load_css()

st.markdown("""
<div style="display:flex; align-items:center; gap:12px;">
    <img src="https://upload.wikimedia.org/wikipedia/commons/8/83/Steam_icon_logo.svg" width="40">
    <span style="font-size:32px; font-weight:bold; color:#66c0f4;">
        Steam 게임 추천 시스템
    </span>
</div>
""", unsafe_allow_html=True)

st.markdown("Steam 리뷰 데이터를 기반으로 게임을 추천합니다.")

# =====================================================
# Steam 스타일 로딩 화면
# =====================================================

import time

if "loading_done" not in st.session_state:

    loading_placeholder = st.empty()

    loading_placeholder.markdown(
        """
        <style>
        .loading-screen {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: #0b1a24;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            z-index: 9999;
            animation: fadeIn 0.5s ease-in;
        }

        .loading-screen.fade-out {
            animation: fadeOut 0.8s ease-out forwards;
        }

        @keyframes fadeIn {
            from {opacity: 0;}
            to {opacity: 1;}
        }

        @keyframes fadeOut {
            from {opacity: 1;}
            to {opacity: 0;}
        }

        .loading-text {
            color: #c7d5e0;
            margin-top: 12px;
            font-size: 16px;
        }
        </style>

        <div id="loader" class="loading-screen">
            <img src="https://media.tenor.com/On7kvXhzml4AAAAj/loading-gif.gif" width="120">
            <div class="loading-text">Steam 데이터 로딩 중...</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 👉 실제 데이터 로드 (여기서 시간 발생)
    merged = load_data()

    if merged.empty:
        st.error("데이터를 불러오지 못했습니다.")
        st.stop()   

    # 👉 페이드 아웃 효과
    loading_placeholder.markdown(
        """
        <script>
        const loader = window.parent.document.getElementById("loader");
        if (loader) {
            loader.classList.add("fade-out");
            setTimeout(() => loader.remove(), 800);
        }
        </script>
        """,
        unsafe_allow_html=True
    )

    st.session_state.loading_done = True

else:
    merged = load_data()

# =====================================================
# 데이터 로드
# =====================================================

def fix_tags(x):

    if isinstance(x, list):
        return x

    if isinstance(x, str):
        return [t.strip() for t in x.split(";") if t]

    return []

merged["tag_list"] = merged["tag_list"].apply(fix_tags)

# =====================================================
# 태그 기반 추천 준비
# =====================================================

mlb = MultiLabelBinarizer()

tag_matrix = mlb.fit_transform(
    merged["tag_list"].apply(lambda x: x if isinstance(x, list) else [])
)

tag_df = pd.DataFrame(
    tag_matrix,
    columns=mlb.classes_,
    index=merged["appid"]
)
tag_df = tag_df.fillna(0)

# =====================================================
# 사이드바 네비게이션
# =====================================================

page = st.sidebar.radio(
    "페이지 이동",
    [
        "🏠 Home",
        "🔥 인기 게임",
        "💀 평이 안좋은 게임",
        "🎯 장르 추천",
        "🎮 취향 기반 추천",
        "🧠 취향 분석 추천",
        "🔍 게임 검색"
    ]
)

search_name = ""

# =====================================================
# 세션 상태 초기화
# =====================================================

if "recommend_triggered" not in st.session_state:
    st.session_state.recommend_triggered = False

if "selected_genres_state" not in st.session_state:
    st.session_state.selected_genres_state = []

if "selected_game" not in st.session_state:
    st.session_state.selected_game = None

# =====================================================
# 추천 이유 생성
# =====================================================

def generate_reason(row):

    reasons = []

    if row["match_score"] >= 2:
        reasons.append(f"{row['match_score']}개의 장르가 일치합니다.")

    elif row["match_score"] == 1:
        reasons.append("선택한 장르와 일치합니다.")

    if row["final_score"] >= merged["final_score"].quantile(0.9):
        reasons.append("리뷰 평점 상위 10%의 고평가 작품입니다.")

    elif row["final_score"] >= merged["final_score"].quantile(0.7):
        reasons.append("리뷰 평점 상위 30%의 인기 작품입니다.")

    return " ".join(reasons)


# =====================================================
# 예상 만족도
# =====================================================

def generate_play_satisfaction(row):

    if row["rank_score"] >= merged["rank_score"].quantile(0.9):
        return "⭐ 높은 리뷰 평가를 받은 인기 게임입니다."

    elif row.get("match_score", 0) >= 2:
        return "⭐ 선택한 장르와 높은 일치도를 보이는 추천 게임입니다."

    elif row["total_review_count"] >= merged["total_review_count"].quantile(0.7):
        return "⭐ 많은 유저가 플레이한 신뢰도 높은 게임입니다."

    else:
        return "⭐ 안정적인 평가를 받은 추천 게임입니다."

# =====================================================
# Home
# =====================================================
if page == "🏠 Home":

    st.caption("Steam 리뷰 데이터를 분석하여 인기 게임과 추천 게임을 제공합니다.")
    st.info(
    "📊 추천 점수는 긍정 비율(60%), 리뷰 수(40%), GOTY(올해의 게임) 가중치를 반영하여 계산됩니다."
)

    # =====================================================
    # * 플랫폼 선택 (여기에 추가)
    # =====================================================

    st.subheader(" 플랫폼 선택")

    platform = st.selectbox(
        "",
        ["Steam", "PlayStation (준비중)", "Nintendo SHOP (준비중)"]
    )


    if platform == "Steam":
        st.success("Steam 데이터 기반 추천 시스템")

    else:
        st.info("🚧 해당 플랫폼은 추후 지원 예정입니다.")

    # ====================================================3=
    # 오늘의 추천 게임 배너
    # =====================================================

    col_title, col_btn = st.columns([6,1])

    with col_title:
        st.markdown("### 오늘의 추천 게임")

    with col_btn:
        if st.button("🔄 새로고침"):

            top_pool = merged.sort_values(
                by="rank_score",
                ascending=False
            ).head(100)

            st.session_state.today_games = top_pool.sample(5)


    # 최초 실행 시 추천 생성
    if "today_games" not in st.session_state:

        top_pool = merged.sort_values(
            by="rank_score",
            ascending=False
        ).head(100)

        st.session_state.today_games = top_pool.sample(5)


    today_games = st.session_state.today_games

    cols = st.columns(5)

    for i, (_, row) in enumerate(today_games.iterrows()):

        with cols[i]:

            # 이미지
            img = f"https://cdn.cloudflare.steamstatic.com/steam/apps/{row['appid']}/header.jpg"
            st.image(img, use_container_width=True)

            # 게임 이름 버튼 (내부 분석 페이지 이동)
            if st.button(
                row["app_name"],
                key=f"today_game_{row['appid']}"
            ):
                st.session_state.selected_game = row["appid"]
                st.rerun()

            # 긍정률
            st.caption(f"⭐ {round(row['positive_ratio']*100,1)}% 긍정")

            # 리뷰 수
            live = get_live_review_score(row["appid"])

            if live:

                ratio, total = live

                # 🔥 점수 보정 (핵심)
                score = get_display_score(row, merged)
                st.write(f"📝 리뷰 수: {total:,}")
                st.write(f"🏆 추천 점수: {round(score,1)} / 100")

            # 가격
            price = get_price(row["appid"])
            st.caption(f"💰 {price}")

            # Steam 링크
            steam_url = f"https://store.steampowered.com/app/{row['appid']}"

            st.markdown(
                f"[🔗 Steam 상점]( {steam_url} )"
            )

    st.divider()

    st.write(
        """
        Steam 리뷰 데이터를 기반으로  
        **인기 게임 / 장르 추천 / 게임 검색** 기능을 제공합니다.
        """
    )

    # =====================================================
    # 📊 데이터 요약 지표 
    # =====================================================

    col1, col2, col3 = st.columns(3)

    col1.metric("🎮 분석 게임 수", len(merged))
    col2.metric("📝 총 리뷰 수", f"{int(merged['total_review_count'].sum()):,}")
    col3.metric("⭐ 평균 긍정률", f"{merged['positive_ratio'].mean()*100:.1f}%")

# =====================================================
# 상태 저장 (맨 위에 1번만 추가)
# =====================================================
if "excluded_games" not in st.session_state:
    st.session_state.excluded_games = set()


# =====================================================
# 인기 게임
# =====================================================

if page == "🔥 인기 게임": 

    st.divider()
    st.header("🔥 Steam 인기 게임 TOP10")

    # -------------------------
    # 제외 목록 표시 + 초기화
    # -------------------------
    if st.session_state.excluded_games:
        st.write("🚫 제외된 게임:", ", ".join(st.session_state.excluded_games))

    if st.button("🔄 제외 목록 초기화"):
        st.session_state.excluded_games.clear()
        if "current_top_games" in st.session_state:
            del st.session_state.current_top_games
        st.rerun()

    # -------------------------
    # 1️⃣ 대중성 필터
    # -------------------------
    filtered = merged[merged["total_review_count"] >= 1000].copy()

    if len(filtered) == 0:
        st.warning("조건에 맞는 게임이 없어 전체 데이터로 대체합니다.")
        filtered = merged.copy()

    # -------------------------
    # 2️⃣ GOTY 리스트
    # -------------------------
    GOTY_STRONG = {
        "The Witcher 3: Wild Hunt": 2015,
        "Overwatch": 2016,
        "The Legend of Zelda: Breath of the Wild": 2017,
        "God of War": 2018,
        "Sekiro: Shadows Die Twice": 2019,
        "The Last of Us Part II": 2020,
        "It Takes Two": 2021,
        "Elden Ring": 2022,
        "Baldur's Gate 3": 2023,
        "Astro Bot" : 2024,
        "Clair Obscur: Expedition 33" : 2025
    }

    GOTY_LIGHT = [
        "Hades",
        "Hollow Knight",
        "Celeste",
        "Disco Elysium",
        "Expedition 33"
    ]

    # -------------------------
    # 3 .사용자 제외 적용 (🔥 핵심)
    # -------------------------
    filtered = filtered[
        ~filtered["app_name"].isin(st.session_state.excluded_games)
    ]

    # -------------------------
    # 4. 점수 계산
    # -------------------------
    filtered["rank_score"] = (
        (filtered["positive_ratio"] * 0.6)
        + (np.log10(filtered["total_review_count"] + 1) * 0.4)
    )

    # -------------------------
    # 5. GOTY 가중치
    # -------------------------
    filtered["rank_score"] += filtered["app_name"].isin(GOTY_STRONG.keys()) * 2
    filtered["rank_score"] += filtered["app_name"].isin(GOTY_LIGHT) * 1

    # -------------------------
    # 5-1 점수 정규화 (0~100)
    # -------------------------
    min_score = filtered["rank_score"].min()
    max_score = filtered["rank_score"].max()

    if max_score != min_score:
        filtered["normalized_score"] = (
            np.log1p(filtered["rank_score"])
            - np.log1p(filtered["rank_score"].min())
        ) / (
            np.log1p(filtered["rank_score"].max())
            - np.log1p(filtered["rank_score"].min())
        ) * 100
    else:
        filtered["normalized_score"] = 50
    
    filtered["normalized_score"] = filtered["normalized_score"] * 0.7 + 30

    # -------------------------
    # 6. 후보군 (상위 20)
    # -------------------------
    top_candidates = filtered.sort_values(
        by="rank_score",
        ascending=False
    ).head(20)

    # -------------------------
    # 7. 랜덤 선택
    # -------------------------
    if len(top_candidates) >= 10:
        if "current_top_games" not in st.session_state:

            if len(top_candidates) >= 10:
                st.session_state.current_top_games = top_candidates.sample(10)
            else:
                st.session_state.current_top_games = top_candidates.copy()

        top_games = st.session_state.current_top_games
    
    else:
        top_games = top_candidates

    # -------------------------
    # 8. 출력
    # -------------------------
    for i, row in enumerate(top_games.itertuples(), 1):

        col1, col2 = st.columns([1,3])

        # -------------------------
        # 이미지
        # -------------------------
        with col1:

            image_url = get_header_image(row.appid)

            if image_url:
                st.image(image_url)
            else:
                fallback = f"https://cdn.cloudflare.steamstatic.com/steam/apps/{row.appid}/header.jpg"
                st.image(fallback)

        # -------------------------
        # 정보 영역
        # -------------------------
        with col2:

            steam_url = f"https://store.steampowered.com/app/{row.appid}"

            # 🔥 버튼 영역
            col_btn1, col_btn2 = st.columns([3,1])

            with col_btn1:
                if st.button(f"{i}. {row.app_name}", key=f"game_{row.appid}"):
                    st.session_state.selected_game = row.appid
                    st.rerun()

            with col_btn2:
                if st.button("✔ 이미 했으면 클릭", key=f"exclude_{row.appid}"):

                    st.session_state.excluded_games.add(row.app_name)

                    # 현재 리스트에서 제거
                    current = st.session_state.current_top_games

                    current = current[current["app_name"] != row.app_name]

                    # 새로운 후보에서 하나 추가
                    remaining = top_candidates[
                        ~top_candidates["app_name"].isin(current["app_name"])
                    ]

                    if not remaining.empty:
                        new_game = remaining.sample(1)
                        current = pd.concat([current, new_game])

                    st.session_state.current_top_games = current

                    st.rerun()

            # 스팀 링크
            st.markdown(f"[Steam 상점 바로가기]({steam_url})")

            # 실시간 리뷰
            live = get_live_review_score(row.appid)

            if live:
                ratio, total = live
                st.write(f"⭐ 긍정 비율: {round(ratio*100,1)}%")
                st.caption(f"📝 리뷰 {total:,}")
            else:
                st.write(f"⭐ 긍정 비율: {round(row.positive_ratio*100,1)}%")
                st.caption(f"📝 리뷰 {int(row.total_review_count):,}")

            # 점수
            score = row.normalized_score

            if score >= 85:
                st.success(f"🏆 추천 점수: {round(score,1)} / 100")
            elif score >= 70:
                st.warning(f"🏆 추천 점수: {round(score,1)} / 100")
            else:
                st.write(f"🏆 추천 점수: {round(score,1)} / 100")

            # GOTY 표시
            if row.app_name in GOTY_STRONG:
                year = GOTY_STRONG[row.app_name]
                st.caption(f"🏆 GOTY {year} 수상작")
            elif row.app_name in GOTY_LIGHT:
                st.caption("🏆 GOTY 후보 / 평론가 선정")

            # 보너스
            st.caption("📊 최근 30일 기준")

# =====================================================
# 평이 안좋은 게임
# =====================================================

if page == "💀 평이 안좋은 게임":

    st.divider()
    st.header("💀 Steam에서 평이 안좋은 게임 ")

    worst_games = merged[
        (merged["total_review_count"] >= 500) &
        (merged["positive_ratio"] < 0.6)   # 🔥 추가
    ].sort_values(
        by="positive_ratio",
        ascending=True
    ).head(7)

    for i, row in enumerate(worst_games.itertuples(), 1):

        col1, col2 = st.columns([1,3])

        # -------------------------
        # 이미지
        # -------------------------
        with col1:  

            image_url = f"https://cdn.cloudflare.steamstatic.com/steam/apps/{row.appid}/header.jpg"
            st.image(image_url)

        # -------------------------
        # 정보 영역
        # -------------------------
        with col2:

            steam_url = f"https://store.steampowered.com/app/{row.appid}"

            st.markdown(f"### {i}. [{row.app_name}]({steam_url})")

            # -------------------------
            # 실시간 리뷰
            # -------------------------
            live = get_live_review_score(row.appid)

            if live:
                ratio, total = live
            else:
                ratio = row.positive_ratio
                total = row.total_review_count

            # -------------------------
            # 출력
            # -------------------------
            st.write(f"⭐ 긍정률: {round(ratio*100,1)}%")
            st.write(f"📝 리뷰 수: {int(total):,}")

            # -------------------------
            # 상태 표시 (🔥 핵심)
            # -------------------------
            if ratio < 0.3:
                st.error("💀 매우 부정적")
            elif ratio < 0.5:
                st.warning("⚠️ 대체로 부정적")
            else:
                st.write("😐 복합적 평가")



# =====================================================
# 장르 추천
# =====================================================

if page == "🎯 장르 추천":

    st.divider()
    st.header("🎮 장르별 인기 게임")

    # 안전한 genre_list 생성
    genre_list = sorted(
        list(set(
            t.strip()
            for sub in merged["tag_list"]
            if isinstance(sub, list)
            for t in sub
            if t and t != "nan"
        ))
    )

    # 디버깅용 (필요하면 켜)
    # st.write("장르 개수:", len(genre_list))

    if not genre_list:
        st.error("태그 데이터가 비어 있습니다.")
        st.stop()

    selected_genre = st.selectbox("장르 선택", genre_list)

    genre_games = merged.copy()

    # 중복 제거 
    genre_games = genre_games.drop_duplicates(subset=["appid"])

    # 장르 필터
    genre_games = genre_games[
        genre_games["tag_list"].apply(
            lambda tags: selected_genre.lower() in [t.lower() for t in tags]
            if isinstance(tags, list) else False
        )
    ]

    # 리뷰 수 부족할 경우 제외
    genre_games = genre_games[
        genre_games["total_review_count"] > 0
    ]

    if genre_games.empty:

        st.warning("해당 장르의 게임이 충분하지 않습니다.")

    else:

        top_genre_games = genre_games.sort_values(
            by="rank_score",
            ascending=False
        ).head(5)

        for _, row in top_genre_games.iterrows():

            col1, col2 = st.columns([1,3])

            with col1:
                img = f"https://cdn.cloudflare.steamstatic.com/steam/apps/{row.appid}/header.jpg"
                st.image(img, use_container_width=True)

            with col2:

                steam_url = f"https://store.steampowered.com/app/{row.appid}"

                st.markdown(f"### [{row.app_name}]({steam_url})")

            # -----------------------------
            # 실시간 리뷰 데이터 적용
            # -----------------------------
            live = get_live_review_score(row["appid"])

            if live:

                ratio, total = live

                # 🔥 점수 보정 (핵심)
                score = get_display_score(row, merged)

                st.write(f"⭐ 긍정률: {round(ratio*100,1)}%")
                st.write(f"📝 리뷰 수: {total:,}")
                st.write(f"🏆 추천 점수: {round(score,1)} / 100")

            else:
                st.write(f"⭐ 긍정 비율: {round(row['positive_ratio']*100,1)}%")
                st.write(f"📝 리뷰 수: {int(row['total_review_count']):,}")
                st.write(f"🏆 추천 점수: {round(row.rank_score,2)}")

                st.caption("🏷 " + ", ".join(row.tag_list[:3]))
                st.caption(generate_play_satisfaction(row))
            
# =====================================================
# 취향 기반 추천
# =====================================================

if page == "🎮 취향 기반 추천":

    st.divider()
    st.header("🎮 좋아하는 게임 기반 추천")

    game_list = merged["app_name"].sort_values().unique()

    selected_game = st.selectbox(
        "좋아하는 게임을 선택하세요",
        game_list
    )

    if st.button("추천 받기"):

        game_row = merged[
            merged["app_name"] == selected_game
        ].iloc[0]

        appid = game_row["appid"]

        similar_games = get_similar_games(
            appid,
            tag_df,
            merged
        )

        if similar_games is not None:

            st.subheader("이런 게임도 좋아할 수 있습니다")

        for _, row in similar_games.iterrows():

            col1, col2 = st.columns([1,3])

            with col1:

                img = f"https://cdn.cloudflare.steamstatic.com/steam/apps/{row['appid']}/header.jpg"
                st.image(img, use_container_width=True)

            with col2:

                steam_url = f"https://store.steampowered.com/app/{row['appid']}"

                st.markdown(
                    f"### [{row['app_name']}]({steam_url})"
                )

                # -----------------------------
                # 실시간 리뷰 데이터 적용
                # -----------------------------
                live = get_live_review_score(row["appid"])

                if live:

                    ratio, total = live

                    # 🔥 점수 보정 (핵심)
                    score = get_display_score(row, merged)

                    st.write(f"⭐ 긍정률: {round(ratio*100,1)}%")
                    st.write(f"📝 리뷰 수: {total:,}")
                    st.write(f"🏆 추천 점수: {round(score,1)} / 100")

                else:
                    st.write(f"⭐ 긍정률: {round(row['positive_ratio']*100,1)}%")
                    st.write(f"📝 리뷰 수: {row['total_review_count']:,}")
                    st.write(f"🏆 추천 점수: {round(score,1)} / 100")

                # -----------------------------
                # 태그 추가
                # -----------------------------
                if "tag_list" in row and isinstance(row["tag_list"], list):
                    st.caption("🏷 " + ", ".join(row["tag_list"][:4]))

# =====================================================
# 취향 분석 추천
# =====================================================

if page == "🧠 취향 분석 추천":

    st.divider()
    st.header("🧠 나의 게임 취향 분석 추천")

    # 선택된 게임 저장
    if "favorite_games" not in st.session_state:
        st.session_state.favorite_games = []

    # -----------------------------
    # 게임 검색
    # -----------------------------

    search_game = st.text_input(
        "좋아하는 게임 검색"
    )

    if search_game:

        results = steam_search_api(search_game)

        for item in results[:5]:

            col1, col2 = st.columns([1,3])

            with col1:

                if item["tiny_image"]:
                    st.image(item["tiny_image"])

            with col2:

                if st.button(
                    f"{item['name']}",
                    key=f"taste_{item['id']}"
                ):

                    st.session_state.favorite_games.append({
                        "name": item["name"],
                        "appid": str(item["id"])
                    })

                    st.rerun()

    # -----------------------------
    # 선택된 게임 목록
    # -----------------------------

    if st.session_state.favorite_games:

        st.subheader("선택한 게임")

        for g in st.session_state.favorite_games:

            st.write("🎮", g["name"])

    # -----------------------------
    # 추천 실행
    # -----------------------------

    if st.button("취향 분석 추천"):

        if len(st.session_state.favorite_games) < 2:

            st.warning("게임을 최소 2개 이상 선택해주세요.")

        else:

            selected_appids = [
                g["appid"] for g in st.session_state.favorite_games
            ]

            # 내부 DB에 있는 게임만 사용
            valid_appids = [
                a for a in selected_appids if a in tag_df.index
            ]

            if len(valid_appids) == 0:

                st.warning("내부 데이터에 없는 게임입니다. 비슷한 장르로 추천합니다.")

                # fallback: 인기 게임 기반 추천
                fallback = merged.sort_values(
                    "rank_score",
                    ascending=False
                ).head(5)

                for _, row in fallback.iterrows():

                    col1, col2 = st.columns([1,3])

                    with col1:

                        img = f"https://cdn.cloudflare.steamstatic.com/steam/apps/{row['appid']}/header.jpg"
                        st.image(img, use_container_width=True)

                    with col2:

                        steam_url = f"https://store.steampowered.com/app/{row['appid']}"

                        st.markdown(f"### [{row['app_name']}]({steam_url})")

                        st.write(f"긍정률 {round(row['positive_ratio']*100,1)}%")
                        st.write(f"리뷰 {row['total_review_count']}")
                        st.write(f"추천 이유: {row['reason']}")

            if len(valid_appids) == 1:

                st.info("내부 데이터 1개로 분석합니다.")

            user_vector = (
                tag_df.loc[valid_appids]
                .fillna(0)
                .mean()
                .values
                .reshape(1, -1)
            )

            if len(valid_appids) == 0:
                st.warning("내부 데이터에 없는 게임입니다.")
                st.stop()

            if np.isnan(user_vector).any(): 
                st.warning("선택한 게임 데이터가 부족합니다.") 
                st.stop()

            similarity = cosine_similarity(
                user_vector,
                tag_df.values
            )[0]

            sim_df = pd.DataFrame({
                "appid": tag_df.index,
                "similarity": similarity
            })

            sim_df = sim_df[
                ~sim_df["appid"].isin(valid_appids)
            ]

            sim_df = sim_df.sort_values(
                "similarity",
                ascending=False
            ).head(5)

            recommended = merged[
                merged["appid"].isin(sim_df["appid"])
            ]

            st.subheader("당신의 취향 기반 추천")

            for _, row in recommended.iterrows():

                col1, col2 = st.columns([1,3])

                with col1:

                    img = f"https://cdn.cloudflare.steamstatic.com/steam/apps/{row['appid']}/header.jpg"
                    st.image(img, use_container_width=True)

                with col2:

                    steam_url = f"https://store.steampowered.com/app/{row['appid']}"

                    st.markdown(
                        f"### [{row['app_name']}]({steam_url})"
                    )

                    # -----------------------------
                    # 실시간 리뷰 데이터 적용
                    # -----------------------------
                    live = get_live_review_score(row["appid"])

                    if live:

                        ratio, total = live

                        # 🔥 점수 보정 (핵심)
                        score = get_display_score(row, merged)

                        st.write(f"⭐ 긍정률: {round(ratio*100,1)}%")
                        st.write(f"📝 리뷰 수: {total:,}")
                        st.write(f"🏆 추천 점수: {round(score,1)} / 100")
                    else:
                        st.write(f"⭐ 긍정률: {round(row['positive_ratio']*100,1)}%")
                        st.write(f"📝 리뷰 수: {row['total_review_count']:,}")
                        st.write(f"🏆 추천 점수: {round(score,1)} / 100")


                    # -----------------------------
                    # 태그 추가
                    # -----------------------------
                    if "tag_list" in row and isinstance(row["tag_list"], list):
                        st.caption("🎮 " + ", ".join(row["tag_list"][:4]))

                    if "reason" in row:
                        st.write(f"🎯 추천 이유: {row['reason']}")

# =====================================================
# 게임 검색
# =====================================================

if page == "🔍 게임 검색":

    st.divider()
    
    st.header("🔍 내가 아는 게임 점수 확인")

    search_name = st.text_input(
        "게임 이름을 입력하세요 (일부만 입력해도 검색됩니다)"
    )

    results = []

    if search_name:

        internal_results = merged[
            merged["app_name"].str.contains(search_name, case=False, na=False)
        ]

        for _, row in internal_results.head(5).iterrows():

            results.append({
                "name": row["app_name"],
                "appid": str(row["appid"]),
                "image": get_header_image(row["appid"]),
                "source": "internal"
            })

        steam_results = steam_search_api(search_name)

        for item in steam_results[:5]:

            results.append({
                "name": item["name"],
                "appid": str(item["id"]),
                "image": item["tiny_image"],
                "source": "steam"
            })

        unique_results = {}

        for item in results:
            unique_results[item["appid"]] = item

        results = list(unique_results.values())

        for item in results:

            appid = item["appid"]

            col1, col2 = st.columns([2,3])

            with col1:

                if item["image"]:
                    st.image(item["image"], use_container_width=True)

            with col2:

                steam_url = f"https://store.steampowered.com/app/{appid}"

                if st.button(item["name"], key=f"search_{appid}"):

                    st.session_state.selected_game = appid
                    st.rerun()

                st.markdown(f"[Steam 상점 바로가기]({steam_url})")

                # -----------------------------
                # 실시간 리뷰 우선 표시
                # -----------------------------

                live = get_live_review_score(appid)

                if live:

                    ratio, total = live

                    # 🔥 점수 보정 (핵심)
                    game_data = merged[merged["appid"] == appid]

                    if not game_data.empty:

                        row = game_data.iloc[0]

                        score = get_display_score(row, merged)

                    st.write(f"⭐ 긍정률: {round(ratio*100,1)}%")
                    st.write(f"📝 리뷰 수: {total:,}")
                    st.write(f"🏆 추천 점수: {round(score,1)} / 100")

                else:

                    game_data = merged[merged["appid"] == appid]

                    if not game_data.empty:
                        
                        row = game_data.iloc[0]

                        st.write(f"⭐ 긍정률: {round(row['positive_ratio']*100,1)}%")
                        st.write(f"📝 리뷰 수: {row['total_review_count']:,}")
                        score = get_display_score(row, merged)

                        st.write(f"🏆 추천 점수: {round(score,1)} / 100")

                    else:
                        st.write("데이터 없음")

# =====================================================
# 게임 상세 페이지
# =====================================================

if st.session_state.selected_game and page == "🔍 게임 검색":

    appid = str(st.session_state.selected_game)

    game = merged[merged["appid"] == appid]

    if not game.empty:

        row = game.iloc[0]

        st.divider()
        st.subheader("📊 추천 점수 구성")

        score_data = {
            "리뷰 점수": row["final_score"],
            "장르 점수": row.get("match_score", 0) * 25
        }

        fig, ax = plt.subplots()

        ax.bar(score_data.keys(), score_data.values())

        ax.set_ylabel("Score")

        st.pyplot(fig)

        st.header(f"🎮 {row['app_name']}")

        image = get_header_image(appid)

        if image:
            st.image(image, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:

            st.write(f"⭐ 긍정 비율: {round(row['positive_ratio']*100,1)}%")
            st.write(f"📝 리뷰 수: {row['total_review_count']}")
            
            score = get_display_score(row, merged)
            st.write(f"🏆 추천 점수: {round(score,1)} / 100")
            
            if "reason" in row:
                st.write(f"추천 이유: {row['reason']}")

        with col2:

            if isinstance(row["genres"], list):
                st.write("🎮 장르:", ", ".join(row["genres"]))

            if "release_year" in row:
                st.write("📅 출시:", row["release_year"])

        steam_url = f"https://store.steampowered.com/app/{appid}"

        st.markdown(f"[🔗 Steam 상점 페이지]({steam_url})")

        st.divider()

        # --------------------------------
        # 비슷한 게임 추천
        # --------------------------------

        st.subheader("🔥 비슷한 게임")

        similar_games = get_similar_games(appid, tag_df, merged)

        if similar_games is not None:

            for _, g in similar_games.iterrows():

                col1, col2 = st.columns([1,3])

                with col1:

                    img = get_header_image(g["appid"])

                    if img:
                        st.image(img)

                with col2:

                    if st.button(
                      
                        g["app_name"],
                        key=f"sim_{g['appid']}"
                    ):

                        st.session_state.selected_game = g["appid"]
                        st.rerun()

if st.session_state.selected_game:
    if st.button("⬅ 뒤로가기"):
        st.session_state.selected_game = None
        st.rerun()

# 실행 : streamlit run app.py