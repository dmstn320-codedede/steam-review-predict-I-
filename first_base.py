# =====================================================
# app.py
# Steam 장르 기반 추천 시스템 
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import glob
import os
import re
import requests

# =====================================================
# CSS 로드
# =====================================================

def load_css():
    with open("styles/style.css", encoding="utf-8") as f:
        css = f.read()

    st.markdown(
        f"<style>{css}</style>",
        unsafe_allow_html=True
    )

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer

st.set_page_config(page_title="Steam 추천 시스템", layout="wide")

load_css()

st.markdown('<div class="hero-title">🎮 Steam 게임 추천 시스템</div>', unsafe_allow_html=True)

st.markdown(
"Steam 리뷰 데이터를 기반으로 게임을 추천합니다."
)

# =====================================================
# 사이드바 네비게이션
# =====================================================

st.sidebar.title("🎮 Steam 추천 시스템")

page = st.sidebar.radio(
    "페이지 이동",
    [
        "🏠 Home",
        "🔥 인기 게임",
        "🎯 장르 추천",
        "🔍 게임 검색"
    ]
)

search_name = ""

# =====================================================
# 세션 상태 초기화
# =====================================================

if "excluded_games" not in st.session_state:
    st.session_state.excluded_games = []

if "recommend_triggered" not in st.session_state:
    st.session_state.recommend_triggered = False

if "selected_genres_state" not in st.session_state:
    st.session_state.selected_genres_state = []

# =====================================================
# 테마 설정
# =====================================================

if "theme" not in st.session_state:
    st.session_state.theme = "dark"

st.sidebar.markdown("### 🎨 테마 설정")

theme = st.sidebar.radio(
    "테마 선택",
    ["Dark", "Light"]
)

if theme == "Dark":
    st.session_state.theme = "dark"
else:
    st.session_state.theme = "light"

# =====================================================
# 테마 적용
# =====================================================

if st.session_state.theme == "dark":
    st.markdown('<div class="dark-mode">', unsafe_allow_html=True)
else:
    st.markdown('<div class="light-mode">', unsafe_allow_html=True)

# =====================================================
# 1. 데이터 로드
# =====================================================

@st.cache_data
def load_review_data():
    file_list = glob.glob("data/recent_reviews/*.xlsx")

    if not file_list:
        st.error("data/recent_reviews 폴더에서 xlsx 파일을 찾지 못했습니다.")
        st.stop()

    review_list = []

    for file in file_list:
        try:
            temp = pd.read_excel(file)
            filename = os.path.basename(file)
            numbers = re.findall(r"\d+", filename)

            if numbers:
                temp["appid"] = str(numbers[-1])

            review_list.append(temp)
        except:
            continue

    review_df = pd.concat(review_list, ignore_index=True)

    if "voted_up" in review_df.columns:
        review_df = review_df.rename(columns={"voted_up": "recommended"})

    review_df["recommended"] = review_df["recommended"].astype(int)
    review_df["appid"] = review_df["appid"].astype(str)

    return review_df


review_df = load_review_data()

# =====================================================
# 2. 점수 계산
# =====================================================

score_df = review_df.groupby("appid").agg(
    positive_ratio=("recommended", "mean"),
    total_review_count=("recommended", "count")
).reset_index()

score_df["final_score"] = (
    score_df["positive_ratio"] * 70
    + np.log1p(score_df["total_review_count"]) * 5
)

score_df["appid"] = score_df["appid"].astype(str)

# =====================================================
# 3. 메타데이터 로드
# =====================================================

@st.cache_data
def load_game_metadata():

    df = pd.read_csv("data/game_metadata_fixed.csv")

    # -------------------------------------------------
    # steamspy_tags 컬럼 존재 여부 확인
    # -------------------------------------------------

    if "steamspy_tags" in df.columns:

        df["tag_list"] = df["steamspy_tags"].fillna("").apply(
            lambda x: x.split(";")
        )

    else:

        # steamspy_tags 없는 경우 빈 태그 생성
        df["tag_list"] = [[] for _ in range(len(df))]

    # -------------------------------------------------
    # 장르 처리
    # -------------------------------------------------

    import ast
    df["genres"] = df["genres"].apply(ast.literal_eval)

    # -------------------------------------------------
    # appid 타입 통일
    # -------------------------------------------------

    df["appid"] = df["appid"].astype(str)

    # -------------------------------------------------
    # 출시 연도 생성
    # -------------------------------------------------

    if "release_date" in df.columns:

        df["release_year"] = pd.to_datetime(
            df["release_date"],
            errors="coerce"
        ).dt.year

    return df


game_df = load_game_metadata()

merged = score_df.merge(
    game_df,
    on="appid",
    how="left"
)

merged = merged.dropna(
    subset=["app_name"]
)

# =====================================================
# 3-1. Steam 인기 랭킹 점수 생성 (추가된 코드)
# =====================================================

merged["rank_score"] = (
    merged["positive_ratio"] * 60
    + np.log1p(merged["total_review_count"]) * 10
)

# =====================================================
# 태그 기반 추천 시스템 준비
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

# =====================================================
# 4. 이미지 API
# =====================================================

@st.cache_data
def get_header_image(appid):
    url = f"https://store.steampowered.com/api/appdetails?appids={appid}&l=english"
    try:
        res = requests.get(url, timeout=5)
        data = res.json()
        if not data.get(str(appid), {}).get("success"):
            return None
        return data[str(appid)]["data"].get("header_image")
    except:
        return None

# =====================================================
# 5. 장르 설정 
# =====================================================

GENRE_TRANSLATION = {
    "Action": "액션",
    "Adventure": "어드벤처",
    "RPG": "RPG",
    "Strategy": "전략",
    "Simulation": "시뮬레이션",
    "Sports": "스포츠",
    "Racing": "레이싱",
    "Indie": "인디",
    "Casual": "캐주얼",
    "Horror": "공포",
    "Shooter": "슈터",
    "Survival": "생존",
    "Sandbox": "샌드박스",
    "Open World": "오픈월드",
    "Platformer": "플랫폼",
    "Fighting": "격투",
    "Puzzle": "퍼즐",
    "MMORPG": "MMORPG",
    "Massively Multiplayer": "대규모 멀티플레이"
}

# =====================================================
# 장르 보정 매핑 (데이터에 없는 장르 처리)
# =====================================================

GENRE_ALIAS = {
    "Shooter": ["Action"],
    "Fighting": ["Action"],
    "Platformer": ["Action","Adventure"],
    "Horror": ["Adventure","Indie"],
    "Puzzle": ["Indie","Casual"],
    "Open World": ["Adventure","RPG"],
    "Survival": ["Adventure","Simulation"],
    "Sandbox": ["Simulation","Indie"]
}

def get_genre_list():

    genre_set = set()

    # 데이터에서 장르 수집
    for genres in merged["genres"].dropna():

        if isinstance(genres, list):

            for g in genres:

                if g in GENRE_TRANSLATION:
                    genre_set.add(g)

    # 기본 장르도 항상 포함
    for g in GENRE_TRANSLATION.keys():
        genre_set.add(g)

    # Steam 인기 장르 강제 추가
    genre_set.add("Survival")
    genre_set.add("Sandbox")

    return sorted(list(genre_set))


genre_list = get_genre_list()

# =====================================================
# 6. 추천 이유 생성
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
# 6-1. 예상 플레이 만족도 생성
# =====================================================

def generate_play_satisfaction(row):

    if row["final_score"] >= merged["final_score"].quantile(0.9):
        return "⭐ 높은 리뷰 평가를 받은 인기 게임입니다."

    elif row["match_score"] >= 2:
        return "⭐ 선택한 장르와 높은 일치도를 보이는 추천 게임입니다."

    elif row["total_review_count"] >= merged["total_review_count"].quantile(0.7):
        return "⭐ 많은 유저가 플레이한 신뢰도 높은 게임입니다."

    else:
        return "⭐ 안정적인 평가를 받은 추천 게임입니다."
    
# =====================================================
# 유사 게임 추천 함수
# =====================================================

def get_similar_games(appid, top_n=5):

    # feature가 없는 경우 (태그 데이터 없음)
    if tag_df.shape[1] == 0:
        return None

    # appid가 없는 경우
    if appid not in tag_df.index:
        return None

    target_vector = tag_df.loc[appid].values.reshape(1, -1)

    similarity = cosine_similarity(
        target_vector,
        tag_df.values
    )[0]

    sim_df = pd.DataFrame({
        "appid": tag_df.index,
        "similarity": similarity
    })

    # 자기 자신 제거
    sim_df = sim_df[sim_df["appid"] != appid]

    # 유사도 높은 순 정렬
    sim_df = sim_df.sort_values(
        "similarity",
        ascending=False
    ).head(top_n)

    if sim_df.empty:
        return None

    return merged[merged["appid"].isin(sim_df["appid"])]

if page == "🏠 Home":

    st.divider()

    st.write(
        """
        Steam 리뷰 데이터를 기반으로  
        **인기 게임 / 장르 추천 / 게임 검색** 기능을 제공합니다.
        """
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("🔥 인기 게임\n\nSteam 리뷰 기반 인기 게임")

    with col2:
        st.success("🎯 장르 추천\n\n선택한 장르 기반 추천")

    with col3:
        st.warning("🔍 게임 검색\n\n내가 아는 게임 점수 확인")
    
# =====================================================
# 인기 게임 페이지
# =====================================================

if page == "🔥 인기 게임":

    st.divider()
    st.header("🔥 Steam 인기 게임 TOP10")

    top_games = merged.sort_values(
        by="rank_score",
        ascending=False
    ).head(10)

    for i, row in enumerate(top_games.itertuples(), 1):

        col1, col2 = st.columns([1,3])

        with col1:
            image_url = get_header_image(row.appid)

            if image_url:
                st.image(image_url)

        with col2:

            steam_url = f"https://store.steampowered.com/app/{row.appid}"

            st.markdown(
                f"### {i}. [{row.app_name}]({steam_url})"
            )

            st.write(f"⭐ 긍정 비율: {round(row.positive_ratio*100,1)}%")
            st.write(f"📝 리뷰 수: {row.total_review_count}")
            st.write(f"🏆 인기 점수: {round(row.rank_score,2)}")


# =====================================================
# 장르 추천 페이지
# =====================================================

if page == "🎯 장르 추천":

    st.divider()
    st.header("🎮 장르별 인기 게임")

    selected_genre = st.selectbox(
        "장르 선택",
        genre_list
    )

    # -----------------------------
    # 장르 매칭 함수
    # -----------------------------
    def genre_match(row):

        genres = row["genres"]
        tags = row["tag_list"]

        # genres 검사
        if isinstance(genres, list):
            if selected_genre in genres:
                return 1

        # tags 검사
        if isinstance(tags, list):

            tags_lower = [t.lower() for t in tags if isinstance(t, str)]

            if selected_genre.lower() in tags_lower:
                return 1

        return 0


    # -----------------------------
    # 장르 필터 적용
    # -----------------------------
    genre_games = merged.copy()

    genre_games["genre_match"] = genre_games.apply(
        genre_match,
        axis=1
    )

    genre_games = genre_games[
        genre_games["genre_match"] == 1
    ]


    # -----------------------------
    # 결과 없을 때
    # -----------------------------
    if genre_games.empty:

        st.warning("해당 장르의 게임이 충분하지 않습니다.")

    else:

        top_genre_games = genre_games.sort_values(
            by="rank_score",
            ascending=False
        ).head(5)


        for row in top_genre_games.itertuples():

            col1, col2 = st.columns([1,3])

            with col1:

                image_url = get_header_image(row.appid)

                if image_url:
                    st.image(image_url)

            with col2:

                steam_url = f"https://store.steampowered.com/app/{row.appid}"

                st.markdown(
                    f"### [{row.app_name}]({steam_url})"
                )

                st.write(f"⭐ 긍정 비율: {round(row.positive_ratio*100,1)}%")
                st.write(f"📝 리뷰 수: {row.total_review_count}")
                st.write(f"🏆 인기 점수: {round(row.rank_score,2)}")

                if hasattr(row, "release_year") and not pd.isna(row.release_year):
                    st.write(f"📅 출시 연도: {int(row.release_year)}")

# =====================================================
# 7. 장르 선택 UI
# =====================================================

st.sidebar.header("장르 선택")

display_genres = [
    f"{g} ({GENRE_TRANSLATION.get(g, g)})"
    for g in genre_list
]

selected_display = st.sidebar.multiselect(
    "Steam 공식 장르 중 선택하세요",
    display_genres,
    default=st.session_state.selected_genres_state
)

if st.sidebar.button("추천 받기"):
    if not selected_display:
        st.sidebar.warning("장르를 하나 이상 선택해주세요.")
    else:
        st.session_state.selected_genres_state = selected_display
        st.session_state.recommend_triggered = True
        st.rerun()

# =====================================================
# 성인 게임 필터
# =====================================================

show_adult_games = st.sidebar.checkbox(
    "🔞 성인 게임 포함",
    value=False
)

# =====================================================
# 성인 게임 태그
# =====================================================

ADULT_TAGS = [
    "Hentai",
    "Sex",
    "Nudity",
    "NSFW",
    "Adult",
    "Sexual Content",
    "Erotic",
    "Mature",
    "Explicit"
]

# =====================================================
# 8. 제외 목록 UI
# =====================================================

st.sidebar.markdown("### 현재 제외된 게임")

if len(st.session_state.excluded_games) == 0:
    st.sidebar.info("제외된 게임이 없습니다.")
else:
    for game in st.session_state.excluded_games:
        col1, col2 = st.sidebar.columns([4, 1])
        with col1:
            st.write(game)
        with col2:
            if st.button("❌", key=f"remove_{game}"):
                st.session_state.excluded_games.remove(game)
                st.rerun()

    if st.sidebar.button("🔄 제외 목록 전체 초기화"):
        st.session_state.excluded_games = []
        st.rerun()

# =====================================================
# 9. 추천 실행
# =====================================================

if page == "🎯 장르 추천" and st.session_state.recommend_triggered:

    selected_genres = [
        g.split(" (")[0]
        for g in st.session_state.selected_genres_state
    ]

    def match_score(genres):

        if not isinstance(genres, list):
            return 0

        lower = [g.lower() for g in genres]

        score = 0

        for g in selected_genres:

            g_lower = g.lower()

            if g_lower in lower:
                score += 1

            # 장르 보정
            if g in GENRE_ALIAS:
                for alt in GENRE_ALIAS[g]:
                    if alt.lower() in lower:
                        score += 1

        return score


    temp_df = merged.copy()

    temp_df["match_score"] = temp_df["genres"].apply(match_score)

    genre_games = temp_df[temp_df["match_score"] > 0].copy()

    # -------------------------------------------------
    # 성인 게임 필터
    # -------------------------------------------------

    if not show_adult_games:

        genre_games = genre_games[
            ~genre_games["tag_list"].apply(
                lambda tags: any(tag in ADULT_TAGS for tag in tags if isinstance(tag, str))
                if isinstance(tags, list) else False
            )
        ]

    # -------------------------------------------------
    # 제외된 게임 필터
    # -------------------------------------------------

    if st.session_state.excluded_games:

        genre_games = genre_games[
            ~genre_games["app_name"].isin(st.session_state.excluded_games)
        ]

    if genre_games.empty:
        st.error("조건에 맞는 게임이 없습니다.")
        st.stop()

    # -------------------------------------------------
    # 추천 점수 계산
    # -------------------------------------------------

    genre_games["service_score"] = (
        genre_games["final_score"] * 0.7
        + genre_games["match_score"] * 25
    )

    # -------------------------------------------------
    # 상위 후보 추출
    # -------------------------------------------------

    top_candidates = genre_games.sort_values(
    "service_score",
    ascending=False
    ).head(20)

    # -------------------------------------------------
    # TOP5 추천
    # -------------------------------------------------

    top3 = top_candidates.head(3)

    random_pool = top_candidates.iloc[3:15]

    if len(random_pool) >= 2:
        random2 = random_pool.sample(2)
    else:
        random2 = random_pool

    top5 = pd.concat([top3, random2])

    st.subheader("🏆 추천 결과 TOP5")

    for i, (_, row) in enumerate(top5.iterrows(), 1):

        st.markdown('<div class="game-card">', unsafe_allow_html=True)

        col1, col2 = st.columns([1,3])

        with col1:

            image_url = get_header_image(row["appid"])

            if image_url:
                st.image(image_url, use_container_width=True)

        # -----------------------------
        # 게임 정보
        # -----------------------------
        with col2:

            steam_url = f"https://store.steampowered.com/app/{row['appid']}"

            st.markdown(
                f"### {i}. 🎮 <a href='{steam_url}' target='_blank'>{row['app_name']}</a>",
                unsafe_allow_html=True
            )

            info_col1, info_col2 = st.columns(2)

            with info_col1:

                st.write(f"🏆 추천 점수: {round(row['service_score'],2)}")

                st.write(
                    f"⭐ 긍정률: {round(row['positive_ratio']*100,1)}%"
                )

                st.write(
                    f"📝 리뷰 수: {row['total_review_count']}"
                )

            with info_col2:

                if "release_year" in row and not pd.isna(row["release_year"]):
                    st.write(f"📅 출시 연도: {int(row['release_year'])}")

                st.write(
                    f"🎮 장르: {', '.join(row['genres'])}"
                )

                st.write(
                    f"🎯 장르 일치: {row['match_score']}개"
                )

            st.write(f"💡 추천 이유: {generate_reason(row)}")

            st.info(generate_play_satisfaction(row))

            # -----------------------------
            # 버튼 영역
            # -----------------------------

            btn_col1, btn_col2 = st.columns(2)

            with btn_col1:

                if st.button(
                    "🔎 비슷한 게임 보기",
                    key=f"similar_{row['appid']}"
                ):
                    st.session_state["selected_game"] = row["appid"]
                    st.rerun()

            with btn_col2:

                if row["app_name"] not in st.session_state.excluded_games:

                    if st.button(
                        "❌ 추천에서 제외",
                        key=f"exclude_{row['appid']}"
                    ):

                        st.session_state.excluded_games.append(row["app_name"])
                        st.rerun()

                else:
                    st.info("이미 제외된 게임")

        st.divider()

# =====================================================
# 10. 게임 점수 자동완성 검색 기능 (Steam 스타일)
# =====================================================

if page == "🔍 게임 검색":

    st.divider()
    st.header("🔍 내가 아는 게임 점수 확인")

    search_name = st.text_input(
        "게임 이름을 입력하세요 (일부만 입력해도 검색됩니다)"
    )

# =====================================================
# Steam 검색 API
# =====================================================

def steam_search_api(keyword):

    url = "https://store.steampowered.com/api/storesearch/"

    params = {
        "term": keyword,
        "l": "koreana",
        "cc": "KR"
    }

    try:
        res = requests.get(url, params=params, timeout=5)
        data = res.json()
        return data.get("items", [])

    except:
        return []

# =====================================================
# 검색 결과 생성
# =====================================================

results = []

if search_name:

    # -----------------------------
    # 1️⃣ 내부 DB 검색
    # -----------------------------

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

    # -----------------------------
    # 2️⃣ Steam API 검색
    # -----------------------------

    steam_results = steam_search_api(search_name)

    for item in steam_results[:5]:

        results.append({
            "name": item["name"],
            "appid": str(item["id"]),
            "image": item["tiny_image"],
            "source": "steam"
        })

    # -------------------------------------------------
    # 중복 제거 (appid 기준)
    # -------------------------------------------------

    unique_results = {}

    for item in results:
        unique_results[item["appid"]] = item

    results = list(unique_results.values())

# =====================================================
# 검색 결과 표시
# =====================================================

if search_name:

    if not results:

        st.error("검색 결과가 없습니다.")

    else:

        for item in results:

            appid = item["appid"]

            col1, col2 = st.columns([2,3])

            with col1:

                if item["image"]:
                    st.image(item["image"], use_container_width=True)

                if st.button("선택", key=f"img_{appid}_{item['source']}"):

                    st.session_state["selected_game"] = appid
                    st.rerun()

            with col2:

                if st.button(item["name"], key=f"name_{appid}_{item['source']}"):

                    st.session_state["selected_game"] = appid
                    st.rerun()

# =====================================================
# 선택된 게임 상세 표시
# =====================================================

if "selected_game" in st.session_state:

    if st.button("🏠 메인 화면으로 돌아가기"):

        del st.session_state["selected_game"]
        st.rerun()

    appid = st.session_state["selected_game"]

    st.divider()

    image_url = get_header_image(appid)

    if image_url:
        st.image(image_url, use_container_width=True)

    result = merged[merged["appid"] == str(appid)]

    

    # -------------------------------------------------
    # 내부 데이터 존재
    # -------------------------------------------------

    if not result.empty:

        row = result.iloc[0]

        st.success("내부 데이터에서 발견")

        steam_url = f"https://store.steampowered.com/app/{row['appid']}"

        st.markdown(
            f"## 🎮 <a href='{steam_url}' target='_blank'>{row['app_name']}</a>",
            unsafe_allow_html=True
        )

        st.write(f"내부 점수: {round(row['final_score'],2)}")
        st.write(f"긍정 비율: {round(row['positive_ratio']*100,1)}%")
        st.write(f"리뷰 수: {row['total_review_count']}개")

        if "release_year" in row and not pd.isna(row["release_year"]):
            st.write(f"출시 연도: {int(row['release_year'])}")

    # -------------------------------------------------
    # 내부 데이터 없음 → 실시간 리뷰 계산
    # -------------------------------------------------

    else:

        st.info("내부 데이터 없음 → 실시간 리뷰 계산")

        try:

            url = f"https://store.steampowered.com/appreviews/{appid}?json=1&filter=recent"

            res = requests.get(url, timeout=5).json()

            summary = res.get("query_summary", {})

            total = summary.get("total_reviews", 0)
            positive = summary.get("total_positive", 0)

            if total > 0:

                ratio = positive / total
                score = ratio * 70 + np.log1p(total) * 5

                st.write(f"실시간 점수: {round(score,2)}")
                st.write(f"긍정 비율: {round(ratio*100,1)}%")
                st.write(f"최근 리뷰 수: {total}개")

            else:
                st.warning("리뷰 데이터 부족")

        except:

            st.error("실시간 리뷰를 가져올 수 없습니다.")

# 실행코드 : streamlit run app.py