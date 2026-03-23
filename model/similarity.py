from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


def get_similar_games(appid, tag_df, merged):

    # 내부 데이터 없는 게임이면 추천 불가
    if appid not in tag_df.index:
        return None

    # 코사인 유사도 계산
    sim_scores = cosine_similarity(
        tag_df.loc[[appid]],
        tag_df
    )[0]

    sim_df = pd.DataFrame({
        "appid": tag_df.index,
        "similarity": sim_scores
    })

    # 자기 자신 제외
    sim_df = sim_df[sim_df["appid"] != appid]

    # 유사도 높은 순
    sim_df = sim_df.sort_values(
        "similarity",
        ascending=False
    ).head(5)

    recommended = merged[
        merged["appid"].isin(sim_df["appid"])
    ].copy()

    # -------------------
    # 추천 이유 생성
    # -------------------

    base_tags = set(tag_df.columns[tag_df.loc[appid] == 1])

    reasons = []

    for gid in recommended["appid"]:

        game_tags = set(tag_df.columns[tag_df.loc[gid] == 1])

        common_tags = base_tags & game_tags

        if common_tags:
            reason = ", ".join(list(common_tags)[:3])
        else:
            reason = "유사 태그"

        reasons.append(reason)

    recommended["reason"] = reasons

    return recommended