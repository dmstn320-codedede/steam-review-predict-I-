import pandas as pd
import numpy as np
import glob
import os
import re
import ast


# =====================================================
# 리뷰 데이터 로드
# =====================================================

def load_review_data():

    file_list = glob.glob("data/recent_reviews/*.xlsx")

    review_list = []

    for file in file_list:

        temp = pd.read_excel(file)

        filename = os.path.basename(file)
        numbers = re.findall(r"\d+", filename)

        if numbers:
            temp["appid"] = str(numbers[-1])

        review_list.append(temp)

    review_df = pd.concat(review_list, ignore_index=True)

    if "voted_up" in review_df.columns:
        review_df = review_df.rename(columns={"voted_up": "recommended"})

    review_df["recommended"] = review_df["recommended"].astype(int)
    review_df["appid"] = review_df["appid"].astype(str)

    print("✅ 리뷰 데이터 로드 완료:", len(review_df))

    return review_df


# =====================================================
# 게임 메타데이터 로드 (🔥 디버깅 포함)
# =====================================================

def load_game_metadata():

    print("\n============================")
    print("📦 CSV 로딩 시작")
    print("============================")

    df = pd.read_csv(
        "data/game_metadata_extended.csv",
        encoding="utf-8",
        engine="python"
    )

    # 컬럼 공백 제거
    df.columns = df.columns.str.strip()

    print("📌 컬럼 목록:", df.columns.tolist())

    # steamspy_tags 확인
    if "steamspy_tags" not in df.columns:
        print("❌ steamspy_tags 컬럼 없음")
        df["tag_list"] = [[] for _ in range(len(df))]
        return df

    print("\n📌 steamspy_tags 샘플:")
    print(df["steamspy_tags"].head(5))

    # 태그 처리
    df["steamspy_tags"] = df["steamspy_tags"].fillna("")

    df["tag_list"] = df["steamspy_tags"].apply(
        lambda x: [t.strip() for t in str(x).split(";") if t]
    )

    print("\n📌 tag_list 샘플:")
    print(df["tag_list"].head(5))

    # 장르 처리
    if "genres" in df.columns:
        df["genres"] = df["genres"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else []
        )
    else:
        df["genres"] = [[] for _ in range(len(df))]

    df["appid"] = df["appid"].astype(str)

    return df


# =====================================================
# 최종 데이터셋
# =====================================================

def build_dataset():

    print("\n🚀 데이터셋 생성 시작")

    review_df = load_review_data()
    game_df = load_game_metadata()

    score_df = review_df.groupby("appid").agg(
        positive_ratio=("recommended", "mean"),
        total_review_count=("recommended", "count")
    ).reset_index()

    print("📊 score_df 생성 완료")

    merged = game_df.merge(score_df, on="appid", how="left")

    merged["positive_ratio"] = merged["positive_ratio"].fillna(0)
    merged["total_review_count"] = merged["total_review_count"].fillna(0)

    merged = merged.dropna(subset=["app_name"])

    merged["rank_score"] = (
        merged["positive_ratio"] * 60
        + np.log1p(merged["total_review_count"]) * 10
    )

    print("\n🎯 최종 merged 확인")
    print("tag_list 샘플:")
    print(merged["tag_list"].head(5))

    print("\n🚀 데이터셋 생성 완료\n")

    return merged