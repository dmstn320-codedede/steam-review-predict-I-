# =====================================================
# Steam 리뷰 예측 모델 비교 (Advanced)
# =====================================================

import pandas as pd
import numpy as np
import glob
import os
import re

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================
# 1. 리뷰 데이터 로드
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

    return review_df


review_df = load_review_data()

# =====================================================
# 2. 게임별 리뷰 집계
# =====================================================

score_df = review_df.groupby("appid").agg(
    positive_ratio=("recommended", "mean"),
    total_review_count=("recommended", "count")
).reset_index()

# =====================================================
# 3. 메타데이터 로드
# =====================================================

meta_df = pd.read_csv("data/game_metadata.csv")

meta_df["appid"] = meta_df["appid"].astype(str)

# =====================================================
# 4. 데이터 합치기
# =====================================================

df = score_df.merge(meta_df, on="appid", how="left")

# =====================================================
# 5. Feature Engineering
# =====================================================

# 로그 리뷰 수
df["log_review_count"] = np.log1p(df["total_review_count"])

# price 결측 처리
if "price" in df.columns:
    df["price"] = df["price"].fillna(0)

# =====================================================
# 6. 장르 One-hot encoding
# =====================================================

if "genres" in df.columns:

    df["genres"] = df["genres"].fillna("Unknown")

    genre_dummies = df["genres"].str.get_dummies(sep=",")

    df = pd.concat([df, genre_dummies], axis=1)

# =====================================================
# 7. Label 생성
# =====================================================

df["label"] = (df["positive_ratio"] >= 0.7).astype(int)

print("Label 분포")
print(df["label"].value_counts())

# =====================================================
# 8. Feature 선택
# =====================================================

features = [
    "positive_ratio",
    "total_review_count",
    "log_review_count"
]

# price 있으면 추가
if "price" in df.columns:
    features.append("price")

# 숫자형 장르 컬럼만 추가
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

genre_features = [
    col for col in numeric_cols
    if col not in features and col not in ["label"]
]

features += genre_features

X = df[features]
y = df["label"]

# =====================================================
# 9. Train Test Split
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =====================================================
# 10. 모델 정의
# =====================================================

models = {

    "Logistic Regression":
        LogisticRegression(max_iter=1000),

    "Decision Tree":
        DecisionTreeClassifier(max_depth=6),

    "Random Forest":
        RandomForestClassifier(n_estimators=100)

}

results = []

# =====================================================
# 11. 모델 학습
# =====================================================

for name, model in models.items():

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)

    results.append((name, acc))

    print(f"{name} Accuracy: {acc:.4f}")

# =====================================================
# 12. 결과 테이블
# =====================================================

result_df = pd.DataFrame(results, columns=["Model","Accuracy"])

print("\n모델 비교 결과")
print(result_df)

# =====================================================
# 13. Feature Importance
# =====================================================

rf = RandomForestClassifier(n_estimators=100)

rf.fit(X_train, y_train)

importance = rf.feature_importances_

importance_df = pd.DataFrame({

    "Feature": X.columns,
    "Importance": importance

}).sort_values("Importance", ascending=False)

print("\nFeature Importance")
print(importance_df.head(10))

# =====================================================
# 14. Feature Importance 그래프
# =====================================================

plt.figure(figsize=(8,5))

sns.barplot(
    data=importance_df.head(10),
    x="Importance",
    y="Feature"
)

plt.title("Top Feature Importance (Random Forest)")
plt.xlabel("Importance")
plt.ylabel("Feature")

plt.show()