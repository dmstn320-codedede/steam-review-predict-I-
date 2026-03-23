# =====================================================
# model_compare.py
# Steam 리뷰 예측 모델 비교
# =====================================================

import pandas as pd
import numpy as np
import glob
import os
import re

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

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
# 2. 게임별 점수 집계
# =====================================================

score_df = review_df.groupby("appid").agg(
    positive_ratio=("recommended", "mean"),
    total_review_count=("recommended", "count")
).reset_index()

# =====================================================
# 3. Label 생성
# =====================================================

score_df["label"] = (score_df["positive_ratio"] >= 0.7).astype(int)

print("===== Label 분포 =====")
print(score_df["label"].value_counts())
print("======================")

# =====================================================
# 4. Feature
# =====================================================

X = score_df[["total_review_count", "positive_ratio"]]
y = score_df["label"]

# =====================================================
# 5. Train/Test Split
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =====================================================
# 6. 모델 정의
# =====================================================

models = {

    "Logistic Regression":
        LogisticRegression(class_weight="balanced", max_iter=1000),

    "Decision Tree":
        DecisionTreeClassifier(max_depth=5, random_state=42),

    "Random Forest":
        RandomForestClassifier(n_estimators=100, random_state=42)

}

results = []

# =====================================================
# 7. 모델 학습 및 평가
# =====================================================

for name, model in models.items():

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)

    results.append((name, acc))

    print(f"{name} Accuracy: {acc:.4f}")

# =====================================================
# 8. 결과 테이블
# =====================================================

result_df = pd.DataFrame(results, columns=["Model", "Accuracy"])

print("\n모델 비교 결과")
print(result_df)

# =====================================================
# 9. Random Forest Feature 중요도
# =====================================================

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

importance = rf_model.feature_importances_

feature_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

print("\nFeature 중요도")
print(feature_df)

# =====================================================
# 10. Feature 중요도 그래프
# =====================================================

plt.figure(figsize=(6,4))

sns.barplot(
    data=feature_df,
    x="Feature",
    y="Importance"
)

plt.title("Feature Importance (Random Forest)")
plt.xlabel("Feature")
plt.ylabel("Importance")

plt.show()