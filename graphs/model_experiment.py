# =====================================================
# model_experiment.py
# Steam 리뷰 예측 기준 모델 (Logistic Regression)
# =====================================================

# 데이터 처리
import pandas as pd
import numpy as np

# 파일 처리
import glob
import os
import re

# 머신러닝
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 시각화
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm

# =====================================================
# 한글 폰트 설정 (Windows 확실한 방법)
# =====================================================

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm

# 맑은 고딕 폰트 경로
font_path = "C:/Windows/Fonts/malgun.ttf"

# 폰트 등록
font_prop = fm.FontProperties(fname=font_path)

# matplotlib에 적용
plt.rcParams["font.family"] = font_prop.get_name()
plt.rcParams["axes.unicode_minus"] = False

sns.set_style("whitegrid")

# =====================================================
# 1. 리뷰 데이터 로드
# =====================================================

def load_review_data():

    file_list = glob.glob("data/recent_reviews/*.xlsx")

    if not file_list:
        raise FileNotFoundError("recent_reviews 폴더에 xlsx 파일이 없습니다.")

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
# 4. Feature 선택
# =====================================================

X = score_df[["total_review_count", "positive_ratio"]]
y = score_df["label"]

# =====================================================
# 5. 데이터 분할
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =====================================================
# 6. Logistic Regression 학습
# =====================================================

model = LogisticRegression(class_weight="balanced", max_iter=1000)

model.fit(X_train, y_train)

# =====================================================
# 7. 예측 및 평가
# =====================================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("====================================")
print("📊 Logistic Regression 결과")
print("====================================")
print(f"Accuracy: {accuracy:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(report)

print("====================================")

# =====================================================
# 8. 그래프 1: 긍정 / 부정 게임 분포
# =====================================================

plt.figure(figsize=(6,4))

sns.countplot(x=score_df["label"], palette="Set2")

plt.title("게임 리뷰 감정 분포")
plt.xlabel("라벨 (0=부정, 1=긍정)")
plt.ylabel("게임 수")

plt.show()

# =====================================================
# 9. 그래프 2: 리뷰 수 분포
# =====================================================

plt.figure(figsize=(6,4))

sns.histplot(score_df["total_review_count"], bins=30)

plt.title("게임별 리뷰 수 분포")
plt.xlabel("총 리뷰 수")
plt.ylabel("빈도")

plt.show()

# =====================================================
# 실행 방법
# =====================================================

# 터미널에서 실행
# python model_experiment.py