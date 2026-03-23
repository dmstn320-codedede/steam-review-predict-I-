import requests
import pandas as pd
import time

# 기존 메타데이터 불러오기
df = pd.read_csv("data/game_metadata_extended.csv")

def get_tags(appid):
    url = f"https://steamspy.com/api.php?request=appdetails&appid={appid}"

    try:
        res = requests.get(url, timeout=5)
        data = res.json()

        tags = data.get("tags", {})

        if tags:
            return list(tags.keys())

        return []

    except:
        return []

# 태그 수집
tag_list = []

for i, row in df.iterrows():

    appid = row["appid"]
    tags = get_tags(appid)

    tag_list.append(";".join(tags))

    print(f"{i+1}/{len(df)} 완료")

    time.sleep(0.3)  # 중요: API 과부하 방지

# 저장
df["steamspy_tags"] = tag_list
df.to_csv("data/game_metadata_extended.csv", index=False)

print("태그 수집 완료")