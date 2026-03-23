import requests
import pandas as pd
import time

url = "https://steamspy.com/api.php?request=all"

print("SteamSpy 데이터 가져오는 중...")

data = requests.get(url).json()

rows = []

for appid, game in data.items():

    rows.append({
        "appid": str(appid),
        "app_name": game.get("name"),
        "genres": str(game.get("genre", "")).split(","),
        "steamspy_tags": game.get("tags", "")
    })

df = pd.DataFrame(rows)

# 장르 리스트 정리
df["genres"] = df["genres"].apply(
    lambda x: [g.strip() for g in x if g]
)

# 태그 문자열
df["steamspy_tags"] = df["steamspy_tags"].apply(
    lambda x: ";".join(x.keys()) if isinstance(x, dict) else ""
)

print("총 게임 수:", len(df))

df.to_csv("data/game_metadata_extended.csv", index=False)

print("저장 완료: data/game_metadata_extended.csv")