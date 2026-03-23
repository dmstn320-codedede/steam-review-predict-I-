import glob
import re
import pandas as pd
import requests
import time
import os

print("🔥 appid 수집 시작")

# 1️⃣ xlsx 파일명에서 appid 추출
file_list = glob.glob("data/recent_reviews/*.xlsx")

appids = []

for file in file_list:
    filename = os.path.basename(file)
    numbers = re.findall(r"\d+", filename)
    if numbers:
        appids.append(numbers[-1])

appids = list(set(appids))

print(f"✅ 총 {len(appids)}개 appid 발견\n")


# 2️⃣ Steam API 호출
def get_game_details(appid):
    try:
        url = f"https://store.steampowered.com/api/appdetails?appids={appid}"
        res = requests.get(url, timeout=10)
        data = res.json()

        if not data[str(appid)]["success"]:
            return None

        game = data[str(appid)]["data"]

        return {
            "appid": str(appid),
            "app_name": game.get("name"),
            "genres": [g["description"] for g in game.get("genres", [])]
        }

    except:
        return None


game_data = []

print("🚀 Steam API 메타데이터 수집 시작...\n")

for i, appid in enumerate(appids):

    info = get_game_details(appid)

    if info:
        game_data.append(info)

    if i % 20 == 0:
        print(f"{i}/{len(appids)} 완료")

    time.sleep(0.2)

game_df = pd.DataFrame(game_data)

# 3️⃣ CSV 저장
game_df.to_csv("data/game_metadata.csv", index=False)

print("\n✅ game_metadata.csv 저장 완료")