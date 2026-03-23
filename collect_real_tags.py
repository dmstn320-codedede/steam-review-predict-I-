import pandas as pd
import requests
import time

df = pd.read_csv("data/game_metadata_extended.csv")

tag_results = []

for i, row in df.iterrows():

    appid = str(row["appid"]).replace(".0","")

    url = f"https://store.steampowered.com/api/appdetails?appids={appid}"

    try:
        res = requests.get(url, timeout=10)
        data = res.json()

        if data.get(appid, {}).get("success"):

            game_data = data[appid]["data"]

            tags = game_data.get("genres", [])

            tag_names = []

            for t in tags:
                tag_names.append(t["description"])

            tag_results.append(";".join(tag_names))

        else:
            tag_results.append("")
            print(f"{appid} 실패")

    except Exception as e:
        print("오류:", e)
        tag_results.append("")

    print(f"{i+1}/{len(df)} 완료")

    time.sleep(0.15)

df["steamspy_tags"] = tag_results

df.to_csv("data/game_metadata_fixed.csv", index=False)

print("태그 수집 완료")