import requests


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