GENRE_TRANSLATION = {
    "Action": "액션",
    "Adventure": "어드벤처",
    "RPG": "RPG",
    "Strategy": "전략",
    "Simulation": "시뮬레이션",
    "Sports": "스포츠",
    "Racing": "레이싱",
    "Indie": "인디",
    "Casual": "캐주얼",
    "Horror": "공포",
    "Shooter": "슈터",
    "Survival": "생존",
    "Sandbox": "샌드박스",
    "Open World": "오픈월드",
    "Platformer": "플랫폼",
    "Fighting": "격투",
    "Puzzle": "퍼즐",
    "MMORPG": "MMORPG",
    "Massively Multiplayer": "대규모 멀티플레이"
}


GENRE_ALIAS = {
    "Shooter": ["Action"],
    "Fighting": ["Action"],
    "Platformer": ["Action","Adventure"],
    "Horror": ["Adventure","Indie"],
    "Puzzle": ["Indie","Casual"],
    "Open World": ["Adventure","RPG"],
    "Survival": ["Adventure","Simulation"],
    "Sandbox": ["Simulation","Indie"]
}


def get_genre_list(merged):

    genre_set = set()

    for genres in merged["genres"].dropna():

        if isinstance(genres, list):

            for g in genres:

                if g in GENRE_TRANSLATION:
                    genre_set.add(g)

    for g in GENRE_TRANSLATION.keys():
        genre_set.add(g)

    return sorted(list(genre_set))