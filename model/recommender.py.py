import pandas as pd

def recommend_games(df, selected_genres):

    def match_score(genres):

        if not isinstance(genres, list):
            return 0

        lower = [g.lower() for g in genres]

        score = 0

        for g in selected_genres:

            if g.lower() in lower:
                score += 1

        return score

    temp_df = df.copy()

    temp_df["match_score"] = temp_df["genres"].apply(match_score)

    genre_games = temp_df[temp_df["match_score"] > 0].copy()

    genre_games["service_score"] = (
        genre_games["final_score"] * 0.7
        + genre_games["match_score"] * 25
    )

    return genre_games.sort_values("service_score", ascending=False)