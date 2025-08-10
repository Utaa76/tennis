import numpy as np
import pandas as pd
import joblib

modelLGBM = joblib.load("modeleLGBM_tennis_2024-20.pkl")
iso = joblib.load("calibrateur-20.pkl")
modelLGBMfinal = joblib.load("modeleLGBM_tennis_2025-20.pkl")
player_stats = joblib.load("players_stats.pkl")
match_df = joblib.load("match_df.pkl")
match_df_test = match_df[match_df['Date'] >= '2024-01-01']
h2h = joblib.load("h2h.pkl")
h2h_surface = joblib.load("h2h_surface.pkl")

# Maps “tour level” & “round”
tournament_level_map = {
    "Grand Slam": 4,
    "Masters 1000": 3,
    "ATP500": 2,
    "ATP250": 1,
}
round_map = {
    '1st Round': 1,
    '2nd Round': 2,
    '3rd Round': 3,
    '4th Round': 4,
    'Quarterfinals': 5,
    'Semifinals': 6,
    'The Final': 7
}

def default_player_stats(player_name):
    return {
        'Player': player_name,
        'matches_played': 0,
        'wins': 0,
        'winrate': 0.5,
        'elo': 1500,
        'elo_hard': 1500,
        'elo_clay': 1500,
        'elo_grass': 1500,
        'elo_carpet': 1500,
        'recent_form': 0.5,
        'mean_clay': 0.5,
        'mean_grass': 0.5,
        'mean_hard': 0.5,
        'recent_games_diff_last': 0,
        'recent_decider_last': 0.5,
        'rank': 500,
        'pts': 0,
        'wrlast50': 0.25,
        'wrlast100': 0.25
    }

def get_player_stats(player_name, player_stats_df):
    if player_name in player_stats:
        return player_stats_df[player_name]
    else:
        return default_player_stats(player_name)

def predict_proba_calibrated(model, X):
    raw = model.predict(X)
    clipped = np.clip(raw, 0, 1)
    calibrated = iso.predict(clipped)
    return calibrated

def remove_margin(odds_list):
    implied_probs = [1/o for o in odds_list]
    total = sum(implied_probs)
    fair_probs = [p/total for p in implied_probs]
    fair_odds = [1/p for p in fair_probs]
    return fair_odds

# 5
def predict_match(
    A, B, surface,
    cote_A, cote_B,
    level_name,                # ex. 'Grand Slam', 'ATP500', …
    round_name,                # ex. 'Quarterfinals', 'Semifinals', …
    rmv_margin=False,
    ev_comparison=False,
    bankroll=100, min_ev=0.05  # mettre 0.1 ?
):
    """
    Renvoie la décision de pari pour un seul match A‑B à `surface`
    avec les cotes Bet365 (ou autres) déjà connues.
    """

    # ---------- mapping tour & round (même qu’off‑line) ----------
    tournament_level_map = {
        "Grand Slam": 4,
        "Masters 1000": 3,
        "ATP500": 2,
        "ATP250": 1,
        "WTA250": 1,
        "WTA500": 2,
        "WTA1000": 3
    }
    round_map = {
        '1st Round': 1, '2nd Round': 2, '3rd Round': 3, '4th Round': 4,
        'Quarterfinals': 5, 'Semifinals': 6, 'The Final': 7
    }

    # ---------- récupérer les stats joueurs ----------
    statsA = get_player_stats(A, player_stats)
    statsB = get_player_stats(B, player_stats)

    # if statsA['elo'] < statsB['elo']:
    #     statsA, statsB = statsB, statsA
    #     A, B = B, A
    #     cote_A, cote_B = cote_B, cote_A

    # ---------- vérifs ----------
    if A not in player_stats:
        print(f"⚠️ Joueur {A} manquant dans les stats. Skip.\n")
        return None
    if B not in player_stats:
        print(f"⚠️ Joueur {B} manquant dans les stats. Skip.\n")
        return None

    df_stats = pd.DataFrame({
        f'{A}': statsA,
        f'{B}': statsB
    })

    # print(df_stats)

    if statsB[f'elo_{surface.lower()}'] > statsA[f'elo_{surface.lower()}']:
        A, B = B, A
        statsA, statsB = statsB, statsA
        cote_A, cote_B = cote_B, cote_A

    # ---------- H2H ----------
    # H2H global
    h2h_A_vs_B = h2h[(A, B)]
    h2h_B_vs_A = h2h[(B, A)]
    total_h2h = h2h_A_vs_B + h2h_B_vs_A
    h2h_diff = ((h2h_A_vs_B - h2h_B_vs_A) / total_h2h) if total_h2h > 0 else 0

    # H2H par surface
    h2h_surf_A_vs_B = h2h_surface[(A, B, surface)]
    h2h_surf_B_vs_A = h2h_surface[(B, A, surface)]
    total_h2h_surf = h2h_surf_A_vs_B + h2h_surf_B_vs_A
    h2h_surface_diff = ((h2h_surf_A_vs_B - h2h_surf_B_vs_A) / total_h2h_surf) if total_h2h_surf > 0 else 0

    lst = remove_margin([cote_A, cote_B])
    
    oddA, oddB = lst[0], lst[1]
    bookOddA, bookOddB = cote_A, cote_B

    # ---------- features ----------
    log_odds_ratio = np.log(oddB / oddA) * 5 if oddA > 0 and oddB > 0 else 0
    feat = pd.DataFrame([{
        'elo_diff': statsA['elo'] - statsB['elo'],
        'elo_surface_diff': statsA.get(f'elo_{surface.lower()}', 1500) -
                            statsB.get(f'elo_{surface.lower()}', 1500),
        'winrate_diff': statsA['winrate'] - statsB['winrate'],
        'surface_winrate_diff': statsA.get(f'mean_{surface.lower()}', 0) -
                                statsB.get(f'mean_{surface.lower()}', 0),
        'wrlast50_diff': statsA['wrlast50'] - statsB['wrlast50'],
        'wrlast100_diff': statsA['wrlast100'] - statsB['wrlast100'],
        'h2h_diff': h2h_diff,
        'h2h_surface_diff': h2h_surface_diff,
        'odds-ratio': log_odds_ratio,
        'level': tournament_level_map.get(level_name, 1),
        'round': round_map.get(round_name, 1),
        'rank_diff': statsA['rank'] - statsB['rank'],
        'pts_diff': statsA['pts'] - statsB['pts'],
        'games_diff_recent': statsA['recent_games_diff_last'] - statsB['recent_games_diff_last'],
        'decider_winrate_diff': statsA['recent_decider_last'] - statsB['recent_decider_last']
    }]).astype(float)

    if not rmv_margin:
        oddA, oddB = cote_A, cote_B

    # print(feat)

    # ---------- probabilités modèles ----------
    # probaLGBM = modelLGBMfinal.predict_proba(feat)[0, 1] # mettre modelLGBMfinal !!!
    probaLGBM = predict_proba_calibrated(modelLGBM, feat)[0]
    
    probA = probaLGBM
    probB = 1 - probA

    evA = (probA * (oddA - 1)) - (1 - probA)
    evB = (probB * (oddB - 1)) - (1 - probB)

    if (ev_comparison):
      if evA > evB:
          winner, odd, cote, p_win = A, oddA, bookOddA, probaLGBM
      else:
          winner, odd, cote, p_win = B, oddB, bookOddB, 1 - probaLGBM
    else:
        if probA > 0.5:
            winner, odd, cote, p_win = A, oddA, bookOddA, probaLGBM
        else:
            winner, odd, cote, p_win = B, oddB, bookOddB, 1 - probaLGBM

    print(f"🎾 \033[1m{A} vs {B}\033[0m ({surface}) — pari : \033[4m{winner}\033[0m 🏆")
    print(f"\t📊 Probabilité modèle : {p_win:.2%} | Cote : {cote}")

    # if (p_win < 0.54):
    #     print("\tProbabilité trop proche de 50%, pas de pari.")
    #     return None

    ev = (p_win * (odd - 1)) - (1 - p_win)
    if ev < min_ev:
        print(f"\t⚠️ \033[91m EV {ev:.4f} < {min_ev}, aucun pari. 🚫\033[0m")
        print("\n")
        return None

    # ---------- Kelly ----------
    b = odd - 1
    fraction_kelly = max((b * p_win - (1 - p_win)) / b, 0)
    mise = fraction_kelly * bankroll * 0.25
    if mise < 0.1:
        print(f"\t⚠️ \033[91m Mise ({mise:.2f} €) trop faible, aucun pari. 💸\033[0m")
        print("\n")
        return None

    gain_attendu = (cote - 1) * mise

    # ---------- affichage ----------
    print(f"\t\t💰 EV : {ev:.4f} | Mise (Kelly) : \033[1;32m{mise:.2f} €\033[0m")
    print(f"\t\t🎯 Gain net attendu : \033[1;32m{gain_attendu:.2f} €\033[0m")
    print("\n")

    return {
        'match': f"{A} vs {B}",
        'winner': winner,
        'surface': surface,
        'probability': p_win,
        'cote': cote,
        'expected_value': ev,
        'mise': mise,
        'gain_attendu': gain_attendu,
        'round': round_name,
        'level': level_name
    }


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/")
def read_root():
    return {"message": "API Tennis Elo est en ligne."}

class MatchInput(BaseModel):
    A: str
    B: str
    surface: str
    cote_A: float
    cote_B: float
    level_name: str
    round_name: str
    bankroll: float = 100
    min_ev: float = 0.05

@app.post("/predict")
def predict(match: MatchInput):
    # Appelle ta fonction predict_match avec les paramètres reçus
    result = predict_match(
        match.A, match.B, match.surface,
        match.cote_A, match.cote_B,
        match.level_name, match.round_name,
        bankroll=match.bankroll, min_ev=match.min_ev,
        ev_comparison=True
    )
    if result is None:
        return {"message": "Pas de pari recommandé pour ce match."}
    return result

"""
	'match': f"{A} vs {B}",
	'winner': winner,
	'surface': surface,
	'probability': p_win,
	'cote': cote,
	'expected_value': ev,
	'mise': mise,
	'gain_attendu': gain_attendu,
	'round': round_name,
	'level': level_name
"""